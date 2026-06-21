"""RAG evaluation pipeline for code_chunks retrieval.

Usage:
    python -m eval.run --config default
    python -m eval.run --config reranker --skip-llm-judge
    python -m eval.run --config dense_only --dataset eval/datasets/code_retrieval_v1.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import typer
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client.models import SparseVector

from agents.tracing import get_langfuse
from eval.dataset import EvalDataset, EvalEntry
from eval.metrics import context_precision, mrr, ndcg_at_k, recall_at_k
from rag.collections import get_qdrant_client
from rag.embedder import VoyageEmbedder
from rag.models import SearchHit
from rag.retriever import HybridRetriever, RetrievalConfig

load_dotenv()

# ---------------------------------------------------------------------------
# Retrieval configs
# ---------------------------------------------------------------------------

EVAL_CONFIGS: dict[str, RetrievalConfig] = {
    "default": RetrievalConfig(hybrid_alpha=0.5, retrieve_k=20, rerank_k=5, use_reranker=False),
    "reranker": RetrievalConfig(hybrid_alpha=0.5, retrieve_k=20, rerank_k=5, use_reranker=True),
    "dense_only": RetrievalConfig(hybrid_alpha=1.0, retrieve_k=20, rerank_k=5, use_reranker=False),
    "sparse_only": RetrievalConfig(hybrid_alpha=0.0, retrieve_k=20, rerank_k=5, use_reranker=False),
}

_RECALL_THRESHOLD = 0.82
_TOP_K = 5
_DEFAULT_DATASET = Path(__file__).parent / "datasets" / "code_retrieval_v1.jsonl"
_REPORTS_DIR = Path(__file__).parent / "reports"
_COLLECTION = "code_chunks"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query: str
    query_type: str
    difficulty: str
    ground_truth_chunk_ids: list[str]
    retrieved_ids: list[str]
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    context_precision: float | None = None


@dataclass
class RunSummary:
    run_id: str
    config_name: str
    collection: str
    dataset_name: str
    run_at: datetime
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    context_precision: float | None
    query_count: int
    query_results: list[QueryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-embedding
# ---------------------------------------------------------------------------


@dataclass
class _PrecomputedQuery:
    entry: EvalEntry
    dense_vec: list[float]
    sparse_vec: SparseVector
    hits: list[SearchHit] = field(default_factory=list)


async def _precompute_embeddings(
    entries: list[EvalEntry], embedder: VoyageEmbedder
) -> list[_PrecomputedQuery]:
    logger.info("Pre-embedding {} queries via Voyage (this may take ~{}min) ...",
                len(entries), round(len(entries) / 8 * 22 / 60, 1))
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    texts = [e.query for e in entries]
    dense_vecs = await embedder.embed_queries(texts)
    sparse_vecs = [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in bm25.embed(texts)
    ]
    return [
        _PrecomputedQuery(entry=e, dense_vec=dv, sparse_vec=sv)
        for e, dv, sv in zip(entries, dense_vecs, sparse_vecs)
    ]


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------


async def _run_eval(
    config_name: str,
    config: RetrievalConfig,
    precomputed: list[_PrecomputedQuery],
    run_id: str,
    skip_llm_judge: bool,
    dataset_name: str,
) -> RunSummary:
    qdrant = get_qdrant_client()
    retriever = HybridRetriever(
        collection=_COLLECTION,
        alpha=config.hybrid_alpha,
        client=qdrant,
    )

    if config.use_reranker:
        from rag.reranker import _get_reranker
        _get_reranker()  # warm up before the timed loop

    query_results: list[QueryResult] = []

    for pq in precomputed:
        raw_hits = await retriever.search_vectors(
            dense_vec=pq.dense_vec,
            sparse_vec=pq.sparse_vec,
            query_label=pq.entry.query,
            top_k=config.retrieve_k,
        )

        if config.use_reranker and raw_hits:
            from rag.reranker import rerank
            final_hits = await rerank(pq.entry.query, raw_hits, config.rerank_k)
        else:
            final_hits = raw_hits[:_TOP_K]

        pq.hits = final_hits  # store for Langfuse task fn
        retrieved_ids = [h.id for h in final_hits]
        gt = set(pq.entry.ground_truth_chunk_ids)

        cp: float | None = None
        if not skip_llm_judge:
            cp = await context_precision(pq.entry.query, final_hits, run_id=run_id)

        query_results.append(
            QueryResult(
                query=pq.entry.query,
                query_type=pq.entry.query_type,
                difficulty=pq.entry.difficulty,
                ground_truth_chunk_ids=pq.entry.ground_truth_chunk_ids,
                retrieved_ids=retrieved_ids,
                recall_at_5=recall_at_k(retrieved_ids, gt, _TOP_K),
                mrr=mrr(retrieved_ids, gt),
                ndcg_at_5=ndcg_at_k(retrieved_ids, gt, _TOP_K),
                context_precision=cp,
            )
        )

    n = len(query_results)
    avg_recall = sum(r.recall_at_5 for r in query_results) / n
    avg_mrr = sum(r.mrr for r in query_results) / n
    avg_ndcg = sum(r.ndcg_at_5 for r in query_results) / n
    cp_vals = [r.context_precision for r in query_results if r.context_precision is not None]
    avg_cp = sum(cp_vals) / len(cp_vals) if cp_vals else None

    return RunSummary(
        run_id=run_id,
        config_name=config_name,
        collection=_COLLECTION,
        dataset_name=dataset_name,
        run_at=datetime.now(timezone.utc),
        recall_at_5=avg_recall,
        mrr=avg_mrr,
        ndcg_at_5=avg_ndcg,
        context_precision=avg_cp,
        query_count=n,
        query_results=query_results,
    )


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS code_retrieval_eval_runs (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              UUID NOT NULL,
    run_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    config_name         TEXT NOT NULL,
    collection          TEXT NOT NULL,
    dataset_name        TEXT NOT NULL,
    recall_at_5         FLOAT NOT NULL,
    mrr                 FLOAT NOT NULL,
    ndcg_at_5           FLOAT NOT NULL,
    context_precision   FLOAT,
    query_count         INT NOT NULL,
    details             JSONB NOT NULL
)
"""


async def _save_to_postgres(summary: RunSummary) -> int | None:
    dsn = (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'migration')}"
        f":{os.environ.get('POSTGRES_PASSWORD', '')}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ.get('POSTGRES_DB', 'migration_agent')}"
    )
    try:
        conn: asyncpg.Connection = await asyncpg.connect(dsn)
    except Exception as exc:
        logger.warning("Cannot connect to Postgres, skipping eval save: {}", exc)
        return None

    try:
        await conn.execute(_CREATE_TABLE)
        details = [asdict(r) for r in summary.query_results]
        row = await conn.fetchrow(
            """
            INSERT INTO code_retrieval_eval_runs
                (run_id, run_at, config_name, collection, dataset_name,
                 recall_at_5, mrr, ndcg_at_5, context_precision, query_count, details)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
            """,
            uuid.UUID(summary.run_id),
            summary.run_at,
            summary.config_name,
            summary.collection,
            summary.dataset_name,
            summary.recall_at_5,
            summary.mrr,
            summary.ndcg_at_5,
            summary.context_precision,
            summary.query_count,
            json.dumps(details),
        )
        row_id: int = row["id"]  # type: ignore[index]
        logger.info("Saved to Postgres: code_retrieval_eval_runs id={}", row_id)
        return row_id
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_report(summary: RunSummary) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.run_at.strftime("%Y%m%d_%H%M%S")
    path = _REPORTS_DIR / f"{summary.config_name}_{ts}.txt"

    # Per query-type and difficulty breakdowns
    def _group(key: str) -> dict[str, list[QueryResult]]:
        groups: dict[str, list[QueryResult]] = {}
        for r in summary.query_results:
            v = getattr(r, key)
            groups.setdefault(v, []).append(r)
        return groups

    def _fmt_group(groups: dict[str, list[QueryResult]]) -> str:
        lines = []
        for label, items in sorted(groups.items()):
            n = len(items)
            rec = sum(r.recall_at_5 for r in items) / n
            m = sum(r.mrr for r in items) / n
            nd = sum(r.ndcg_at_5 for r in items) / n
            lines.append(f"  {label:<22} (n={n:>2}):  Recall={rec:.3f}  MRR={m:.3f}  NDCG={nd:.3f}")
        return "\n".join(lines)

    misses = [r for r in summary.query_results if r.recall_at_5 == 0.0]

    threshold_pass = summary.recall_at_5 >= _RECALL_THRESHOLD
    threshold_mark = "PASS ✓" if threshold_pass else f"FAIL ✗  (threshold={_RECALL_THRESHOLD:.2f})"

    cp_line = f"{summary.context_precision:.3f}" if summary.context_precision is not None else "n/a (--skip-llm-judge)"

    lines = [
        "=" * 60,
        "RAG Evaluation Report",
        "=" * 60,
        f"Config:      {summary.config_name}",
        f"Dataset:     {summary.dataset_name} ({summary.query_count} entries)",
        f"Collection:  {summary.collection}",
        f"Run ID:      {summary.run_id}",
        f"Run at:      {summary.run_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "METRICS",
        "-" * 40,
        f"  Recall@5:          {summary.recall_at_5:.4f}",
        f"  MRR:               {summary.mrr:.4f}",
        f"  NDCG@5:            {summary.ndcg_at_5:.4f}",
        f"  Context Precision: {cp_line}",
        "",
        "BY QUERY TYPE",
        "-" * 40,
        _fmt_group(_group("query_type")),
        "",
        "BY DIFFICULTY",
        "-" * 40,
        _fmt_group(_group("difficulty")),
        "",
        "THRESHOLD CHECK",
        "-" * 40,
        f"  Recall@5 = {summary.recall_at_5:.4f} >= {_RECALL_THRESHOLD:.2f}  →  {threshold_mark}",
    ]

    if misses:
        lines += ["", f"MISSES ({len(misses)} queries with Recall@5 = 0)", "-" * 40]
        for r in misses:
            lines.append(f"  [{r.query_type}/{r.difficulty}] {r.query[:70]}")

    report = "\n".join(lines) + "\n"
    path.write_text(report)
    return path


# ---------------------------------------------------------------------------
# Langfuse dataset run
# ---------------------------------------------------------------------------


def _track_langfuse(
    dataset: EvalDataset,
    precomputed: list[_PrecomputedQuery],
    summary: RunSummary,
) -> None:
    try:
        from langfuse.experiment import Evaluation

        lf = get_langfuse()

        # Ensure dataset exists
        lf.create_dataset(name=dataset.name, description="microblog Flask→FastAPI retrieval eval")

        # Create/update items (deterministic IDs → idempotent)
        for entry in dataset.entries:
            lf.create_dataset_item(
                dataset_name=dataset.name,
                input={"query": entry.query},
                expected_output={"chunk_ids": entry.ground_truth_chunk_ids},
                metadata={"query_type": entry.query_type, "difficulty": entry.difficulty},
                id=dataset.item_id(entry),
            )

        # Build results lookup keyed by query text
        results_map = {qr.query: qr for qr in summary.query_results}

        def _task(*, item: Any, **kw: Any) -> dict[str, Any]:
            query = item.input["query"]
            qr = results_map.get(query)
            if qr is None:
                return {}
            return {
                "retrieved_ids": qr.retrieved_ids,
                "recall_at_5": qr.recall_at_5,
                "mrr": qr.mrr,
                "ndcg_at_5": qr.ndcg_at_5,
                "context_precision": qr.context_precision,
            }

        def _eval_recall(*, input: Any, output: Any, expected_output: Any = None, **kw: Any) -> Evaluation:
            return Evaluation(name="recall_at_5", value=float(output.get("recall_at_5", 0.0)))

        def _eval_mrr(*, input: Any, output: Any, expected_output: Any = None, **kw: Any) -> Evaluation:
            return Evaluation(name="mrr", value=float(output.get("mrr", 0.0)))

        def _eval_ndcg(*, input: Any, output: Any, expected_output: Any = None, **kw: Any) -> Evaluation:
            return Evaluation(name="ndcg_at_5", value=float(output.get("ndcg_at_5", 0.0)))

        evaluators = [_eval_recall, _eval_mrr, _eval_ndcg]

        if summary.context_precision is not None:
            def _eval_cp(*, input: Any, output: Any, expected_output: Any = None, **kw: Any) -> Evaluation:
                return Evaluation(name="context_precision", value=float(output.get("context_precision") or 0.0))
            evaluators.append(_eval_cp)

        lf_dataset = lf.get_dataset(dataset.name)
        lf.run_experiment(
            name=dataset.name,
            run_name=f"{summary.config_name}/{summary.run_id[:8]}",
            data=list(lf_dataset.items),
            task=_task,
            evaluators=evaluators,  # type: ignore[arg-type]
            metadata={"config_name": summary.config_name, "run_id": summary.run_id},
        )
        logger.info(
            "Langfuse dataset run created: {}/{}", dataset.name, summary.run_id[:8]
        )
    except Exception as exc:
        logger.warning("Langfuse tracking failed (non-fatal): {}", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("default", help=f"Retrieval config: {list(EVAL_CONFIGS)}"),
    dataset: Path = typer.Option(_DEFAULT_DATASET, help="Path to JSONL eval dataset"),
    skip_llm_judge: bool = typer.Option(False, "--skip-llm-judge", help="Skip Groq context-precision judge"),
    collection: str = typer.Option(_COLLECTION, help="Qdrant collection to query"),
) -> None:
    """Run the RAG evaluation pipeline and write a report to eval/reports/."""
    if config not in EVAL_CONFIGS:
        typer.echo(f"Unknown config {config!r}. Choices: {list(EVAL_CONFIGS)}", err=True)
        raise typer.Exit(1)

    asyncio.run(_async_main(config, dataset, skip_llm_judge, collection))


async def _async_main(
    config_name: str,
    dataset_path: Path,
    skip_llm_judge: bool,
    collection: str,
) -> None:
    run_id = str(uuid.uuid4())
    eval_config = EVAL_CONFIGS[config_name]

    logger.info("Starting eval: config={} run_id={}", config_name, run_id)

    dataset = EvalDataset.from_jsonl(dataset_path)
    logger.info("Loaded dataset '{}': {} entries", dataset.name, len(dataset.entries))

    embedder = VoyageEmbedder()
    precomputed = await _precompute_embeddings(dataset.entries, embedder)

    summary = await _run_eval(
        config_name=config_name,
        config=eval_config,
        precomputed=precomputed,
        run_id=run_id,
        skip_llm_judge=skip_llm_judge,
        dataset_name=dataset.name,
    )

    await _save_to_postgres(summary)

    report_path = _write_report(summary)
    logger.info("Report written to {}", report_path)

    _track_langfuse(dataset, precomputed, summary)

    # Print summary to stdout
    typer.echo(f"\nRecall@5={summary.recall_at_5:.4f}  MRR={summary.mrr:.4f}  NDCG@5={summary.ndcg_at_5:.4f}", color=True)
    if summary.context_precision is not None:
        typer.echo(f"Context Precision={summary.context_precision:.4f}")
    threshold_pass = summary.recall_at_5 >= _RECALL_THRESHOLD
    status = "PASS" if threshold_pass else "FAIL"
    typer.echo(f"Threshold Recall@5 >= {_RECALL_THRESHOLD}: {status}")
    typer.echo(f"Report: {report_path}")


if __name__ == "__main__":
    app()
