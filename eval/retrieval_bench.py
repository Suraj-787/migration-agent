"""Retrieval benchmark: Recall@5 and MRR across alpha configs, with/without reranker.

Runs 20 hand-written (query, expected_source) pairs against the doc_chunks collection.
Saves results to Postgres eval_runs table.

Usage:
    uv run python -m eval.retrieval_bench
    uv run python -m eval.retrieval_bench --collection doc_chunks --top-k 5
    uv run python -m eval.retrieval_bench --no-reranker   # skip reranker pass
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import asyncpg
from dotenv import load_dotenv
from loguru import logger

from rag.collections import get_qdrant_client
from rag.embedder import VoyageEmbedder
from rag.retriever import HybridRetriever, RetrievalConfig

load_dotenv()

# ---------------------------------------------------------------------------
# Ground-truth query pairs
# Each tuple: (query_text, expected_source_stem)
# expected_source_stem matches the `source` payload field (markdown filename stem)
# ---------------------------------------------------------------------------

QUERIES: list[tuple[str, str]] = [
    # fastapi_templates (5)
    ("replace flask render_template with FastAPI equivalent", "fastapi_templates"),
    ("Jinja2Templates TemplateResponse how to use in FastAPI", "fastapi_templates"),
    ("mount static files and templates directory in FastAPI", "fastapi_templates"),
    ("how to serve HTML pages from FastAPI route handlers", "fastapi_templates"),
    ("FastAPI setup Jinja2 template directory path", "fastapi_templates"),
    # flask_templates (5)
    ("flask render_template function how to use", "flask_templates"),
    ("jinja2 template inheritance extends block syntax", "flask_templates"),
    ("passing context variables to flask template", "flask_templates"),
    ("url_for static file reference in flask jinja2 template", "flask_templates"),
    ("jinja2 if for block control flow in flask", "flask_templates"),
    # fastapi_bigger_applications (5)
    ("migrate Flask Blueprint to FastAPI APIRouter", "fastapi_bigger_applications"),
    ("split FastAPI application into multiple router files", "fastapi_bigger_applications"),
    ("include_router register router in FastAPI main app", "fastapi_bigger_applications"),
    ("APIRouter prefix tags FastAPI route registration", "fastapi_bigger_applications"),
    ("Flask register_blueprint equivalent FastAPI", "fastapi_bigger_applications"),
    # fastapi_forms (5)
    ("replace flask request.form access with FastAPI Form", "fastapi_forms"),
    ("python-multipart installation for FastAPI form handling", "fastapi_forms"),
    ("Annotated Form field type hint FastAPI endpoint", "fastapi_forms"),
    ("receive form data instead of JSON body in FastAPI", "fastapi_forms"),
    ("FastAPI login form username password Form field", "fastapi_forms"),
]

ALPHA_CONFIGS: list[float] = [0.0, 0.5, 0.7, 1.0]

_COLLECTION = os.environ.get("BENCH_COLLECTION", "doc_chunks")
_TOP_K = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query: str
    expected_source: str
    alpha: float
    use_reranker: bool
    hits: list[str]  # source field of each hit
    found_at_rank: int | None  # 1-indexed rank of first relevant hit, None if not found


@dataclass
class RunMetrics:
    alpha: float
    use_reranker: bool
    recall_at_5: float
    mrr: float
    query_count: int
    per_query: list[QueryResult]


def compute_metrics(results: list[QueryResult]) -> tuple[float, float]:
    """Return (Recall@5, MRR) for a list of QueryResult."""
    if not results:
        return 0.0, 0.0
    recall_hits = sum(1 for r in results if r.found_at_rank is not None)
    mrr_sum = sum(1.0 / r.found_at_rank for r in results if r.found_at_rank is not None)
    return recall_hits / len(results), mrr_sum / len(results)


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id         BIGSERIAL PRIMARY KEY,
    run_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    collection TEXT        NOT NULL,
    alpha      FLOAT       NOT NULL,
    top_k      INT         NOT NULL,
    recall_at_5 FLOAT      NOT NULL,
    mrr        FLOAT       NOT NULL,
    query_count INT        NOT NULL,
    details    JSONB       NOT NULL,
    use_reranker BOOLEAN   NOT NULL DEFAULT FALSE
)
"""

_ADD_RERANKER_COL = """
ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS use_reranker BOOLEAN NOT NULL DEFAULT FALSE
"""


async def _save_run(
    conn: asyncpg.Connection, metrics: RunMetrics, collection: str, top_k: int
) -> int:
    await conn.execute(_CREATE_TABLE)
    await conn.execute(_ADD_RERANKER_COL)
    details = [asdict(r) for r in metrics.per_query]
    row = await conn.fetchrow(
        """
        INSERT INTO eval_runs
            (run_at, collection, alpha, top_k, recall_at_5, mrr, query_count, details, use_reranker)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """,
        datetime.now(timezone.utc),
        collection,
        metrics.alpha,
        top_k,
        metrics.recall_at_5,
        metrics.mrr,
        metrics.query_count,
        json.dumps(details),
        metrics.use_reranker,
    )
    return row["id"]  # type: ignore[index]


async def _get_pg_conn() -> asyncpg.Connection | None:
    dsn = (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'migration')}"
        f":{os.environ.get('POSTGRES_PASSWORD', '')}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ.get('POSTGRES_DB', 'migration_agent')}"
    )
    try:
        return await asyncpg.connect(dsn)
    except Exception as exc:
        logger.warning("Cannot connect to Postgres, skipping eval_runs save: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedQuery:
    query: str
    expected_source: str
    dense_vec: list[float]
    sparse_vec: object  # SparseVector — avoid importing at module level


async def precompute_embeddings(
    queries: list[tuple[str, str]],
    embedder: VoyageEmbedder,
) -> list[PrecomputedQuery]:
    """Embed all queries once so alpha configs can share vectors (avoids N×alpha Voyage calls)."""
    from fastembed import SparseTextEmbedding
    from qdrant_client.models import SparseVector

    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    texts = [q for q, _ in queries]

    logger.info("Embedding {} queries via Voyage in one batch call ...", len(texts))
    dense_vecs = await embedder.embed_queries(texts)

    sparse_results = list(bm25.embed(texts))
    sparse_vecs = [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in sparse_results
    ]

    return [
        PrecomputedQuery(query=q, expected_source=src, dense_vec=dv, sparse_vec=sv)
        for (q, src), dv, sv in zip(queries, dense_vecs, sparse_vecs)
    ]


async def run_alpha(
    alpha: float,
    precomputed: list[PrecomputedQuery],
    collection: str,
    top_k: int,
    client: object,
    use_reranker: bool = False,
) -> RunMetrics:
    retriever = HybridRetriever(
        collection=collection,
        alpha=alpha,
        client=client,  # type: ignore[arg-type]
    )

    # For reranker pass: retrieve top-20 candidates then rerank to top_k.
    retrieve_k = 20 if use_reranker else top_k
    config: RetrievalConfig | None = None
    if use_reranker:
        config = RetrievalConfig(
            hybrid_alpha=alpha,
            retrieve_k=retrieve_k,
            rerank_k=top_k,
            use_reranker=True,
        )

    query_results: list[QueryResult] = []
    for pq in precomputed:
        if use_reranker:
            # search() with config triggers the full rerank pipeline
            hits = await retriever.search(
                pq.query,
                config=config,
            )
        else:
            hits = await retriever.search_vectors(
                dense_vec=pq.dense_vec,
                sparse_vec=pq.sparse_vec,  # type: ignore[arg-type]
                query_label=pq.query,
                top_k=top_k,
            )
        hit_sources = [h.payload.get("source", "") for h in hits]

        found_at_rank: int | None = None
        for rank, src in enumerate(hit_sources, start=1):
            if src == pq.expected_source:
                found_at_rank = rank
                break

        query_results.append(
            QueryResult(
                query=pq.query,
                expected_source=pq.expected_source,
                alpha=alpha,
                use_reranker=use_reranker,
                hits=hit_sources,
                found_at_rank=found_at_rank,
            )
        )
        status = f"rank {found_at_rank}" if found_at_rank else "MISS"
        reranker_tag = "+reranker" if use_reranker else "         "
        logger.debug(
            "alpha={:.1f} {} | {} | {}", alpha, reranker_tag, status, pq.query[:55]
        )

    recall, mrr = compute_metrics(query_results)
    return RunMetrics(
        alpha=alpha,
        use_reranker=use_reranker,
        recall_at_5=recall,
        mrr=mrr,
        query_count=len(query_results),
        per_query=query_results,
    )


async def main(
    collection: str = _COLLECTION,
    top_k: int = _TOP_K,
    run_reranker: bool = True,
) -> None:
    n_configs = len(ALPHA_CONFIGS) * (2 if run_reranker else 1)
    logger.info(
        "Starting retrieval benchmark: {} queries × {} configs", len(QUERIES), n_configs
    )
    embedder = VoyageEmbedder()
    client = get_qdrant_client()

    # Embed all queries once; reuse across all alpha configs
    precomputed = await precompute_embeddings(QUERIES, embedder)

    all_metrics: list[RunMetrics] = []

    # --- baseline pass (no reranker) ---
    for alpha in ALPHA_CONFIGS:
        logger.info("Running alpha={:.1f} (no reranker) ...", alpha)
        metrics = await run_alpha(
            alpha=alpha,
            precomputed=precomputed,
            collection=collection,
            top_k=top_k,
            client=client,
            use_reranker=False,
        )
        all_metrics.append(metrics)
        logger.info(
            "alpha={:.1f}  Recall@{}={:.3f}  MRR={:.3f}",
            alpha,
            top_k,
            metrics.recall_at_5,
            metrics.mrr,
        )

    # --- reranker pass ---
    if run_reranker:
        logger.info("--- Starting reranker pass (retrieve-20 → rerank → top-{}) ---", top_k)
        # Warm up model before the timed loop
        from rag.reranker import _get_reranker
        _get_reranker()

        for alpha in ALPHA_CONFIGS:
            logger.info("Running alpha={:.1f} (+reranker) ...", alpha)
            metrics = await run_alpha(
                alpha=alpha,
                precomputed=precomputed,
                collection=collection,
                top_k=top_k,
                client=client,
                use_reranker=True,
            )
            all_metrics.append(metrics)
            logger.info(
                "alpha={:.1f} +reranker  Recall@{}={:.3f}  MRR={:.3f}",
                alpha,
                top_k,
                metrics.recall_at_5,
                metrics.mrr,
            )

    # Save to Postgres
    conn = await _get_pg_conn()
    if conn is not None:
        try:
            for m in all_metrics:
                run_id = await _save_run(conn, m, collection, top_k)
                logger.info(
                    "Saved eval_run id={} (alpha={:.1f}, reranker={})",
                    run_id,
                    m.alpha,
                    m.use_reranker,
                )
        finally:
            await conn.close()

    # Print summary table
    _print_table(all_metrics, top_k)

    best = max(all_metrics, key=lambda m: (m.recall_at_5, m.mrr))
    logger.info(
        "Best config: alpha={:.1f} reranker={} → Recall@{}={:.3f}, MRR={:.3f}",
        best.alpha,
        best.use_reranker,
        top_k,
        best.recall_at_5,
        best.mrr,
    )


def _print_table(metrics: list[RunMetrics], top_k: int) -> None:
    header = f"{'alpha':>6}  {'reranker':>8}  {'Recall@' + str(top_k):>10}  {'MRR':>8}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m.alpha:>6.1f}  {'yes' if m.use_reranker else 'no':>8}  "
            f"{m.recall_at_5:>10.3f}  {m.mrr:>8.3f}"
        )
    print("-" * len(header) + "\n")


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def cli(
        collection: str = typer.Option(_COLLECTION, help="Qdrant collection to benchmark"),
        top_k: int = typer.Option(_TOP_K, help="Number of results to retrieve per query"),
        no_reranker: bool = typer.Option(False, "--no-reranker", help="Skip the reranker pass"),
    ) -> None:
        """Run retrieval benchmark across alpha configs (with/without reranker) and report metrics."""
        asyncio.run(main(collection=collection, top_k=top_k, run_reranker=not no_reranker))

    app()
