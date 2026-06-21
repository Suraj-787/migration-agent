"""HybridRetriever: parallel dense+sparse Qdrant queries with client-side weighted RRF."""
from __future__ import annotations

import asyncio
import time
from typing import Any, cast

from fastembed import SparseTextEmbedding
from loguru import logger
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector

from rag.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, get_qdrant_client
from rag.embedder import VoyageEmbedder
from rag.models import SearchHit

# Re-export so existing importers of `rag.retriever.SearchHit` keep working.
__all__ = ["SearchHit", "MetadataFilter", "RetrievalConfig", "HybridRetriever"]

_BM25_MODEL: SparseTextEmbedding | None = None


def _get_bm25() -> SparseTextEmbedding:
    global _BM25_MODEL
    if _BM25_MODEL is None:
        logger.info("Loading BM25 model for retriever (first use)")
        _BM25_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _BM25_MODEL


class MetadataFilter(BaseModel):
    """Payload filter applied server-side on Qdrant queries.

    All set fields are combined with AND (Qdrant `must`).
    - language:   matches code_chunks payload field "language"
    - node_type:  matches code_chunks payload field "node_type"
    - file_path:  exact match on payload field "file_path"
    """

    language: str | None = None
    node_type: str | None = None
    file_path: str | None = None


class RetrievalConfig(BaseModel):
    """Controls the hybrid-retrieve → rerank pipeline."""

    hybrid_alpha: float = 0.5
    retrieve_k: int = 20
    rerank_k: int = 6
    use_reranker: bool = True


def _build_filter(mf: MetadataFilter | None) -> Filter | None:
    """Convert a MetadataFilter to a Qdrant Filter (must-all AND)."""
    if mf is None:
        return None
    conditions: list[FieldCondition] = []
    if mf.language:
        conditions.append(FieldCondition(key="language", match=MatchValue(value=mf.language)))
    if mf.node_type:
        conditions.append(FieldCondition(key="node_type", match=MatchValue(value=mf.node_type)))
    if mf.file_path:
        conditions.append(FieldCondition(key="file_path", match=MatchValue(value=mf.file_path)))
    return Filter(must=cast(list[Any], conditions)) if conditions else None


class HybridRetriever:
    """Parallel dense+sparse retrieval with client-side weighted RRF fusion.

    RRF score per document:
        score = alpha * 1/(k + dense_rank) + (1-alpha) * 1/(k + sparse_rank)

    Documents absent from one list contribute 0 from that side.
    """

    def __init__(
        self,
        collection: str,
        alpha: float = 0.5,
        rrf_k: int = 60,
        embedder: VoyageEmbedder | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self._collection = collection
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._embedder = embedder or VoyageEmbedder()
        self._client = client or get_qdrant_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 10,
        prefetch_limit: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[SearchHit]:
        """Run hybrid retrieval → optional reranking and return results.

        When `config` is supplied it governs alpha, retrieve_k, rerank_k, and
        use_reranker; `top_k` and `prefetch_limit` are ignored in that case.
        Every call is logged to Langfuse as a retrieval span.
        """
        t0 = time.monotonic()

        # Resolve effective parameters
        alpha = config.hybrid_alpha if config else self._alpha
        retrieve_k = config.retrieve_k if config else top_k * 4 if prefetch_limit is None else prefetch_limit
        final_k = config.rerank_k if config else top_k
        use_reranker = config.use_reranker if config else False

        dense_vec, sparse_vec = await asyncio.gather(
            self._embedder.embed_query(query),
            asyncio.to_thread(self._bm25_embed, query),
        )

        # Temporarily override alpha if config specifies a different value
        saved_alpha = self._alpha
        self._alpha = alpha
        try:
            raw_hits = await self.search_vectors(
                dense_vec=dense_vec,
                sparse_vec=sparse_vec,
                query_label=query,
                top_k=retrieve_k,
                metadata_filter=metadata_filter,
            )
        finally:
            self._alpha = saved_alpha

        retrieved_ids = [h.id for h in raw_hits]

        reranked_ids: list[str] | None = None
        if use_reranker and raw_hits:
            from rag.reranker import rerank  # lazy import avoids circular dep
            raw_hits = await rerank(query, raw_hits, final_k)
            reranked_ids = [h.id for h in raw_hits]

        results = raw_hits[:final_k]
        latency_ms = (time.monotonic() - t0) * 1000.0

        self._log_langfuse(
            query=query,
            retrieved_ids=retrieved_ids,
            reranked_ids=reranked_ids,
            latency_ms=latency_ms,
            alpha=alpha,
            use_reranker=use_reranker,
        )

        return results

    async def search_vectors(
        self,
        dense_vec: list[float],
        sparse_vec: SparseVector,
        query_label: str = "",
        top_k: int = 10,
        prefetch_limit: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchHit]:
        """Like search() but accepts pre-computed vectors; embedding step is skipped.

        Useful for benchmarking multiple alpha configs over the same query without
        re-calling the Voyage API per config.
        """
        limit = prefetch_limit if prefetch_limit is not None else top_k * 4
        qfilter = _build_filter(metadata_filter)

        # Query Qdrant for dense and sparse results in parallel
        dense_resp, sparse_resp = await asyncio.gather(
            self._client.query_points(
                collection_name=self._collection,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
            ),
            self._client.query_points(
                collection_name=self._collection,
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
            ),
        )

        dense_points = dense_resp.points
        sparse_points = sparse_resp.points

        logger.debug(
            "Dense top-5 [alpha={:.1f}] query='{}': {}",
            self._alpha,
            query_label[:60],
            [
                {
                    "id": str(p.id),
                    "score": round(p.score, 4),
                    "title": (p.payload or {}).get("title", (p.payload or {}).get("file_path", "")),
                }
                for p in dense_points[:5]
            ],
        )
        logger.debug(
            "Sparse top-5 [alpha={:.1f}] query='{}': {}",
            self._alpha,
            query_label[:60],
            [
                {
                    "id": str(p.id),
                    "score": round(p.score, 4),
                    "title": (p.payload or {}).get("title", (p.payload or {}).get("file_path", "")),
                }
                for p in sparse_points[:5]
            ],
        )

        # Client-side weighted RRF
        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for rank, point in enumerate(dense_points):
            pid = str(point.id)
            scores[pid] = scores.get(pid, 0.0) + self._alpha / (self._rrf_k + rank + 1)
            payloads[pid] = point.payload or {}

        for rank, point in enumerate(sparse_points):
            pid = str(point.id)
            scores[pid] = scores.get(pid, 0.0) + (1.0 - self._alpha) / (self._rrf_k + rank + 1)
            if pid not in payloads:
                payloads[pid] = point.payload or {}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [SearchHit(id=pid, score=score, payload=payloads[pid]) for pid, score in ranked]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bm25_embed(self, query: str) -> SparseVector:
        result = list(_get_bm25().embed([query]))[0]
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    def _log_langfuse(
        self,
        *,
        query: str,
        retrieved_ids: list[str],
        reranked_ids: list[str] | None,
        latency_ms: float,
        alpha: float,
        use_reranker: bool,
    ) -> None:
        try:
            from agents.tracing import trace_span

            trace_span(
                "retrieval",
                span_type="retriever",
                input={"query": query, "alpha": alpha, "use_reranker": use_reranker},
                output={"retrieved_ids": retrieved_ids, "reranked_ids": reranked_ids},
                metadata={"latency_ms": round(latency_ms, 2), "collection": self._collection},
            )
        except Exception as exc:
            logger.warning("Langfuse tracing failed (non-fatal): {}", exc)
