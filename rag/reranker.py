"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3 via the rerankers library.

Model is loaded once as a module-level singleton (CPU, ~600 MB).
All public functions are async — CPU work is offloaded via asyncio.to_thread.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from rag.models import SearchHit

_RERANKER = None  # rerankers.Reranker, typed as Any to avoid importing at module level


def _get_reranker() -> object:
    global _RERANKER
    if _RERANKER is None:
        from rerankers import Reranker

        logger.info("Loading BAAI/bge-reranker-v2-m3 (first use, CPU)")
        _RERANKER = Reranker("BAAI/bge-reranker-v2-m3", model_type="cross-encoder")
        logger.info("bge-reranker-v2-m3 ready")
    return _RERANKER


def _rerank_sync(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    reranker = _get_reranker()
    docs = [h.payload.get("content", "") for h in hits]
    results = reranker.rank(query=query, docs=docs)  # type: ignore[attr-defined]
    ranked = sorted(results.results, key=lambda r: r.score, reverse=True)[:top_k]
    return [
        SearchHit(id=hits[r.doc_id].id, score=float(r.score), payload=hits[r.doc_id].payload)
        for r in ranked
    ]


async def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Rerank `hits` against `query` using bge-reranker-v2-m3; return top `top_k`."""
    if not hits:
        return []
    logger.debug("Reranking {} candidates → top-{}", len(hits), top_k)
    return await asyncio.to_thread(_rerank_sync, query, hits, top_k)
