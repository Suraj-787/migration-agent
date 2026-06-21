"""POST /search — hybrid retrieval via HybridRetriever (client-side weighted RRF + optional reranker)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from rag.collections import get_qdrant_client
from rag.embedder import VoyageEmbedder
from rag.retriever import HybridRetriever, RetrievalConfig

router = APIRouter(tags=["search"])

_EMBEDDER: VoyageEmbedder | None = None
_RETRIEVER: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _RETRIEVER, _EMBEDDER
    if _RETRIEVER is None:
        _EMBEDDER = VoyageEmbedder()
        _RETRIEVER = HybridRetriever(
            collection="doc_chunks",
            embedder=_EMBEDDER,
            client=get_qdrant_client(),
        )
    return _RETRIEVER


class SearchRequest(BaseModel):
    query: str
    collection: str = "doc_chunks"
    top_k: int = 10
    use_reranker: bool = False
    hybrid_alpha: float = 0.5


class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


@router.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest) -> list[SearchResult]:
    """Hybrid dense+sparse retrieval with optional cross-encoder reranking.

    Results are logged to Langfuse as a retrieval span.
    """
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if req.top_k < 1 or req.top_k > 100:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 100")

    retriever = _get_retriever()
    # Swap collection if caller requests a different one
    retriever._collection = req.collection

    config = RetrievalConfig(
        hybrid_alpha=req.hybrid_alpha,
        retrieve_k=20 if req.use_reranker else req.top_k,
        rerank_k=req.top_k,
        use_reranker=req.use_reranker,
    )

    try:
        hits = await retriever.search(req.query, config=config)
    except Exception as exc:
        logger.error("Search failed for collection '{}': {}", req.collection, exc)
        raise HTTPException(status_code=502, detail=f"Search backend error: {exc}") from exc

    return [SearchResult(id=h.id, score=h.score, payload=h.payload) for h in hits]
