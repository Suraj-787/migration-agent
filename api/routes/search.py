"""POST /search — hybrid dense + sparse RRF retrieval via Qdrant server-side fusion."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastembed import SparseTextEmbedding
from loguru import logger
from pydantic import BaseModel
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from rag.collections import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)
from rag.embedder import VoyageEmbedder

router = APIRouter(tags=["search"])

_BM25_MODEL: SparseTextEmbedding | None = None
_EMBEDDER: VoyageEmbedder | None = None


def _get_bm25() -> SparseTextEmbedding:
    global _BM25_MODEL
    if _BM25_MODEL is None:
        logger.info("Loading BM25 model for search (first use)")
        _BM25_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _BM25_MODEL


def _get_embedder() -> VoyageEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = VoyageEmbedder()
    return _EMBEDDER


class SearchRequest(BaseModel):
    query: str
    collection: str
    top_k: int = 10


class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


@router.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest) -> list[SearchResult]:
    """Hybrid dense + sparse retrieval with server-side RRF fusion."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if req.top_k < 1 or req.top_k > 100:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 100")

    embedder = _get_embedder()
    bm25 = _get_bm25()

    # Dense query embedding via Voyage
    dense_vec = await embedder.embed_query(req.query)

    # Sparse BM25 query embedding
    sparse_result = list(bm25.embed([req.query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    client = get_qdrant_client()
    try:
        response = await client.query_points(
            collection_name=req.collection,
            prefetch=[
                Prefetch(
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    limit=req.top_k * 2,
                ),
                Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=req.top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=req.top_k,
            with_payload=True,
        )
    except Exception as exc:
        logger.error("Qdrant query failed for collection '{}': {}", req.collection, exc)
        raise HTTPException(status_code=502, detail=f"Search backend error: {exc}") from exc

    return [
        SearchResult(
            id=str(point.id),
            score=point.score,
            payload=point.payload or {},
        )
        for point in response.points
    ]
