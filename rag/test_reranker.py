"""Tests for rag/reranker.py.

Unit tests mock the rerankers library — no GPU/CPU model required.
Integration tests require Qdrant + VOYAGE_API_KEY and run the full
retrieve → rerank pipeline end-to-end.

Run: uv run pytest rag/test_reranker.py -xvs
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
_DOC_COLLECTION = "doc_chunks"


def _qdrant_is_up() -> bool:
    try:
        r = httpx.get(f"{QDRANT_URL}/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _voyage_key_set() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


_integration_skip = pytest.mark.skipif(
    not _qdrant_is_up() or not _voyage_key_set(),
    reason="Integration tests require Qdrant (docker compose up -d) and VOYAGE_API_KEY",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hits(n: int) -> list:
    from rag.models import SearchHit

    return [
        SearchHit(
            id=str(i),
            score=1.0 / (i + 1),
            payload={"content": f"document {i} about flask and jinja2", "source": f"doc_{i}"},
        )
        for i in range(n)
    ]


def _make_mock_reranker(scores: list[float]) -> MagicMock:
    """Return a mock Reranker whose .rank() returns results with given scores (index = doc_id)."""
    mock_result = MagicMock()
    mock_results = []
    for doc_id, score in enumerate(scores):
        r = MagicMock()
        r.doc_id = doc_id
        r.score = score
        mock_results.append(r)
    mock_result.results = mock_results
    reranker = MagicMock()
    reranker.rank.return_value = mock_result
    return reranker


# ---------------------------------------------------------------------------
# Unit tests — no I/O, model mocked out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_empty_hits_returns_empty() -> None:
    from rag.reranker import rerank

    result = await rerank("query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_rerank_reorders_by_score() -> None:
    """Reranker score should override the original RRF ordering."""
    from rag.reranker import rerank

    hits = _make_hits(3)
    # Original order: doc 0 (score 1.0), doc 1 (0.5), doc 2 (0.33)
    # Reranker says doc 2 is best, then doc 0, then doc 1
    mock_reranker = _make_mock_reranker([0.5, 0.1, 0.9])

    with patch("rag.reranker._get_reranker", return_value=mock_reranker):
        result = await rerank("flask templates", hits, top_k=3)

    assert result[0].id == "2"  # highest reranker score
    assert result[1].id == "0"
    assert result[2].id == "1"


@pytest.mark.asyncio
async def test_rerank_top_k_truncates_results() -> None:
    from rag.reranker import rerank

    hits = _make_hits(10)
    mock_reranker = _make_mock_reranker([float(i) for i in range(10)])

    with patch("rag.reranker._get_reranker", return_value=mock_reranker):
        result = await rerank("query", hits, top_k=4)

    assert len(result) == 4


@pytest.mark.asyncio
async def test_rerank_scores_updated_to_reranker_scores() -> None:
    from rag.reranker import rerank

    hits = _make_hits(2)
    mock_reranker = _make_mock_reranker([0.42, 0.77])

    with patch("rag.reranker._get_reranker", return_value=mock_reranker):
        result = await rerank("query", hits, top_k=2)

    # After reranking, scores should be the reranker relevance scores
    result_by_id = {h.id: h.score for h in result}
    assert abs(result_by_id["1"] - 0.77) < 1e-6  # doc 1 has highest reranker score
    assert abs(result_by_id["0"] - 0.42) < 1e-6


@pytest.mark.asyncio
async def test_retrieval_config_defaults() -> None:
    from rag.retriever import RetrievalConfig

    cfg = RetrievalConfig()
    assert cfg.hybrid_alpha == 0.5
    assert cfg.retrieve_k == 20
    assert cfg.rerank_k == 6
    assert cfg.use_reranker is True


def test_reranker_singleton_loaded_once() -> None:
    """_get_reranker() must return the same object on repeated calls."""
    import rag.reranker as reranker_mod

    saved = reranker_mod._RERANKER
    try:
        mock = MagicMock()
        reranker_mod._RERANKER = mock
        r1 = reranker_mod._get_reranker()
        r2 = reranker_mod._get_reranker()
        assert r1 is r2 is mock
    finally:
        reranker_mod._RERANKER = saved


# ---------------------------------------------------------------------------
# Integration tests — require Qdrant + ingested doc_chunks + VOYAGE_API_KEY
# ---------------------------------------------------------------------------


@_integration_skip
@pytest.mark.asyncio
async def test_reranker_pipeline_returns_results() -> None:
    """Full retrieve → rerank pipeline should return ≤ rerank_k hits."""
    from rag.retriever import HybridRetriever, RetrievalConfig

    cfg = RetrievalConfig(hybrid_alpha=0.5, retrieve_k=20, rerank_k=6, use_reranker=True)
    retriever = HybridRetriever(collection=_DOC_COLLECTION, alpha=cfg.hybrid_alpha)
    hits = await retriever.search(
        "replace flask render_template with FastAPI Jinja2Templates",
        config=cfg,
    )
    assert 1 <= len(hits) <= cfg.rerank_k
    for h in hits:
        assert h.id
        assert isinstance(h.score, float)


@_integration_skip
@pytest.mark.asyncio
async def test_reranker_improves_or_maintains_top1() -> None:
    """With reranker the top-1 hit should still contain relevant content."""
    from rag.retriever import HybridRetriever, RetrievalConfig

    cfg = RetrievalConfig(hybrid_alpha=0.5, retrieve_k=20, rerank_k=6, use_reranker=True)
    retriever = HybridRetriever(collection=_DOC_COLLECTION, alpha=cfg.hybrid_alpha)
    hits = await retriever.search(
        "Jinja2Templates TemplateResponse how to use in FastAPI",
        config=cfg,
    )
    assert hits, "Expected at least one result after reranking"
    top_content = hits[0].payload.get("content", "")
    assert "Jinja2" in top_content or "template" in top_content.lower(), (
        f"Expected Jinja2-related content at top-1, got: {top_content[:120]}"
    )


@_integration_skip
@pytest.mark.asyncio
async def test_retrieval_latency_under_500ms() -> None:
    """End-to-end retrieve+rerank must complete in under 500 ms on first warm call."""
    import time

    from rag.retriever import HybridRetriever, RetrievalConfig

    cfg = RetrievalConfig(hybrid_alpha=0.5, retrieve_k=20, rerank_k=6, use_reranker=True)
    retriever = HybridRetriever(collection=_DOC_COLLECTION, alpha=cfg.hybrid_alpha)

    # Warm up model (first load may take several seconds — skip that)
    from rag.reranker import _get_reranker
    _get_reranker()

    t0 = time.monotonic()
    await retriever.search("migrate Flask Blueprint to FastAPI APIRouter", config=cfg)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert elapsed_ms < 500.0, f"Retrieval+rerank took {elapsed_ms:.0f} ms (limit 500 ms)"
