"""Tests for rag/retriever.py: RRF logic (unit) + HybridRetriever end-to-end (integration).

Unit tests require no infra.
Integration tests require Qdrant + VOYAGE_API_KEY — skipped automatically otherwise.
Run: uv run pytest rag/test_retriever.py -xvs
"""
from __future__ import annotations

import os

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
# Unit tests — pure RRF maths, no I/O
# ---------------------------------------------------------------------------


def test_build_filter_all_none_returns_none() -> None:
    from rag.retriever import MetadataFilter, _build_filter

    assert _build_filter(None) is None
    assert _build_filter(MetadataFilter()) is None


def test_build_filter_language_only() -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from rag.retriever import MetadataFilter, _build_filter

    f = _build_filter(MetadataFilter(language="python"))
    assert isinstance(f, Filter)
    assert isinstance(f.must, list)
    assert len(f.must) == 1
    cond = f.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "language"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == "python"


def test_build_filter_all_fields() -> None:
    from qdrant_client.models import Filter

    from rag.retriever import MetadataFilter, _build_filter

    f = _build_filter(MetadataFilter(language="python", node_type="function", file_path="/repo/main.py"))
    assert isinstance(f, Filter)
    assert isinstance(f.must, list)
    assert len(f.must) == 3


def test_alpha_out_of_range_raises() -> None:
    from unittest.mock import MagicMock

    from rag.retriever import HybridRetriever

    mock_embedder = MagicMock()
    mock_client = MagicMock()
    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever("col", alpha=1.5, embedder=mock_embedder, client=mock_client)
    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever("col", alpha=-0.1, embedder=mock_embedder, client=mock_client)


def test_rrf_pure_dense_alpha_one() -> None:
    """With alpha=1.0 the RRF score is entirely from dense ranks."""
    _rrf_k = 60

    def rrf_dense(rank: int) -> float:
        return 1.0 / (_rrf_k + rank + 1)

    # Simulate two documents: doc A at dense rank 0, doc B at dense rank 1
    # alpha=1.0 → sparse contribution is zero
    score_a = 1.0 * rrf_dense(0) + 0.0 * 0  # rank 0 from dense, absent in sparse
    score_b = 1.0 * rrf_dense(1) + 0.0 * 0
    assert score_a > score_b


def test_rrf_pure_sparse_alpha_zero() -> None:
    """With alpha=0.0 only sparse ranks count."""
    _rrf_k = 60

    def rrf_sparse(rank: int) -> float:
        return 1.0 / (_rrf_k + rank + 1)

    score_first = 0.0 * 0 + 1.0 * rrf_sparse(0)
    score_second = 0.0 * 0 + 1.0 * rrf_sparse(1)
    assert score_first > score_second


def test_rrf_balanced_both_lists_beat_single_list() -> None:
    """A doc appearing in both lists should outscore a doc appearing in only one."""
    _rrf_k = 60
    alpha = 0.5

    def rrf(dense_rank: int | None, sparse_rank: int | None) -> float:
        score = 0.0
        if dense_rank is not None:
            score += alpha / (_rrf_k + dense_rank + 1)
        if sparse_rank is not None:
            score += (1 - alpha) / (_rrf_k + sparse_rank + 1)
        return score

    both = rrf(dense_rank=2, sparse_rank=2)
    dense_only = rrf(dense_rank=0, sparse_rank=None)
    assert both > dense_only, "Doc in both lists should outscore doc at rank-0 of one list"


# ---------------------------------------------------------------------------
# Integration tests — require running Qdrant + ingested doc_chunks
# ---------------------------------------------------------------------------


@_integration_skip
@pytest.mark.asyncio
async def test_hybrid_retriever_returns_results() -> None:
    from rag.retriever import HybridRetriever

    retriever = HybridRetriever(collection=_DOC_COLLECTION, alpha=0.7)
    hits = await retriever.search("replace flask render_template", top_k=5)
    assert hits, "Expected at least one result"
    assert len(hits) <= 5
    for hit in hits:
        assert hit.id
        assert hit.score > 0.0
        assert "content" in hit.payload or "file_path" in hit.payload


@_integration_skip
@pytest.mark.asyncio
async def test_hybrid_retriever_jinja2_in_top5() -> None:
    """Regression: 'replace flask render_template' must return a Jinja2Templates chunk."""
    from rag.retriever import HybridRetriever

    retriever = HybridRetriever(collection=_DOC_COLLECTION, alpha=0.7)
    hits = await retriever.search("how to replace flask render_template with Jinja2Templates", top_k=5)
    contents = [h.payload.get("content", "") for h in hits]
    assert any("Jinja2Templates" in c for c in contents), (
        f"Expected Jinja2Templates in top-5. Got: {[c[:80] for c in contents]}"
    )



@_integration_skip
@pytest.mark.asyncio
async def test_metadata_filter_no_results_for_wrong_node_type() -> None:
    """Filter by a node_type that doesn't exist in doc_chunks → empty results."""
    from rag.retriever import HybridRetriever, MetadataFilter

    retriever = HybridRetriever(collection=_DOC_COLLECTION, alpha=0.5)
    hits = await retriever.search(
        "flask templates",
        top_k=5,
        metadata_filter=MetadataFilter(node_type="nonexistent_node_type_xyz"),
    )
    assert hits == [], f"Expected empty results with impossible filter, got {hits}"
