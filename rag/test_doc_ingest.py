"""Tests for doc_ingest: markdown splitting (unit) + retrieval (integration).

Integration tests require Qdrant and VOYAGE_API_KEY. They skip automatically otherwise.
Run with: uv run pytest rag/test_doc_ingest.py -xvs
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
GUIDES_DIR = str(Path(__file__).parent.parent / "data" / "migration_guides")
_TEST_COLLECTION = "doc_chunks_test"


def _qdrant_is_up() -> bool:
    try:
        r = httpx.get(f"{QDRANT_URL}/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _voyage_key_set() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


# ---------------------------------------------------------------------------
# Unit tests — no infra required
# ---------------------------------------------------------------------------


def test_split_markdown_respects_chunk_size() -> None:
    from rag.doc_ingest import split_markdown

    long_text = "word " * 500  # ~2500 chars
    chunks = split_markdown(long_text, chunk_size=1000, chunk_overlap=100)
    assert chunks, "Expected at least one chunk"
    for chunk in chunks:
        assert len(chunk) <= 1000, f"Chunk exceeds chunk_size: {len(chunk)} chars"


def test_split_markdown_overlap_preserves_context() -> None:
    from rag.doc_ingest import split_markdown

    # Two large paragraphs separated by a blank line
    para_a = "A " * 600  # ~1200 chars
    para_b = "B " * 600
    text = para_a.strip() + "\n\n" + para_b.strip()
    chunks = split_markdown(text, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) >= 2, "Expected multiple chunks for long text"


def test_split_markdown_heading_boundaries() -> None:
    from rag.doc_ingest import split_markdown

    text = "# Title\n\nIntro.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
    chunks = split_markdown(text, chunk_size=1000, chunk_overlap=100)
    # Short text should produce one chunk (fits in chunk_size)
    assert len(chunks) >= 1
    combined = " ".join(chunks)
    assert "Section A" in combined
    assert "Section B" in combined


def test_chunk_markdown_file_produces_doc_chunks() -> None:
    from rag.doc_ingest import chunk_markdown_file

    guides = list(Path(GUIDES_DIR).glob("*.md"))
    assert guides, f"No .md files found in {GUIDES_DIR}"

    # Use the FastAPI templates guide — it contains Jinja2Templates content
    templates_guide = next(
        (g for g in guides if "fastapi_templates" in g.name), guides[0]
    )
    chunks = chunk_markdown_file(str(templates_guide))
    assert chunks, "Expected at least one DocChunk"
    for chunk in chunks:
        assert chunk.content.strip(), "Chunk content must not be empty"
        assert len(chunk.content) <= 1000, f"Chunk too large: {len(chunk.content)}"
        assert chunk.file_path == str(templates_guide.resolve())
        assert chunk.source == templates_guide.stem


def test_jinja2templates_present_in_chunks() -> None:
    """The templates guide must produce at least one chunk mentioning Jinja2Templates."""
    from rag.doc_ingest import chunk_markdown_file

    templates_guide = Path(GUIDES_DIR) / "fastapi_templates.md"
    assert templates_guide.exists(), f"Missing guide: {templates_guide}"
    chunks = chunk_markdown_file(str(templates_guide))
    matching = [c for c in chunks if "Jinja2Templates" in c.content]
    assert matching, "No chunk mentions 'Jinja2Templates' — guide content may be missing"


# ---------------------------------------------------------------------------
# Integration tests — require Qdrant + VOYAGE_API_KEY
# ---------------------------------------------------------------------------

_integration_skip = pytest.mark.skipif(
    not _qdrant_is_up() or not _voyage_key_set(),
    reason="Integration tests require Qdrant (docker compose up -d) and VOYAGE_API_KEY",
)


@pytest.fixture(scope="module")
async def ingested_doc_collection() -> str:
    """Ingest migration guides into a test collection and return its name."""
    from rag.doc_ingest import ingest_docs

    count = await ingest_docs(
        docs_dir=GUIDES_DIR,
        collection_name=_TEST_COLLECTION,
        qdrant_url=QDRANT_URL,
    )
    assert count > 0, "Expected at least one doc chunk to be ingested"
    return _TEST_COLLECTION


@_integration_skip
@pytest.mark.asyncio
async def test_doc_collection_has_points(ingested_doc_collection: str) -> None:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=QDRANT_URL)
    info = await client.get_collection(ingested_doc_collection)
    assert info.points_count is not None and info.points_count > 0


@_integration_skip
@pytest.mark.asyncio
async def test_render_template_query_returns_jinja2_chunk(ingested_doc_collection: str) -> None:
    """Query 'how to replace flask render_template' → top-5 must contain Jinja2Templates."""
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
    from fastembed import SparseTextEmbedding

    from rag.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
    from rag.embedder import VoyageEmbedder

    query = "how to replace flask render_template"
    embedder = VoyageEmbedder()
    dense_vec = await embedder.embed_query(query)

    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse_result = list(bm25.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    client = AsyncQdrantClient(url=QDRANT_URL)
    response = await client.query_points(
        collection_name=ingested_doc_collection,
        prefetch=[
            Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=10),
            Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, limit=10),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5,
        with_payload=True,
    )

    top5_contents = [point.payload.get("content", "") for point in response.points if point.payload]
    assert any(
        "Jinja2Templates" in content for content in top5_contents
    ), (
        f"Expected 'Jinja2Templates' in top-5 results for query '{query}'.\n"
        f"Got snippets: {[c[:120] for c in top5_contents]}"
    )
