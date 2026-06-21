"""Integration test for the ingest pipeline.

Requires a running Qdrant instance. Skips automatically if Qdrant is unreachable.
Run with: uv run pytest rag/test_ingest.py -xvs
"""
from __future__ import annotations

import os

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
FIXTURE_REPO = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "sample_flask_app"
)


def _qdrant_is_up() -> bool:
    try:
        r = httpx.get(f"{QDRANT_URL}/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_is_up(),
    reason="Qdrant not reachable — start docker compose up -d qdrant",
)


@pytest.fixture(scope="module")
async def ingested_collection() -> str:
    """Ingest the sample Flask app and return the collection name."""
    from rag.ingest import ingest_repo

    collection = "code_chunks_test"
    count = await ingest_repo(
        repo_path=os.path.abspath(FIXTURE_REPO),
        collection_name=collection,
        qdrant_url=QDRANT_URL,
    )
    assert count > 0, "Expected at least one chunk to be ingested"
    return collection


@pytest.mark.asyncio
async def test_vectors_exist_after_ingest(ingested_collection: str) -> None:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=QDRANT_URL)
    info = await client.get_collection(ingested_collection)
    assert info.points_count is not None and info.points_count > 0, (
        f"Collection '{ingested_collection}' has no points after ingestion"
    )


@pytest.mark.asyncio
async def test_named_vectors_present(ingested_collection: str) -> None:
    from qdrant_client import AsyncQdrantClient
    from rag.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    client = AsyncQdrantClient(url=QDRANT_URL)
    results, _ = await client.scroll(
        collection_name=ingested_collection,
        limit=1,
        with_vectors=True,
    )
    assert results, "Scroll returned no points"
    point = results[0]
    assert isinstance(point.vector, dict), "Expected named vector dict"
    assert DENSE_VECTOR_NAME in point.vector, f"Missing '{DENSE_VECTOR_NAME}' vector"
    assert SPARSE_VECTOR_NAME in point.vector, f"Missing '{SPARSE_VECTOR_NAME}' vector"


@pytest.mark.asyncio
async def test_payload_fields_present(ingested_collection: str) -> None:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=QDRANT_URL)
    results, _ = await client.scroll(
        collection_name=ingested_collection,
        limit=5,
        with_payload=True,
    )
    required_fields = {"file_path", "language", "node_type", "start_line", "end_line", "content"}
    for point in results:
        payload = point.payload or {}
        missing = required_fields - set(payload.keys())
        assert not missing, f"Point {point.id} missing payload fields: {missing}"


@pytest.mark.asyncio
async def test_dense_vector_dimension(ingested_collection: str) -> None:
    from qdrant_client import AsyncQdrantClient
    from rag.collections import DENSE_VECTOR_NAME

    client = AsyncQdrantClient(url=QDRANT_URL)
    results, _ = await client.scroll(
        collection_name=ingested_collection,
        limit=1,
        with_vectors=True,
    )
    point = results[0]
    assert isinstance(point.vector, dict)
    dense = point.vector[DENSE_VECTOR_NAME]
    assert isinstance(dense, list) and len(dense) == 1024, (
        f"Expected 1024-dim dense vector, got {len(dense) if isinstance(dense, list) else type(dense)}"
    )
