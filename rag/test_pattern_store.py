"""Tests for rag/pattern_store.py.

Unit tests: pure logic, no I/O.
Integration tests: require Qdrant (docker compose up -d) + VOYAGE_API_KEY.
Run: uv run pytest rag/test_pattern_store.py -xvs
"""
from __future__ import annotations

import os
import uuid

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


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
# Unit tests — no I/O
# ---------------------------------------------------------------------------


def test_signature_key_format() -> None:
    from rag.pattern_store import MigrationPattern, _signature_key

    p = MigrationPattern(
        before_code="@app.route('/x')",
        after_code="@router.get('/x')",
        migration_type="route_decorator",
        source_framework="flask",
        target_framework="fastapi",
    )
    key = _signature_key(p)
    assert " -> " in key
    assert key.startswith("@app.route('/x')")
    assert key.endswith("route_decorator")


def test_pattern_id_is_deterministic() -> None:
    from rag.pattern_store import _pattern_id

    key = "some before code -> route_decorator"
    assert _pattern_id(key) == _pattern_id(key)
    assert _pattern_id(key) != _pattern_id("different key -> something")


def test_pattern_id_is_valid_uuid() -> None:
    from rag.pattern_store import _pattern_id

    raw = _pattern_id("@app.route('/x') -> route_decorator")
    parsed = uuid.UUID(raw)
    assert str(parsed) == raw


def test_to_payload_from_payload_roundtrip() -> None:
    from rag.pattern_store import MigrationPattern, _from_payload, _to_payload

    original = MigrationPattern(
        before_code="flask.abort(404)",
        after_code="raise HTTPException(status_code=404)",
        migration_type="http_exception",
        source_framework="flask",
        target_framework="fastapi",
        success_count=3,
    )
    payload = _to_payload(original)
    reconstructed = _from_payload(payload)

    assert reconstructed.before_code == original.before_code
    assert reconstructed.after_code == original.after_code
    assert reconstructed.migration_type == original.migration_type
    assert reconstructed.source_framework == original.source_framework
    assert reconstructed.target_framework == original.target_framework
    assert reconstructed.success_count == original.success_count


def test_migration_pattern_defaults() -> None:
    from rag.pattern_store import MigrationPattern

    p = MigrationPattern(
        before_code="x",
        after_code="y",
        migration_type="m",
        source_framework="flask",
        target_framework="fastapi",
    )
    assert p.success_count == 1
    assert p.embedded_signature == []


def test_canonical_patterns_count() -> None:
    from rag.pattern_store import _CANONICAL_PATTERNS

    assert len(_CANONICAL_PATTERNS) == 10
    types = {p.migration_type for p in _CANONICAL_PATTERNS}
    assert "route_decorator" in types
    assert "http_exception" in types
    assert all(p.source_framework == "flask" for p in _CANONICAL_PATTERNS)
    assert all(p.target_framework == "fastapi" for p in _CANONICAL_PATTERNS)


# ---------------------------------------------------------------------------
# Integration tests — require Qdrant + VOYAGE_API_KEY
# ---------------------------------------------------------------------------


@_integration_skip
@pytest.mark.asyncio
async def test_record_and_duplicate_detection() -> None:
    """Insert a pattern, verify retrieval, then re-insert to confirm duplicate detection.

    Sleeps between Voyage API calls to stay under the 3-RPM free-tier limit.
    """
    import asyncio

    from rag.pattern_store import MigrationPattern, PatternStore

    store = PatternStore()
    pattern = MigrationPattern(
        before_code="flask.abort(500)",
        after_code="raise HTTPException(status_code=500)",
        migration_type="http_exception",
        source_framework="flask",
        target_framework="fastapi",
    )

    # First insert — should create a new point
    id1 = await store.record_success(pattern)
    assert id1, "Expected a non-empty point ID"

    # Space out API calls to stay under 3 RPM
    await asyncio.sleep(22)

    # Duplicate insert — same pattern, should return same ID and increment count
    id2 = await store.record_success(pattern)
    assert id1 == id2, f"Duplicate should return same ID, got {id1!r} vs {id2!r}"

    await asyncio.sleep(22)

    # Retrieval smoke-check
    results = await store.find_similar("flask.abort(500)", "http_exception", k=3)
    assert results, "Expected at least one result after inserting"
    assert results[0].migration_type == "http_exception"


@_integration_skip
@pytest.mark.asyncio
async def test_find_similar_route_decorator_top1() -> None:
    """Query with a slightly different route input; top-1 must be the route_decorator pattern.

    Seeds the store first (batched embed — 2 API calls with built-in 22s sleep between
    batches), then queries.
    """
    import asyncio

    from rag.pattern_store import PatternStore

    store = PatternStore()
    await store.seed_canonical()

    # One additional rate-limit gap before the find_similar embed call
    await asyncio.sleep(22)

    # Slightly different from the seed: includes methods=['GET']
    results = await store.find_similar(
        query_code="@app.route('/users', methods=['GET'])\ndef list_users(): ...",
        migration_type="route_decorator",
        k=3,
    )
    assert results, "Expected at least one result"
    assert results[0].migration_type == "route_decorator", (
        f"Expected top-1 to be 'route_decorator', got '{results[0].migration_type}'"
    )
    assert "fastapi" in results[0].target_framework.lower()
