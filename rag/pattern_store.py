"""PatternStore: record and retrieve learned before/after migration patterns.

Patterns are stored in the `migration_patterns` Qdrant collection.
The dense vector is the 1024-dim Voyage embedding of `before_code + " -> " + migration_type`.
Duplicate detection uses cosine similarity >= 0.95 — matches increment success_count instead
of inserting a new point.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastembed import SparseTextEmbedding
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from rag.collections import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_qdrant_client,
)
from rag.embedder import VoyageEmbedder

_COLLECTION = "migration_patterns"
_DUPLICATE_THRESHOLD = 0.95

_BM25_MODEL: SparseTextEmbedding | None = None


def _get_bm25() -> SparseTextEmbedding:
    global _BM25_MODEL
    if _BM25_MODEL is None:
        logger.info("Loading BM25 model for pattern store (first use)")
        _BM25_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _BM25_MODEL


class MigrationPattern(BaseModel):
    before_code: str
    after_code: str
    migration_type: str
    source_framework: str
    target_framework: str
    success_count: int = 1
    embedded_signature: list[float] = Field(default_factory=list)


def _signature_key(pattern: MigrationPattern) -> str:
    return f"{pattern.before_code} -> {pattern.migration_type}"


def _pattern_id(signature_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, signature_key))


def _to_payload(pattern: MigrationPattern) -> dict[str, Any]:
    return {
        "before_code": pattern.before_code,
        "after_code": pattern.after_code,
        "migration_type": pattern.migration_type,
        "source_framework": pattern.source_framework,
        "target_framework": pattern.target_framework,
        "success_count": pattern.success_count,
    }


def _from_payload(payload: dict[str, Any]) -> MigrationPattern:
    return MigrationPattern(
        before_code=payload["before_code"],
        after_code=payload["after_code"],
        migration_type=payload["migration_type"],
        source_framework=payload["source_framework"],
        target_framework=payload["target_framework"],
        success_count=int(payload.get("success_count", 1)),
    )


class PatternStore:
    def __init__(
        self,
        embedder: VoyageEmbedder | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._embedder = embedder or VoyageEmbedder()
        self._client = client or get_qdrant_client()

    async def _ensure_ready(self) -> None:
        await ensure_collection(self._client, _COLLECTION)

    async def record_success(self, pattern: MigrationPattern) -> str:
        """Upsert a pattern into the store.

        If a point with cosine similarity >= 0.95 already exists, increments its
        success_count instead of inserting a duplicate. Returns the point ID.

        Signatures are stored as document embeddings so the duplicate-check search
        (symmetric doc→doc) yields consistent cosine scores near 1.0 for identical text.
        """
        await self._ensure_ready()

        sig_key = _signature_key(pattern)
        # Embed as document — patterns are corpus items, not queries
        dense_vecs = await self._embedder.embed_texts([sig_key])
        dense_vec = dense_vecs[0]

        # Duplicate check — dense search for top-1
        results = await self._client.query_points(
            collection_name=_COLLECTION,
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=1,
            with_payload=True,
        )

        if results.points:
            top = results.points[0]
            if top.score >= _DUPLICATE_THRESHOLD:
                existing_count = int((top.payload or {}).get("success_count", 1))
                await self._client.set_payload(
                    collection_name=_COLLECTION,
                    payload={"success_count": existing_count + 1},
                    points=[top.id],
                )
                logger.debug(
                    "Pattern duplicate (score={:.4f}), incremented success_count to {} for id={}",
                    top.score,
                    existing_count + 1,
                    top.id,
                )
                return str(top.id)

        # New pattern — compute sparse vector and upsert
        sparse_result = list(_get_bm25().embed([sig_key]))[0]
        point_id = _pattern_id(sig_key)

        await self._client.upsert(
            collection_name=_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_result.indices.tolist(),
                            values=sparse_result.values.tolist(),
                        ),
                    },
                    payload=_to_payload(pattern),
                )
            ],
        )
        logger.info(
            "Recorded new migration pattern '{}' (id={})", pattern.migration_type, point_id
        )
        return point_id

    async def find_similar(
        self, query_code: str, migration_type: str, k: int = 3
    ) -> list[MigrationPattern]:
        """Return the k most similar stored patterns for the given code snippet."""
        await self._ensure_ready()

        query_sig = f"{query_code} -> {migration_type}"
        dense_vec = await self._embedder.embed_query(query_sig)

        results = await self._client.query_points(
            collection_name=_COLLECTION,
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=k,
            with_payload=True,
        )

        return [_from_payload(p.payload or {}) for p in results.points if p.payload]

    async def seed_canonical(self) -> int:
        """Seed the store with 10 canonical Flask → FastAPI patterns.

        Idempotent via deterministic uuid5 IDs — re-seeding overwrites existing points
        with the same data (no duplicate rows). Uses a single batched embed call for all
        10 signatures so we stay within the Voyage 3-RPM free-tier limit.

        Returns the number of points upserted (always 10 on first run, 10 on re-runs).
        """
        await self._ensure_ready()

        sig_keys = [_signature_key(p) for p in _CANONICAL_PATTERNS]

        # One batched embed call → _embed_batched handles inter-batch sleep (22s)
        # Batch size 8 → two API calls: [0:8] sleep 22s [8:10]
        logger.info("Embedding {} canonical patterns (batched)", len(sig_keys))
        dense_vecs = await self._embedder.embed_texts(sig_keys)

        bm25 = _get_bm25()
        sparse_results = list(bm25.embed(sig_keys))

        points = [
            PointStruct(
                id=_pattern_id(sig_key),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload=_to_payload(pattern),
            )
            for pattern, sig_key, dense_vec, sparse in zip(
                _CANONICAL_PATTERNS, sig_keys, dense_vecs, sparse_results
            )
        ]

        await self._client.upsert(collection_name=_COLLECTION, points=points)
        logger.info("Canonical seed complete: {} patterns upserted", len(points))
        return len(points)


# ---------------------------------------------------------------------------
# Canonical seed patterns
# ---------------------------------------------------------------------------

_CANONICAL_PATTERNS: list[MigrationPattern] = [
    MigrationPattern(
        before_code="@app.route('/x')\ndef view():\n    ...",
        after_code="@router.get('/x')\nasync def view():\n    ...",
        migration_type="route_decorator",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="data = flask.request.get_json()",
        after_code="data = await request.json()",
        migration_type="request_body",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="return render_template('x.html', **ctx)",
        after_code="return templates.TemplateResponse(request, 'x.html', ctx)",
        migration_type="template_response",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="from flask import Blueprint\nbp = Blueprint('name', __name__)",
        after_code="from fastapi import APIRouter\nrouter = APIRouter()",
        migration_type="router_setup",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="@login_required\ndef protected():\n    ...",
        after_code="async def protected(user=Depends(get_current_user)):\n    ...",
        migration_type="auth_dependency",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="flask.abort(404)",
        after_code="raise HTTPException(status_code=404)",
        migration_type="http_exception",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="db.session.add(obj)\ndb.session.commit()",
        after_code="async with db.begin():\n    db.add(obj)",
        migration_type="db_session",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="from flask import g\nuser = g.user",
        after_code="user = request.state.user",
        migration_type="request_state",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="flash('Operation successful')",
        after_code="response.headers['X-Flash-Message'] = 'Operation successful'",
        migration_type="flash_message",
        source_framework="flask",
        target_framework="fastapi",
    ),
    MigrationPattern(
        before_code="url_for('view_name', id=1)",
        after_code="request.url_for('view_name', id=1)",
        migration_type="url_generation",
        source_framework="flask",
        target_framework="fastapi",
    ),
]
