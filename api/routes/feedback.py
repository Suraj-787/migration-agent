"""POST /feedback/pattern — record a successful before/after migration pair."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from rag.pattern_store import MigrationPattern, PatternStore

router = APIRouter(tags=["feedback"])

_STORE: PatternStore | None = None


def _get_store() -> PatternStore:
    global _STORE
    if _STORE is None:
        _STORE = PatternStore()
    return _STORE


class PatternFeedbackRequest(BaseModel):
    before_code: str
    after_code: str
    migration_type: str
    source_framework: str
    target_framework: str


class PatternFeedbackResponse(BaseModel):
    id: str
    migration_type: str


@router.post("/feedback/pattern", response_model=PatternFeedbackResponse)
async def record_pattern(req: PatternFeedbackRequest) -> PatternFeedbackResponse:
    """Record a successful migration pattern.

    If a semantically identical pattern already exists (cosine similarity >= 0.95),
    its success_count is incremented rather than inserting a duplicate.
    """
    if not req.before_code.strip():
        raise HTTPException(status_code=422, detail="before_code must not be empty")
    if not req.after_code.strip():
        raise HTTPException(status_code=422, detail="after_code must not be empty")
    if not req.migration_type.strip():
        raise HTTPException(status_code=422, detail="migration_type must not be empty")

    pattern = MigrationPattern(
        before_code=req.before_code,
        after_code=req.after_code,
        migration_type=req.migration_type,
        source_framework=req.source_framework,
        target_framework=req.target_framework,
    )

    try:
        point_id = await _get_store().record_success(pattern)
    except Exception as exc:
        logger.error("Failed to record pattern '{}': {}", req.migration_type, exc)
        raise HTTPException(status_code=502, detail=f"Pattern store error: {exc}") from exc

    return PatternFeedbackResponse(id=point_id, migration_type=req.migration_type)
