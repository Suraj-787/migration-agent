"""Redis-backed cost accumulator and migration_runs Postgres table writer.

Each LLM call atomically increments Redis counters keyed by thread_id.
At run end, finalize_node calls flush_run_cost() to UPSERT a row in
migration_runs (idempotent, safe for retries).
"""
from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

_TTL_SECONDS = 86_400  # keep Redis keys for 24 h


def _dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


def _redis_client():  # type: ignore[return]
    from redis.asyncio import Redis  # type: ignore[import-untyped]

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(url, decode_responses=True)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


async def increment_run_cost(
    thread_id: str,
    model_key: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> float:
    """Atomically increment cost counters for *thread_id*. Returns new total cost."""
    try:
        async with _redis_client() as r:
            prefix = f"migration_cost:{thread_id}"
            pipe = r.pipeline()
            pipe.incrbyfloat(f"{prefix}:total", cost_usd)
            pipe.incrby(f"{prefix}:input_tokens", input_tokens)
            pipe.incrby(f"{prefix}:output_tokens", output_tokens)
            pipe.incrbyfloat(f"{prefix}:model:{model_key}", cost_usd)
            pipe.expire(f"{prefix}:total", _TTL_SECONDS)
            pipe.expire(f"{prefix}:input_tokens", _TTL_SECONDS)
            pipe.expire(f"{prefix}:output_tokens", _TTL_SECONDS)
            pipe.expire(f"{prefix}:model:{model_key}", _TTL_SECONDS)
            results = await pipe.execute()
            return float(results[0])
    except Exception as exc:
        logger.warning("[cost] Redis increment failed (non-fatal): {}", exc)
        return 0.0


async def get_run_cost(thread_id: str) -> float:
    """Return the current accumulated estimated cost for *thread_id*."""
    try:
        async with _redis_client() as r:
            val = await r.get(f"migration_cost:{thread_id}:total")
            return float(val) if val else 0.0
    except Exception:
        return 0.0


async def get_run_breakdown(thread_id: str) -> dict[str, Any]:
    """Return cost/token summary dict for *thread_id* from Redis."""
    try:
        async with _redis_client() as r:
            prefix = f"migration_cost:{thread_id}"
            total = await r.get(f"{prefix}:total")
            input_tok = await r.get(f"{prefix}:input_tokens")
            output_tok = await r.get(f"{prefix}:output_tokens")

            model_breakdown: dict[str, float] = {}
            async for key in r.scan_iter(f"{prefix}:model:*"):
                val = await r.get(key)
                model_name = key.removeprefix(f"{prefix}:model:")
                model_breakdown[model_name] = float(val) if val else 0.0

            return {
                "total_cost_usd": float(total) if total else 0.0,
                "total_input_tokens": int(input_tok) if input_tok else 0,
                "total_output_tokens": int(output_tok) if output_tok else 0,
                "model_breakdown": model_breakdown,
            }
    except Exception as exc:
        logger.warning("[cost] Redis get_run_breakdown failed: {}", exc)
        return {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "model_breakdown": {},
        }


async def mark_ceiling_exceeded(thread_id: str) -> None:
    """Set Redis flag indicating the cost ceiling was exceeded for *thread_id*."""
    try:
        async with _redis_client() as r:
            await r.set(
                f"migration_cost:{thread_id}:ceiling_hit", "1", ex=_TTL_SECONDS
            )
    except Exception as exc:
        logger.warning("[cost] Redis mark_ceiling_exceeded failed: {}", exc)


async def ceiling_was_exceeded(thread_id: str) -> bool:
    """Return True if the cost ceiling was hit for this run."""
    try:
        async with _redis_client() as r:
            return bool(await r.get(f"migration_cost:{thread_id}:ceiling_hit"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Postgres migration_runs table
# ---------------------------------------------------------------------------

_CREATE_MIGRATION_RUNS = """
CREATE TABLE IF NOT EXISTS migration_runs (
    thread_id            TEXT PRIMARY KEY,
    run_id               TEXT,
    total_input_tokens   INTEGER          NOT NULL DEFAULT 0,
    total_output_tokens  INTEGER          NOT NULL DEFAULT 0,
    estimated_cost_usd   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    model_breakdown      JSONB            NOT NULL DEFAULT '{}',
    started_at           TIMESTAMPTZ,
    completed_at         TIMESTAMPTZ      DEFAULT NOW()
)
"""

_UPSERT_MIGRATION_RUNS = """
INSERT INTO migration_runs (
    thread_id, run_id, total_input_tokens, total_output_tokens,
    estimated_cost_usd, model_breakdown, started_at, completed_at
) VALUES ($1, $2, $3, $4, $5, $6, to_timestamp($7), NOW())
ON CONFLICT (thread_id) DO UPDATE SET
    total_input_tokens  = EXCLUDED.total_input_tokens,
    total_output_tokens = EXCLUDED.total_output_tokens,
    estimated_cost_usd  = EXCLUDED.estimated_cost_usd,
    model_breakdown     = EXCLUDED.model_breakdown,
    completed_at        = NOW()
"""


async def flush_run_cost(
    thread_id: str,
    run_id: str,
    started_at: float,
) -> None:
    """Write Redis cost counters to the migration_runs Postgres table (UPSERT)."""
    breakdown = await get_run_breakdown(thread_id)
    try:
        import asyncpg  # type: ignore[import-untyped]

        conn: asyncpg.Connection = await asyncpg.connect(_dsn(), timeout=10.0)
        try:
            await conn.execute(_CREATE_MIGRATION_RUNS)
            await conn.execute(
                _UPSERT_MIGRATION_RUNS,
                thread_id,
                run_id,
                breakdown["total_input_tokens"],
                breakdown["total_output_tokens"],
                breakdown["total_cost_usd"],
                json.dumps(breakdown["model_breakdown"]),
                started_at,
            )
        finally:
            await conn.close()
        logger.info(
            "[cost] Flushed run cost — thread_id={} cost=${:.6f}",
            thread_id,
            breakdown["total_cost_usd"],
        )
    except Exception as exc:
        logger.warning("[cost] Failed to flush run cost to Postgres (non-fatal): {}", exc)
