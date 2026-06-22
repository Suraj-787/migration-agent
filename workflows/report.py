"""Persist MigrationReport to Postgres migration_reports table.

Called by finalize_node after every migration run. Errors are swallowed so a
DB hiccup never prevents the graph from completing.
"""
from __future__ import annotations

import json
import os

from loguru import logger

from workflows.state import MigrationReport


def _dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS migration_reports (
    thread_id            TEXT PRIMARY KEY,
    total_tasks          INTEGER          NOT NULL,
    succeeded            INTEGER          NOT NULL,
    failed               INTEGER          NOT NULL,
    success_rate         DOUBLE PRECISION NOT NULL,
    total_tokens_used    INTEGER          NOT NULL,
    estimated_cost_usd   DOUBLE PRECISION NOT NULL,
    per_module_branch_names JSONB         NOT NULL DEFAULT '{}',
    duration_seconds     DOUBLE PRECISION NOT NULL,
    final_status         TEXT             NOT NULL,
    created_at           TIMESTAMPTZ      DEFAULT NOW()
)
"""

_UPSERT = """
INSERT INTO migration_reports (
    thread_id, total_tasks, succeeded, failed, success_rate,
    total_tokens_used, estimated_cost_usd, per_module_branch_names,
    duration_seconds, final_status
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (thread_id) DO UPDATE SET
    total_tasks          = EXCLUDED.total_tasks,
    succeeded            = EXCLUDED.succeeded,
    failed               = EXCLUDED.failed,
    success_rate         = EXCLUDED.success_rate,
    total_tokens_used    = EXCLUDED.total_tokens_used,
    estimated_cost_usd   = EXCLUDED.estimated_cost_usd,
    per_module_branch_names = EXCLUDED.per_module_branch_names,
    duration_seconds     = EXCLUDED.duration_seconds,
    final_status         = EXCLUDED.final_status
"""


async def persist_report(report: MigrationReport) -> None:
    """Insert or update a MigrationReport row. Silently logs on failure."""
    try:
        import asyncpg  # type: ignore[import-untyped]

        conn: asyncpg.Connection = await asyncpg.connect(_dsn(), timeout=10.0)
        try:
            await conn.execute(_CREATE_TABLE)
            await conn.execute(
                _UPSERT,
                report.thread_id,
                report.total_tasks,
                report.succeeded,
                report.failed,
                report.success_rate,
                report.total_tokens_used,
                report.estimated_cost_usd,
                json.dumps(report.per_module_branch_names),
                report.duration_seconds,
                report.final_status,
            )
        finally:
            await conn.close()
        logger.info(
            "[report] Persisted MigrationReport thread_id={} status={} succeeded={}/{}",
            report.thread_id,
            report.final_status,
            report.succeeded,
            report.total_tasks,
        )
    except Exception as exc:
        logger.warning("[report] Failed to persist report (non-fatal): {}", exc)
