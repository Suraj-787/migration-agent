"""Cost report CLI.

Usage:
    python -m api.tools.cost_report --days 7

Outputs an ASCII table summarising migration runs from the last N days:

    date       | runs | modules_attempted | modules_succeeded | total_tokens | estimated_cost_usd
    -----------+------+-------------------+-------------------+--------------+--------------------
    2026-06-23 |    3 |                12 |                10 |       48 231 |             $0.0000
    ...

Data is pulled from migration_runs (cost + token counts) LEFT JOINed with
migration_reports (module counts) on thread_id.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime


def _dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


_QUERY = """
SELECT
    DATE(COALESCE(r.started_at, rp.created_at) AT TIME ZONE 'UTC') AS day,
    COUNT(r.thread_id)                                              AS runs,
    COALESCE(SUM(rp.total_tasks), 0)::BIGINT                       AS modules_attempted,
    COALESCE(SUM(rp.succeeded), 0)::BIGINT                         AS modules_succeeded,
    COALESCE(SUM(r.total_input_tokens + r.total_output_tokens), 0)::BIGINT AS total_tokens,
    COALESCE(SUM(r.estimated_cost_usd), 0.0)                       AS estimated_cost_usd
FROM migration_runs r
LEFT JOIN migration_reports rp USING (thread_id)
WHERE COALESCE(r.started_at, rp.created_at) >= NOW() - ($1 || ' days')::INTERVAL
GROUP BY day
ORDER BY day DESC
"""


async def _fetch(days: int) -> list[dict]:  # type: ignore[type-arg]
    import asyncpg  # type: ignore[import-untyped]

    conn: asyncpg.Connection = await asyncpg.connect(_dsn(), timeout=10.0)
    try:
        rows = await conn.fetch(_QUERY, str(days))
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def _render(rows: list[dict]) -> str:  # type: ignore[type-arg]
    headers = [
        "date", "runs", "modules_attempted", "modules_succeeded",
        "total_tokens", "estimated_cost_usd",
    ]
    col_widths = [len(h) for h in headers]

    formatted: list[list[str]] = []
    for row in rows:
        day_str = row["day"].strftime("%Y-%m-%d") if isinstance(row["day"], datetime) else str(row["day"])
        cells = [
            day_str,
            str(row["runs"]),
            str(row["modules_attempted"]),
            str(row["modules_succeeded"]),
            f"{row['total_tokens']:,}",
            f"${row['estimated_cost_usd']:.4f}",
        ]
        formatted.append(cells)
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "-+-".join("-" * w for w in col_widths)
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines = [header_row, sep]
    for cells in formatted:
        lines.append(" | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)))
    return "\n".join(lines)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Migration cost report")
    parser.add_argument("--days", type=int, default=7, help="Look-back window in days")
    args = parser.parse_args()

    rows = asyncio.run(_fetch(args.days))
    if not rows:
        print(f"No migration runs found in the last {args.days} day(s).")
        return
    print(f"\nMigration cost report — last {args.days} day(s)\n")
    print(_render(rows))
    print()


if __name__ == "__main__":
    main()
