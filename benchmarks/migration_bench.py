"""Migration benchmark runner.

Triggers migrations via the running HTTP API, polls until terminal status,
collects MigrationReport metrics from Postgres, and writes a markdown report.

Usage:
    python -m benchmarks.migration_bench [--repos all|fixture|local]

Options:
    --repos all      Run all 5 repos (includes network clones, 30+ min each)
    --repos fixture  Repos 4+5 only — local paths, no network needed (default fast baseline)
    --repos local    Alias for fixture

Environment:
    API_BASE_URL     Override API address (default: http://localhost:8000)
    POSTGRES_*       Standard DB env vars (same as the app)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from benchmarks.repos import REPOS, BenchmarkRepo

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent
_API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
_POLL_INTERVAL = 5  # seconds between status polls
_POLL_TIMEOUT = 1800  # 30-minute hard stop
_TERMINAL = {"success", "partial", "failed", "cost_ceiling_exceeded"}


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class BenchmarkResult(BaseModel):
    repo_name: str
    migration_type: str
    thread_id: str | None = None
    final_status: str = "error"
    module_pass_rate: float | None = None
    avg_attempts: float | None = None
    total_cost_usd: float | None = None
    total_time_seconds: float | None = None
    diff_similarity: str = "N/A — no human PR diff available"
    error: str | None = None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _migration_params(migration_type: str) -> dict[str, Any]:
    match migration_type:
        case "flask_to_fastapi":
            return {
                "source_framework": "flask",
                "source_version": "latest",
                "target_spec": {
                    "target_framework": "fastapi",
                    "target_version": "0.115",
                    "custom_rules": [],
                },
            }
        case "python2_to_python3":
            return {
                "source_framework": "python2",
                "source_version": "2.7",
                "target_spec": {
                    "target_framework": "python3",
                    "target_version": "3.11",
                    "custom_rules": [],
                },
            }
        case "django_fbv_to_cbv":
            return {
                "source_framework": "django",
                "source_version": "latest",
                "target_spec": {
                    "target_framework": "django_cbv",
                    "target_version": "4.2",
                    "custom_rules": [],
                },
            }
        case _:
            raise ValueError(f"Unknown migration_type: {migration_type!r}")


async def _start_migration(
    client: httpx.AsyncClient, repo_path: str, migration_type: str
) -> str:
    """POST /migrations and return thread_id."""
    payload: dict[str, Any] = {"repo_path": repo_path, **_migration_params(migration_type)}
    resp = await client.post(f"{_API_BASE}/migrations", json=payload, timeout=60.0)
    resp.raise_for_status()
    return str(resp.json()["thread_id"])


async def _poll_until_done(client: httpx.AsyncClient, thread_id: str) -> str:
    """Poll GET /migrations/{thread_id} until terminal status. Returns final_status."""
    deadline = time.monotonic() + _POLL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            resp = await client.get(
                f"{_API_BASE}/migrations/{thread_id}", timeout=10.0
            )
        except httpx.TransportError as exc:
            logger.warning("[poll] Transport error for {}: {} — retrying", thread_id, exc)
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        if resp.status_code == 404:
            # Migration not yet in checkpointer; keep waiting.
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        resp.raise_for_status()
        status: str = resp.json()["final_status"]
        logger.info("[poll] thread_id={} status={}", thread_id, status)
        if status in _TERMINAL:
            return status
        await asyncio.sleep(_POLL_INTERVAL)

    raise TimeoutError(f"Migration {thread_id} did not complete within {_POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# Metric fetchers (Postgres + checkpointer)
# ---------------------------------------------------------------------------


def _pg_dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


async def _fetch_report_row(thread_id: str) -> dict[str, Any] | None:
    """Read metrics from migration_reports; fall back to migration_runs if no row found."""
    try:
        conn: asyncpg.Connection = await asyncpg.connect(_pg_dsn(), timeout=10.0)
        try:
            row = await conn.fetchrow(
                "SELECT total_tasks, succeeded, estimated_cost_usd, duration_seconds "
                "FROM migration_reports WHERE thread_id = $1",
                thread_id,
            )
            if row:
                return dict(row)

            # Fallback: migration_runs is always written by flush_run_cost.
            row = await conn.fetchrow(
                "SELECT succeeded, total_tasks, estimated_cost_usd "
                "FROM migration_runs WHERE thread_id = $1",
                thread_id,
            )
            if row:
                total: int = row["total_tasks"] or 0
                succeeded: int = row["succeeded"] or 0
                return {
                    "total_tasks": total,
                    "succeeded": succeeded,
                    "estimated_cost_usd": float(row["estimated_cost_usd"]),
                    "duration_seconds": None,
                }
            return None
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("[bench] Could not fetch report row for {}: {}", thread_id, exc)
        return None


async def _fetch_avg_attempts(thread_id: str) -> float | None:
    """Read attempt_count from the final LangGraph checkpoint channel_values."""
    try:
        from langchain_core.runnables import RunnableConfig
        from workflows.checkpointer import checkpointer_context

        async with checkpointer_context() as cp:
            config = RunnableConfig(configurable={"thread_id": thread_id})
            tup = await cp.aget_tuple(config)
            if tup is None:
                return None
            attempt_count: dict[str, int] = (
                tup.checkpoint.get("channel_values", {}).get("attempt_count") or {}
            )
            if not attempt_count:
                return None
            return sum(attempt_count.values()) / len(attempt_count)
    except Exception as exc:
        logger.warning("[bench] Could not fetch attempt_count for {}: {}", thread_id, exc)
        return None


# ---------------------------------------------------------------------------
# Repo preparation
# ---------------------------------------------------------------------------


async def _clone_repo(url: str, dest: Path) -> None:
    """Shallow-clone url into dest."""
    logger.info("Cloning {} → {}", url, dest)
    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth=1", url, str(dest),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {stderr.decode().strip()}")


async def _prepare_repo(repo: BenchmarkRepo, tmpdir: Path) -> str:
    """Return the absolute repo_path to submit to POST /migrations."""
    if repo.local_path is not None:
        resolved = repo.resolve_path(_PROJECT_ROOT)
        assert resolved is not None
        return resolved

    clone_dest = tmpdir / repo.name
    await _clone_repo(repo.url, clone_dest)  # type: ignore[arg-type]
    target = clone_dest / repo.subdir if repo.subdir else clone_dest
    return str(target.resolve())


# ---------------------------------------------------------------------------
# Per-repo runner
# ---------------------------------------------------------------------------


async def run_repo(
    client: httpx.AsyncClient,
    repo: BenchmarkRepo,
    tmpdir: Path,
) -> BenchmarkResult:
    result = BenchmarkResult(repo_name=repo.name, migration_type=repo.migration_type)
    try:
        repo_path = await _prepare_repo(repo, tmpdir)
        logger.info("Starting migration — repo={} path={}", repo.name, repo_path)

        thread_id = await _start_migration(client, repo_path, repo.migration_type)
        result.thread_id = thread_id

        final_status = await _poll_until_done(client, thread_id)
        result.final_status = final_status

        report_row = await _fetch_report_row(thread_id)
        if report_row:
            total: int = report_row["total_tasks"]
            succeeded: int = report_row["succeeded"]
            result.module_pass_rate = succeeded / total if total > 0 else 0.0
            result.total_cost_usd = float(report_row["estimated_cost_usd"])
            if report_row["duration_seconds"] is not None:
                result.total_time_seconds = float(report_row["duration_seconds"])

        result.avg_attempts = await _fetch_avg_attempts(thread_id)

    except Exception as exc:
        logger.error("Benchmark error for {}: {}", repo.name, exc)
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _agent_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _fmt_rate(v: float | None) -> str:
    return f"{v:.2%}" if v is not None else "—"


def _fmt_cost(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "—"


def _fmt_time(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _fmt_att(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _build_report(results: list[BenchmarkResult], run_ts: str) -> str:
    lines: list[str] = [
        f"# Migration Benchmark Report — {run_ts}",
        "",
        "## System Info",
        "",
        f"- **Date:** {run_ts}",
        f"- **Migration-Agent version:** {_agent_version()}",
        "- **Default LLM:** Gemini Flash",
        "- **Transform LLM:** Qwen3 Coder (OpenRouter)",
        "- **Classification LLM:** Groq Llama",
        "",
        "## Results",
        "",
        "| Repo | Migration Type | Module Pass Rate | Avg Attempts | Total Cost (USD) | Time (s) | Diff Similarity | Status |",
        "|------|---------------|:---------------:|:------------:|:---------------:|:--------:|:---------------:|--------|",
    ]

    pass_rates: list[float] = []
    total_cost = 0.0
    total_time = 0.0

    for r in results:
        error_suffix = f" ⚠ `{r.error[:60]}`" if r.error else ""
        lines.append(
            f"| {r.repo_name} | {r.migration_type} | {_fmt_rate(r.module_pass_rate)} "
            f"| {_fmt_att(r.avg_attempts)} | {_fmt_cost(r.total_cost_usd)} "
            f"| {_fmt_time(r.total_time_seconds)} | {r.diff_similarity} "
            f"| {r.final_status}{error_suffix} |"
        )
        if r.module_pass_rate is not None:
            pass_rates.append(r.module_pass_rate)
        if r.total_cost_usd is not None:
            total_cost += r.total_cost_usd
        if r.total_time_seconds is not None:
            total_time += r.total_time_seconds

    if pass_rates:
        agg_rate = sum(pass_rates) / len(pass_rates)
        lines.append(
            f"| **AGGREGATE** | — | **{_fmt_rate(agg_rate)}** | — "
            f"| **{_fmt_cost(total_cost)}** | **{_fmt_time(total_time)}** | N/A | — |"
        )

    target_ok = bool(pass_rates) and (sum(pass_rates) / len(pass_rates)) > 0.70
    verdict = "**PASS**" if target_ok else "**FAIL**"

    lines += [
        "",
        "## Target",
        "",
        f"- `module_pass_rate > 0.70`: {verdict}",
        "",
        "> `diff_similarity` is N/A — no human PR diff available for these repos.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _main(repo_filter: str) -> None:
    if repo_filter in ("fixture", "local"):
        repos = [r for r in REPOS if r.local_path is not None]
    else:
        repos = list(REPOS)

    logger.info("Benchmark starting — {} repos, filter={}", len(repos), repo_filter)

    results_dir = _PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with httpx.AsyncClient() as client:
            results: list[BenchmarkResult] = []
            for repo in repos:
                logger.info("─── {} ───", repo.name)
                result = await run_repo(client, repo, Path(tmpdir))
                results.append(result)
                logger.info(
                    "  status={}  pass_rate={}  cost={}  time={}s",
                    result.final_status,
                    _fmt_rate(result.module_pass_rate),
                    _fmt_cost(result.total_cost_usd),
                    _fmt_time(result.total_time_seconds),
                )

    report = _build_report(results, run_ts)
    out_path = results_dir / f"REPORT_{run_ts}.md"
    out_path.write_text(report, encoding="utf-8")

    logger.info("Report written → {}", out_path)
    print(f"\n{report}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run migration benchmarks and produce a markdown report."
    )
    parser.add_argument(
        "--repos",
        choices=["all", "fixture", "local"],
        default="all",
        help=(
            "Which repos to benchmark: "
            "'all' = all 5 repos (needs network); "
            "'fixture'/'local' = repos 4+5 only (local paths, fast baseline)"
        ),
    )
    args = parser.parse_args()
    asyncio.run(_main(args.repos))


if __name__ == "__main__":
    main()
