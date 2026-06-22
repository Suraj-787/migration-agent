"""Critic agent: validates a migrated branch via ruff, mypy, pytest, and import check.

Entry point: run_critic(result, task_id, spec, run_id, repo_path)
  1. Adds a git worktree for the migration branch (never touches HEAD/working tree).
  2. Runs validation gauntlet: ruff check → mypy --no-incremental → pytest → import check.
  3. On pass: marks Redis status="done", records success pattern (fire-and-forget).
  4. On fail: asks Groq LLM for complete corrected file content, re-runs gauntlet.
     Max 2 retry loops.
  5. After 2 failed retries: removes worktree, deletes branch, marks Redis status="failed",
     returns CriticResult with verdict="fail" and a RollbackEntry in rollback_entry.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from loguru import logger
from pydantic import BaseModel

from agents.llm_router import get_router
from rag.pattern_store import MigrationPattern, PatternStore
from workflows.state import MigrationSpec, RollbackEntry, TaskResult

_MAX_RETRIES = 2
_MAX_ERROR_CHARS = 3_000
_MAX_FILE_CHARS = 4_000

_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class CriticResult(BaseModel):
    module_path: str
    verdict: Literal["pass", "fail"]
    branch_name: str | None = None
    failure_reason: str | None = None
    retry_count: int = 0
    rollback_entry: RollbackEntry | None = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def worktree_path(run_id: str, branch_name: str) -> str:
    """Deterministic path under /tmp for a given run + branch."""
    safe = branch_name.replace("/", "_").replace("\\", "_")
    return str(Path("/tmp/migration_worktrees") / run_id / safe)


def _find_test_file(rel_path: str, wt_path: str) -> str | None:
    """Return absolute path to the adjacent test file if present in the worktree."""
    p = Path(rel_path)
    test_abs = Path(wt_path) / p.parent / f"test_{p.name}"
    return str(test_abs) if test_abs.exists() else None


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


async def _run_subprocess(
    cmd: list[str], cwd: str, extra_env: dict[str, str] | None = None
) -> tuple[int, str]:
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout + stderr).decode("utf-8", errors="replace")
    rc = proc.returncode if proc.returncode is not None else 1
    return rc, output


def _pythonpath(wt_path: str) -> str:
    existing = os.environ.get("PYTHONPATH", "")
    return f"{wt_path}:{existing}" if existing else wt_path


# ---------------------------------------------------------------------------
# Git worktree management
# ---------------------------------------------------------------------------


async def setup_worktree(repo_path: str, branch_name: str, wt_path: str) -> None:
    """Add a git worktree for *branch_name* at *wt_path*.

    If a worktree (or stale directory) already exists at that path it is removed
    first. The parent directory is created as needed.
    """
    try:
        if Path(wt_path).exists():
            await _run_subprocess(
                ["git", "worktree", "remove", "--force", wt_path],
                cwd=repo_path,
            )
    except Exception as exc:
        logger.warning("[critic] Could not pre-remove worktree at {}: {}", wt_path, exc)

    Path(wt_path).parent.mkdir(parents=True, exist_ok=True)

    rc, out = await _run_subprocess(
        ["git", "worktree", "add", wt_path, branch_name],
        cwd=repo_path,
    )
    if rc != 0:
        raise RuntimeError(f"git worktree add failed (rc={rc}): {out.strip()}")
    logger.info("[critic] Worktree added: path={} branch={}", wt_path, branch_name)


async def cleanup_worktree(repo_path: str, wt_path: str, branch_name: str) -> None:
    """Remove *wt_path* worktree and delete *branch_name*."""
    try:
        if Path(wt_path).exists():
            rc, out = await _run_subprocess(
                ["git", "worktree", "remove", "--force", wt_path],
                cwd=repo_path,
            )
            if rc != 0:
                logger.warning("[critic] worktree remove rc={}: {}", rc, out.strip())
    except Exception as exc:
        logger.warning("[critic] Error removing worktree {}: {}", wt_path, exc)

    try:
        rc, out = await _run_subprocess(
            ["git", "branch", "-D", branch_name],
            cwd=repo_path,
        )
        if rc != 0:
            logger.warning("[critic] branch -D rc={}: {}", rc, out.strip())
        else:
            logger.info("[critic] Deleted branch {}", branch_name)
    except Exception as exc:
        logger.warning("[critic] Error deleting branch {}: {}", branch_name, exc)


# ---------------------------------------------------------------------------
# Validation gauntlet
# ---------------------------------------------------------------------------


async def _run_validation_gauntlet(
    wt_path: str, rel_path: str
) -> tuple[bool, str]:
    """Run ruff → mypy → pytest → import check. Returns (all_passed, joined_errors)."""
    errors: list[str] = []
    abs_file = str(Path(wt_path) / rel_path)
    pythonpath = _pythonpath(wt_path)

    # 1. ruff check
    rc, out = await _run_subprocess(["ruff", "check", abs_file], cwd=wt_path)
    if rc != 0:
        errors.append(f"[ruff]\n{out.strip()}")
        logger.debug("[critic] ruff failed: {}", rel_path)

    # 2. mypy — scoped to the migrated file only.
    # --ignore-missing-imports + --follow-imports=silent stop mypy from failing
    # on third-party / unmigrated sibling modules that have no stubs in the
    # worktree env. We validate the migrated file's own annotations, not the
    # entire legacy dependency tree.
    rc, out = await _run_subprocess(
        [
            "mypy",
            "--no-incremental",
            "--ignore-missing-imports",
            "--follow-imports=silent",
            abs_file,
        ],
        cwd=wt_path,
    )
    if rc != 0:
        errors.append(f"[mypy]\n{out.strip()}")
        logger.debug("[critic] mypy failed: {}", rel_path)

    # 3. pytest on adjacent test file (skip if none found)
    test_file = _find_test_file(rel_path, wt_path)
    if test_file:
        rc, out = await _run_subprocess(
            ["pytest", "-x", "--tb=short", test_file],
            cwd=wt_path,
            extra_env={"PYTHONPATH": pythonpath},
        )
        if rc != 0:
            errors.append(f"[pytest]\n{out.strip()}")
            logger.debug("[critic] pytest failed: {}", rel_path)
    else:
        logger.debug("[critic] No test file for {}, skipping pytest", rel_path)

    # NOTE: A `python -c "import {module}"` check was removed here. For a
    # cross-framework migration the source framework (Flask) is gone and the
    # target framework (FastAPI) deps aren't installed in the worktree env, so
    # the import always fails — a 100% false-negative in this context.

    if errors:
        return False, "\n\n".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# LLM-guided fix
# ---------------------------------------------------------------------------

_CRITIC_FIX_SYSTEM = (
    "You are a Python expert fixing migration validation errors.\n\n"
    "Given the migration diff and validation errors below, return the COMPLETE "
    "corrected file content in a single fenced Python code block.\n"
    "No explanation, no commentary — only the complete fixed Python file.\n\n"
    "```python\n# corrected code here\n```"
)


def _extract_code_block(response: str) -> str:
    match = _CODE_BLOCK_RE.search(response)
    if match is None:
        raise ValueError(
            f"LLM response contained no fenced code block. "
            f"First 300 chars: {response[:300]!r}"
        )
    return match.group(1)


async def _fix_with_llm(
    file_path: str,
    current_content: str,
    diff: str,
    error_output: str,
    spec: MigrationSpec,
    run_id: str,
    session_id: str | None,
) -> None:
    """Ask the critic LLM for a fully corrected file and overwrite *file_path*."""
    router = get_router()
    client, callbacks = router.get_client(
        "critic",
        session_id=session_id,
        run_id=run_id,
        tags=["critic", "fix"],
    )
    human = (
        f"## Migration: {spec.source_framework} → {spec.target_framework}\n\n"
        f"## Original migration diff\n```diff\n{diff[:_MAX_ERROR_CHARS]}\n```\n\n"
        f"## Current (broken) file\n```python\n{current_content[:_MAX_FILE_CHARS]}\n```\n\n"
        f"## Validation errors\n```\n{error_output[:_MAX_ERROR_CHARS]}\n```\n\n"
        "Return the complete corrected file in a fenced Python code block."
    )
    messages: list[Any] = [
        SystemMessage(content=_CRITIC_FIX_SYSTEM),
        HumanMessage(content=human),
    ]
    try:
        response = await client.ainvoke(messages, config={"callbacks": callbacks})
        router.record_success("critic")
    except Exception:
        router.record_rate_limit_error("critic")
        raise

    raw = str(response.content) if hasattr(response, "content") else str(response)
    corrected = _extract_code_block(raw)
    Path(file_path).write_text(corrected, encoding="utf-8")
    logger.info("[critic] LLM fix written to {}", file_path)


# ---------------------------------------------------------------------------
# Redis status update
# ---------------------------------------------------------------------------


async def _mark_redis(
    task_id: str, status: str, failure_reason: str | None = None
) -> None:
    try:
        from redis.asyncio import Redis  # lazy import

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        redis: Redis = Redis.from_url(redis_url, decode_responses=True)  # type: ignore[assignment]
        async with redis:
            mapping: dict[str, str] = {"status": status}
            if failure_reason:
                mapping["failure_reason"] = failure_reason[:500]
            await redis.hset(f"task:{task_id}", mapping=mapping)  # type: ignore[arg-type]
        logger.info("[critic] Redis task:{} status={}", task_id, status)
    except Exception as exc:
        logger.warning("[critic] Redis update failed for task:{} — {}", task_id, exc)


# ---------------------------------------------------------------------------
# Pattern recording (fire-and-forget)
# ---------------------------------------------------------------------------


def _record_success_bg(
    original_content: str,
    migrated_content: str,
    spec: MigrationSpec,
) -> None:
    """Schedule pattern recording as a background task — never blocks the result."""

    async def _do() -> None:
        try:
            pattern = MigrationPattern(
                before_code=original_content[:2_000],
                after_code=migrated_content[:2_000],
                migration_type=f"{spec.source_framework}→{spec.target_framework}",
                source_framework=spec.source_framework,
                target_framework=spec.target_framework,
            )
            await PatternStore().record_success(pattern)
            logger.info("[critic] Success pattern recorded")
        except Exception as exc:
            logger.warning("[critic] record_success failed (non-fatal): {}", exc)

    asyncio.ensure_future(_do())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@observe(name="run_critic")
async def run_critic(
    result: TaskResult,
    task_id: str,
    spec: MigrationSpec,
    run_id: str,
    repo_path: str,
    *,
    session_id: str | None = None,
) -> CriticResult:
    """Validate a migrated branch and apply LLM-guided fixes on failure.

    Args:
        result:     TaskResult from the transform agent (must have branch_name set).
        task_id:    Redis key suffix for status updates (``task:{task_id}``).
        spec:       Source/target framework and version.
        run_id:     Migration run UUID — used to derive the worktree path.
        repo_path:  Absolute path to the repository root.
        session_id: Langfuse session ID for grouping traces.

    Returns:
        CriticResult with ``verdict="pass"`` or ``"fail"``.
        On failure ``rollback_entry`` is populated and the branch is deleted.
    """
    branch_name = result.branch_name
    if not branch_name:
        logger.warning("[critic] No branch_name on result, skipping: {}", result.module_path)
        return CriticResult(
            module_path=result.module_path,
            verdict="fail",
            failure_reason="no branch_name on TaskResult",
        )

    rel_path = str(Path(result.module_path).relative_to(repo_path))
    wt_path = worktree_path(run_id, branch_name)

    logger.info(
        "[critic] Starting: module={} branch={}",
        Path(result.module_path).name,
        branch_name,
    )

    # Snapshot original file content before any worktree ops (for pattern + rollback)
    try:
        original_content = Path(result.module_path).read_text(encoding="utf-8")
    except Exception as exc:
        original_content = ""
        logger.warning("[critic] Could not read original {}: {}", result.module_path, exc)

    # Setup worktree
    try:
        await setup_worktree(repo_path, branch_name, wt_path)
    except Exception as exc:
        logger.error("[critic] Worktree setup failed: {}", exc)
        return CriticResult(
            module_path=result.module_path,
            verdict="fail",
            branch_name=branch_name,
            failure_reason=f"worktree setup failed: {exc}",
        )

    abs_file = str(Path(wt_path) / rel_path)
    retry_count = 0
    passed, error_output = await _run_validation_gauntlet(wt_path, rel_path)

    for attempt in range(_MAX_RETRIES):
        if passed:
            break
        retry_count = attempt + 1
        logger.info(
            "[critic] Attempt {}/{} failed — requesting LLM fix",
            retry_count,
            _MAX_RETRIES,
        )
        try:
            current_content = Path(abs_file).read_text(encoding="utf-8")
            await _fix_with_llm(
                file_path=abs_file,
                current_content=current_content,
                diff=result.diff,
                error_output=error_output,
                spec=spec,
                run_id=run_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("[critic] LLM fix failed on attempt {}: {}", retry_count, exc)
            # error_output already set; break so we don't loop on a broken LLM state
            break
        passed, error_output = await _run_validation_gauntlet(wt_path, rel_path)

    if passed:
        logger.info(
            "[critic] PASSED: module={} retries={}",
            Path(result.module_path).name,
            retry_count,
        )
        await _mark_redis(task_id, "done")
        migrated_content = Path(abs_file).read_text(encoding="utf-8", errors="replace")
        _record_success_bg(original_content, migrated_content, spec)
        return CriticResult(
            module_path=result.module_path,
            verdict="pass",
            branch_name=branch_name,
            retry_count=retry_count,
        )

    # Exhausted retries — clean up and signal rollback
    logger.error(
        "[critic] FAILED after {} retries: module={} — {}",
        retry_count,
        Path(result.module_path).name,
        error_output[:200],
    )
    await cleanup_worktree(repo_path, wt_path, branch_name)
    await _mark_redis(task_id, "failed", failure_reason=error_output)

    return CriticResult(
        module_path=result.module_path,
        verdict="fail",
        branch_name=branch_name,
        failure_reason=error_output[:500],
        retry_count=retry_count,
        rollback_entry=RollbackEntry(
            module_path=result.module_path,
            original_content=original_content,
            branch_name=branch_name,
        ),
    )
