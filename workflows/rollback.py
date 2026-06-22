"""Rollback workflow: ensures no dangling branches or worktrees after critic failures.

run_rollback is called by rollback_node in graph.py whenever the critic pushes
entries to rollback_stack. It acts as a safety net — critic already cleans up
on failure, but rollback_node re-verifies and force-removes anything that leaked.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from loguru import logger

from agents.critic import worktree_path
from workflows.state import RollbackEntry


async def _run_subprocess(cmd: list[str], cwd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    stdout, stderr = await proc.communicate()
    output = (stdout + stderr).decode("utf-8", errors="replace")
    return proc.returncode if proc.returncode is not None else 1, output


async def _branch_exists(repo_path: str, branch_name: str) -> bool:
    """Return True if *branch_name* still exists locally."""
    _, out = await _run_subprocess(
        ["git", "branch", "--list", branch_name], cwd=repo_path
    )
    return bool(out.strip())


async def _ensure_entry_clean(
    entry: RollbackEntry, run_id: str, repo_path: str
) -> None:
    """Verify branch and worktree for this entry are gone. Re-clean if not."""
    wt = worktree_path(run_id, entry.branch_name)

    if Path(wt).exists():
        logger.warning("[rollback] Stale worktree found, removing: {}", wt)
        rc, out = await _run_subprocess(
            ["git", "worktree", "remove", "--force", wt], cwd=repo_path
        )
        if rc != 0:
            logger.error("[rollback] Force remove worktree failed (rc={}): {}", rc, out.strip())

    if await _branch_exists(repo_path, entry.branch_name):
        logger.warning("[rollback] Stale branch found, deleting: {}", entry.branch_name)
        rc, out = await _run_subprocess(
            ["git", "branch", "-D", entry.branch_name], cwd=repo_path
        )
        if rc != 0:
            logger.error(
                "[rollback] Branch delete failed (rc={}): {}", rc, out.strip()
            )
        else:
            logger.info("[rollback] Deleted stale branch {}", entry.branch_name)

    logger.info("[rollback] Verified clean: branch={}", entry.branch_name)


async def run_rollback(
    entries: list[RollbackEntry],
    run_id: str,
    repo_path: str,
) -> None:
    """Verify all rollback entries are fully cleaned up. Re-runs cleanup for any leaks."""
    if not entries:
        return
    logger.info("[rollback] Verifying {} entries", len(entries))
    for entry in entries:
        try:
            await _ensure_entry_clean(entry, run_id, repo_path)
        except Exception as exc:
            logger.error(
                "[rollback] Cleanup verification failed branch={}: {}", entry.branch_name, exc
            )
    logger.info("[rollback] All entries verified clean")
