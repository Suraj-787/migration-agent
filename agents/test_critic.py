"""Unit tests for agents/critic.py.

No real git repo, LLM, Redis, or Qdrant required — all external I/O is patched.
Tests cover:
 - worktree_path: deterministic, safe path derivation
 - _find_test_file: adjacent test file discovery
 - run_critic happy path: all validations pass → verdict="pass", Redis marked done
 - run_critic retry path: first gauntlet fails, LLM fix applied, second gauntlet passes
 - run_critic exhausted: 2 retries fail → verdict="fail", rollback_entry set, branch deleted
 - fire-and-forget: record_success exception must not propagate
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents.critic import (
    CriticResult,
    _extract_code_block,
    _find_test_file,
    run_critic,
    worktree_path,
)
from workflows.state import MigrationSpec, TaskResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_REPO = str(
    Path(__file__).parent.parent / "tests" / "fixtures" / "sample_flask_app"
)
FIXTURE_APP = str(Path(FIXTURE_REPO) / "app.py")

_SPEC = MigrationSpec(
    source_framework="flask",
    target_framework="fastapi",
    source_version="2.3",
    target_version="0.115",
)

_TASK_RESULT = TaskResult(
    module_path=FIXTURE_APP,
    status="transformed",
    diff="- from flask import Flask\n+ from fastapi import FastAPI",
    branch_name="migration/run-abc/app",
    tokens_used=100,
)

_MIGRATED_CODE = textwrap.dedent("""\
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def index() -> dict:
        return {"hello": "world"}
""")

_CANNED_FIX_RESPONSE = f"Fixed file:\n\n```python\n{_MIGRATED_CODE}```\n"


# ---------------------------------------------------------------------------
# Pure-function tests (no I/O)
# ---------------------------------------------------------------------------


def test_worktree_path_no_slashes() -> None:
    path = worktree_path("run-123", "migration/run-123/app.models")
    assert "/tmp/migration_worktrees/run-123/" in path
    assert "/" not in path.split("/tmp/migration_worktrees/run-123/")[1]


def test_worktree_path_deterministic() -> None:
    assert worktree_path("r1", "br/x") == worktree_path("r1", "br/x")


def test_find_test_file_present(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("pass")
    (tmp_path / "test_app.py").write_text("pass")
    result = _find_test_file("app.py", str(tmp_path))
    assert result is not None
    assert result.endswith("test_app.py")


def test_find_test_file_absent(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("pass")
    assert _find_test_file("app.py", str(tmp_path)) is None


def test_extract_code_block_python_tag() -> None:
    raw = "Preamble\n```python\ncode here\n```\nEpilogue"
    assert _extract_code_block(raw) == "code here\n"


def test_extract_code_block_bare_tag() -> None:
    raw = "```\nbare\n```"
    assert _extract_code_block(raw) == "bare\n"


def test_extract_code_block_missing_raises() -> None:
    with pytest.raises(ValueError, match="no fenced code block"):
        _extract_code_block("no block here at all")


# ---------------------------------------------------------------------------
# Helpers — patch stacks
# ---------------------------------------------------------------------------


def _make_gauntlet_pass() -> AsyncMock:
    return AsyncMock(return_value=(True, ""))


def _make_gauntlet_fail(msg: str = "ruff: E501") -> AsyncMock:
    return AsyncMock(return_value=(False, msg))


def _make_setup_worktree() -> AsyncMock:
    return AsyncMock(return_value=None)


def _make_cleanup_worktree() -> AsyncMock:
    return AsyncMock(return_value=None)


def _make_mark_redis() -> AsyncMock:
    return AsyncMock(return_value=None)


def _make_llm_fix() -> AsyncMock:
    return AsyncMock(return_value=None)


# ---------------------------------------------------------------------------
# run_critic — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_critic_all_pass() -> None:
    with (
        patch("agents.critic.setup_worktree", _make_setup_worktree()),
        patch("agents.critic._run_validation_gauntlet", _make_gauntlet_pass()),
        patch("agents.critic._mark_redis", _make_mark_redis()) as mock_redis,
        patch("agents.critic._record_success_bg") as mock_record,
        patch("pathlib.Path.read_text", return_value=_MIGRATED_CODE),
    ):
        result: CriticResult = await run_critic(
            result=_TASK_RESULT,
            task_id="task-001",
            spec=_SPEC,
            run_id="run-abc",
            repo_path=FIXTURE_REPO,
        )

    assert result.verdict == "pass"
    assert result.branch_name == "migration/run-abc/app"
    assert result.retry_count == 0
    assert result.rollback_entry is None
    mock_redis.assert_awaited_once_with("task-001", "done")
    mock_record.assert_called_once()


# ---------------------------------------------------------------------------
# run_critic — no branch_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_critic_no_branch() -> None:
    no_branch = TaskResult(
        module_path=FIXTURE_APP,
        status="failed",
        branch_name=None,
    )
    result = await run_critic(
        result=no_branch,
        task_id="task-002",
        spec=_SPEC,
        run_id="run-abc",
        repo_path=FIXTURE_REPO,
    )
    assert result.verdict == "fail"
    assert "no branch_name" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# run_critic — first gauntlet fails, LLM fix works on retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_critic_retry_success() -> None:
    gauntlet = AsyncMock(side_effect=[(False, "mypy error"), (True, "")])

    with (
        patch("agents.critic.setup_worktree", _make_setup_worktree()),
        patch("agents.critic._run_validation_gauntlet", gauntlet),
        patch("agents.critic._fix_with_llm", _make_llm_fix()) as mock_fix,
        patch("agents.critic._mark_redis", _make_mark_redis()) as mock_redis,
        patch("agents.critic._record_success_bg"),
        patch("pathlib.Path.read_text", return_value=_MIGRATED_CODE),
    ):
        result = await run_critic(
            result=_TASK_RESULT,
            task_id="task-003",
            spec=_SPEC,
            run_id="run-abc",
            repo_path=FIXTURE_REPO,
        )

    assert result.verdict == "pass"
    assert result.retry_count == 1
    mock_fix.assert_awaited_once()
    mock_redis.assert_awaited_once_with("task-003", "done")


# ---------------------------------------------------------------------------
# run_critic — both retries exhausted → rollback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_critic_exhausted_retries() -> None:
    gauntlet = AsyncMock(
        side_effect=[(False, "err0"), (False, "err1"), (False, "err2")]
    )
    mock_cleanup = _make_cleanup_worktree()

    with (
        patch("agents.critic.setup_worktree", _make_setup_worktree()),
        patch("agents.critic._run_validation_gauntlet", gauntlet),
        patch("agents.critic._fix_with_llm", _make_llm_fix()),
        patch("agents.critic.cleanup_worktree", mock_cleanup),
        patch("agents.critic._mark_redis", _make_mark_redis()) as mock_redis,
        patch("pathlib.Path.read_text", return_value="original code"),
    ):
        result = await run_critic(
            result=_TASK_RESULT,
            task_id="task-004",
            spec=_SPEC,
            run_id="run-abc",
            repo_path=FIXTURE_REPO,
        )

    assert result.verdict == "fail"
    assert result.retry_count == 2
    assert result.rollback_entry is not None
    assert result.rollback_entry.branch_name == "migration/run-abc/app"
    assert result.rollback_entry.original_content == "original code"
    mock_cleanup.assert_awaited_once()
    mock_redis.assert_awaited_once_with("task-004", "failed", failure_reason="err2")


# ---------------------------------------------------------------------------
# run_critic — LLM fix itself raises → still counts as failed attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_critic_llm_fix_raises() -> None:
    gauntlet = AsyncMock(side_effect=[(False, "ruff error"), (False, "still broken")])
    broken_fix = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    mock_cleanup = _make_cleanup_worktree()

    with (
        patch("agents.critic.setup_worktree", _make_setup_worktree()),
        patch("agents.critic._run_validation_gauntlet", gauntlet),
        patch("agents.critic._fix_with_llm", broken_fix),
        patch("agents.critic.cleanup_worktree", mock_cleanup),
        patch("agents.critic._mark_redis", _make_mark_redis()),
        patch("pathlib.Path.read_text", return_value="code"),
    ):
        result = await run_critic(
            result=_TASK_RESULT,
            task_id="task-005",
            spec=_SPEC,
            run_id="run-abc",
            repo_path=FIXTURE_REPO,
        )

    # LLM raised → loop broke early; result is still fail with rollback
    assert result.verdict == "fail"
    assert result.rollback_entry is not None


# ---------------------------------------------------------------------------
# run_critic — worktree setup failure → fail without rollback entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_critic_worktree_setup_fails() -> None:
    with patch(
        "agents.critic.setup_worktree",
        AsyncMock(side_effect=RuntimeError("git error")),
    ):
        result = await run_critic(
            result=_TASK_RESULT,
            task_id="task-006",
            spec=_SPEC,
            run_id="run-abc",
            repo_path=FIXTURE_REPO,
        )

    assert result.verdict == "fail"
    assert "worktree setup failed" in (result.failure_reason or "")
    assert result.rollback_entry is None


# ---------------------------------------------------------------------------
# record_success exception must not propagate (fire-and-forget contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_success_exception_is_swallowed() -> None:
    """_record_success_bg must not raise even when PatternStore throws."""
    gauntlet = AsyncMock(return_value=(True, ""))

    async def _exploding_record(*_: object, **__: object) -> None:
        raise RuntimeError("Qdrant down")

    with (
        patch("agents.critic.setup_worktree", _make_setup_worktree()),
        patch("agents.critic._run_validation_gauntlet", gauntlet),
        patch("agents.critic._mark_redis", _make_mark_redis()),
        patch("agents.critic.PatternStore") as mock_store_cls,
        patch("pathlib.Path.read_text", return_value="code"),
    ):
        mock_store_cls.return_value.record_success = AsyncMock(
            side_effect=RuntimeError("Qdrant down")
        )
        # Should complete without raising
        result = await run_critic(
            result=_TASK_RESULT,
            task_id="task-007",
            spec=_SPEC,
            run_id="run-abc",
            repo_path=FIXTURE_REPO,
        )

    assert result.verdict == "pass"
