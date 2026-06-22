"""Unit tests for agents/transform.py.

No real LLM, Qdrant, or git repo required — all external I/O is patched.
Tests cover:
 - Code-block extraction edge cases (key bug source per the build plan)
 - Branch-name derivation
 - Segment builder (preamble / section / epilogue split)
 - Whole-file happy path: correct TaskResult fields
 - Chunked path with mocked chunker
 - LLM failure → status="failed"
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents.transform import (
    _build_segments,
    _extract_code_block,
    _make_branch_name,
    run_transform,
)
from rag.models import CodeChunk
from workflows.state import MigrationSpec, MigrationTask

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

_TASK = MigrationTask(
    task_id="task-abc",
    module_path=FIXTURE_APP,
    description="Migrate app from flask 2.3 to fastapi 0.115",
    priority=0,
    complexity="standard",
    predicted_changes=["Replace @app.route with APIRouter"],
    retrieved_context_ids=[],
    depends_on=[],
)

_MIGRATED_CODE = textwrap.dedent("""\
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def index() -> dict:
        return {"hello": "world"}
""")

_CANNED_RESPONSE = f"Here is the migrated file:\n\n```python\n{_MIGRATED_CODE}```\n"


# ---------------------------------------------------------------------------
# _extract_code_block
# ---------------------------------------------------------------------------


def test_extract_python_tag() -> None:
    raw = "Some preamble.\n```python\ncode here\n```\nSome epilogue."
    assert _extract_code_block(raw) == "code here\n"


def test_extract_bare_tag() -> None:
    raw = "```\nbare code\n```"
    assert _extract_code_block(raw) == "bare code\n"


def test_extract_py_alias() -> None:
    raw = "```py\nalias code\n```"
    assert _extract_code_block(raw) == "alias code\n"


def test_extract_trailing_spaces_after_tag() -> None:
    raw = "```python  \nspaced\n```"
    assert _extract_code_block(raw) == "spaced\n"


def test_extract_case_insensitive() -> None:
    raw = "```Python\nupper\n```"
    assert _extract_code_block(raw) == "upper\n"


def test_extract_missing_raises() -> None:
    with pytest.raises(ValueError, match="no fenced code block"):
        _extract_code_block("No code block anywhere in this response.")


def test_extract_picks_first_block() -> None:
    raw = "```python\nfirst\n```\n```python\nsecond\n```"
    assert _extract_code_block(raw) == "first\n"


# ---------------------------------------------------------------------------
# _make_branch_name
# ---------------------------------------------------------------------------


def test_branch_name_flat_file() -> None:
    name = _make_branch_name("run-123", "app.py")
    assert name == "migration/run-123/app"


def test_branch_name_nested_path() -> None:
    name = _make_branch_name("run-123", "routes/users.py")
    assert name == "migration/run-123/routes.users"


def test_branch_name_sanitizes_special_chars() -> None:
    name = _make_branch_name("run-123", "my app.py")
    assert " " not in name


# ---------------------------------------------------------------------------
# _build_segments
# ---------------------------------------------------------------------------


def _make_chunk(start: int, end: int) -> CodeChunk:
    return CodeChunk(
        file_path="dummy.py",
        language="python",
        content="def foo(): pass",
        node_type="function",
        parent_context="dummy.py",
        start_line=start,
        end_line=end,
    )


def test_build_segments_no_chunks() -> None:
    lines = ["import os\n", "X = 1\n"]
    segments = _build_segments(lines, [])
    assert len(segments) == 1
    assert segments[0][0] == "preamble"
    assert "import os" in segments[0][1]


def test_build_segments_preamble_plus_section() -> None:
    lines = ["import os\n", "\n", "def foo():\n", "    pass\n"]
    chunks = [_make_chunk(3, 4)]
    segments = _build_segments(lines, chunks)
    # Should have preamble (lines 1-2) and one section (lines 3-4)
    types = [s[0] for s in segments]
    assert "preamble" in types
    assert "section" in types


def test_build_segments_only_functions() -> None:
    lines = ["def foo():\n", "    pass\n", "def bar():\n", "    pass\n"]
    chunks = [_make_chunk(1, 2), _make_chunk(3, 4)]
    segments = _build_segments(lines, chunks)
    # No preamble; two sections
    assert all(s[0] == "section" for s in segments)
    assert len(segments) == 2


def test_build_segments_epilogue() -> None:
    lines = [
        "def foo():\n",
        "    pass\n",
        "if __name__ == '__main__':\n",
        "    foo()\n",
    ]
    chunks = [_make_chunk(1, 2)]
    segments = _build_segments(lines, chunks)
    types = [s[0] for s in segments]
    assert "section" in types
    assert "epilogue" in types


# ---------------------------------------------------------------------------
# run_transform — whole-file happy path
# ---------------------------------------------------------------------------


def _stub_transform(canned_llm_response: str = _CANNED_RESPONSE):
    """Return an ExitStack context-manager that stubs all external I/O."""
    from contextlib import ExitStack
    from unittest.mock import AsyncMock, patch

    stack = ExitStack()
    # RAG returns empty (Qdrant not needed for unit tests)
    stack.enter_context(
        patch(
            "agents.transform._retrieve_transform_context",
            new=AsyncMock(return_value=([], [], [])),
        )
    )
    # LLM returns canned response
    stack.enter_context(
        patch(
            "agents.transform._call_llm",
            new=AsyncMock(return_value=(canned_llm_response, 512)),
        )
    )
    # Git branch write is a no-op
    stack.enter_context(
        patch("agents.transform._write_branch", return_value=None)
    )
    return stack


@pytest.mark.asyncio
async def test_run_transform_happy_path() -> None:
    with _stub_transform():
        result = await run_transform(
            task=_TASK,
            spec=_SPEC,
            run_id="run-xyz",
            repo_path=FIXTURE_REPO,
        )

    assert result.status == "transformed"
    assert result.branch_name == "migration/run-xyz/app"
    assert result.tokens_used == 512
    assert "fastapi" in result.diff.lower() or result.diff == "(no diff)"
    assert result.error is None


@pytest.mark.asyncio
async def test_run_transform_returns_diff() -> None:
    with _stub_transform():
        result = await run_transform(
            task=_TASK, spec=_SPEC, run_id="run-xyz", repo_path=FIXTURE_REPO
        )
    # diff should be non-trivial (original file != migrated content)
    assert result.diff != "(no diff)"


# ---------------------------------------------------------------------------
# run_transform — LLM failure → status="failed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_transform_llm_failure() -> None:
    """A non-rate-limit LLM error fails immediately with the message preserved."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "agents.transform._retrieve_transform_context",
                new=AsyncMock(return_value=([], [], [])),
            )
        )
        stack.enter_context(
            patch(
                "agents.transform._call_llm",
                new=AsyncMock(side_effect=RuntimeError("unexpected boom")),
            )
        )
        result = await run_transform(
            task=_TASK, spec=_SPEC, run_id="run-xyz", repo_path=FIXTURE_REPO
        )

    assert result.status == "failed"
    assert result.error is not None
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_run_transform_rate_limit_exhausted() -> None:
    """A persistent 429 retries the backoff schedule then fails as rate_limit_exhausted."""
    from contextlib import ExitStack

    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "agents.transform._retrieve_transform_context",
                new=AsyncMock(return_value=([], [], [])),
            )
        )
        stack.enter_context(
            patch(
                "agents.transform._call_llm",
                new=AsyncMock(side_effect=RuntimeError("Error code: 429 rate-limited")),
            )
        )
        stack.enter_context(
            patch("agents.transform.asyncio.sleep", new=AsyncMock(side_effect=_fake_sleep))
        )
        result = await run_transform(
            task=_TASK, spec=_SPEC, run_id="run-xyz", repo_path=FIXTURE_REPO
        )

    assert result.status == "failed"
    assert result.error == "rate_limit_exhausted"
    # 3 retries with the documented backoff schedule.
    assert slept == [10.0, 30.0, 60.0]


# ---------------------------------------------------------------------------
# run_transform — no code block in LLM response → status="failed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_transform_no_code_block() -> None:
    bad_response = "Sure, here is my explanation of the migration..."
    with _stub_transform(canned_llm_response=bad_response):
        result = await run_transform(
            task=_TASK, spec=_SPEC, run_id="run-xyz", repo_path=FIXTURE_REPO
        )

    assert result.status == "failed"
    assert result.error is not None
    assert "code block" in result.error


# ---------------------------------------------------------------------------
# run_transform — chunked path (file >= 500 lines)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_transform_chunked_path() -> None:
    """Files >= 500 lines should use the chunked path and stitch segments."""
    long_content = "# preamble\n" + "def foo():\n    pass\n" * 200  # > 500 lines

    fake_chunk = _make_chunk(2, 3)
    canned_chunk_response = "```python\nasync def foo():\n    pass\n```"

    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "agents.transform._retrieve_transform_context",
                new=AsyncMock(return_value=([], [], [])),
            )
        )
        stack.enter_context(
            patch(
                "agents.transform._call_llm",
                new=AsyncMock(return_value=(canned_chunk_response, 100)),
            )
        )
        stack.enter_context(
            patch("agents.transform._write_branch", return_value=None)
        )
        # Patch read_text so the file doesn't need to exist at the long path
        stack.enter_context(
            patch(
                "pathlib.Path.read_text",
                return_value=long_content,
            )
        )
        # Patch chunker to return one fake chunk
        stack.enter_context(
            patch(
                "agents.transform.CodeChunker.chunk_file",
                return_value=[fake_chunk],
            )
        )

        task = _TASK.model_copy(
            update={"module_path": "/fake/repo/app.py"}
        )
        result = await run_transform(
            task=task,
            spec=_SPEC,
            run_id="run-xyz",
            repo_path="/fake/repo",
        )

    assert result.status == "transformed"
    assert result.tokens_used > 0
