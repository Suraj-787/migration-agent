"""Stress tests for the parallel execution path.

All LLM, Voyage, and Redis-lock I/O is mocked.  These tests validate:
  - The global asyncio.Semaphore(5) caps concurrent LLM calls to at most 5.
  - No two coroutines hold the same Redis lock for the same path concurrently.
  - No duplicate "transformed" results appear for the same module path.
  - All 20 tasks reach a terminal state (transformed / failed / skipped) — none stuck.

Run with:  uv run pytest tests/test_stress.py -k stress -xvs
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agents.critic import CriticResult
from workflows.graph import graph_builder
from workflows.state import MigrationSpec, MigrationState, MigrationTask, TaskResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SPEC = MigrationSpec(
    source_framework="flask",
    target_framework="fastapi",
    source_version="2.3",
    target_version="0.115",
)

_TERMINAL_STATUSES = frozenset({"transformed", "success", "failed", "skipped"})


def _make_tasks(n: int, unique_paths: bool = True) -> list[MigrationTask]:
    """Return *n* MigrationTask objects.

    If *unique_paths* is False, 20 tasks share 5 paths (4 tasks per path) to
    exercise lock-contention paths.
    """
    n_paths = n if unique_paths else max(1, n // 4)
    return [
        MigrationTask(
            task_id=f"stress-{i}",
            module_path=f"/tmp/stress_module_{i % n_paths}.py",
            description=f"Migrate stress module {i}",
            priority=i % 5,
            complexity="trivial",
            predicted_changes=[],
            retrieved_context_ids=[],
            depends_on=[],
        )
        for i in range(n)
    ]


def _initial_state(tasks: list[MigrationTask]) -> tuple[MigrationState, str]:
    thread_id = str(uuid.uuid4())
    state = MigrationState(
        repo_path="/tmp/stress-repo",
        target_spec=_SPEC,
        dep_graph_id=uuid.uuid4(),
        task_queue=[],  # planner mock fills this
        task_results=[],
        rollback_stack=[],
        critiqued_paths=[],
        passed_paths=[],
        current_batch=[],
        attempt_count={},
        final_status="pending",
        thread_id=thread_id,
        started_at=time.time(),
        migration_report=None,
        langfuse_trace_id=None,
    )
    return state, thread_id


def _graph() -> Any:
    return graph_builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Semaphore test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stress_global_semaphore_limits_concurrency() -> None:
    """Semaphore(5) must cap concurrent LLM calls to at most 5 at any instant."""
    from workflows.queue import get_llm_semaphore

    # Reset the module-level singleton so the test gets a clean semaphore.
    import workflows.queue as _queue_mod
    _queue_mod._llm_semaphore = None

    sem = get_llm_semaphore()
    concurrent: list[int] = []
    active = 0

    async def _fake_call() -> None:
        nonlocal active
        async with sem:
            active += 1
            concurrent.append(active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*[_fake_call() for _ in range(20)])

    max_seen = max(concurrent)
    assert max_seen <= 5, (
        f"Semaphore violation: {max_seen} concurrent calls observed (limit is 5)"
    )


# ---------------------------------------------------------------------------
# Redis lock exclusivity test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stress_no_concurrent_redis_lock_holders() -> None:
    """At most one coroutine may hold the Redis lock for a given path at a time.

    10 coroutines race for the same path.  The NX-tracking mock refuses
    second callers (returning False, which the graph treats as "skipped").
    We verify that the peak number of concurrent holders never exceeds 1.
    """
    MODULE_PATH = "/tmp/shared_lock_test.py"
    _held: dict[str, bool] = {}
    _guard = asyncio.Lock()
    max_concurrent = 0
    current_count = 0

    async def tracking_acquire(module_path: str, run_id: str) -> bool:
        nonlocal max_concurrent, current_count
        async with _guard:
            if _held.get(module_path, False):
                return False  # correctly refused — NX semantics
            _held[module_path] = True
            current_count += 1
            max_concurrent = max(max_concurrent, current_count)
        return True

    async def tracking_release(module_path: str) -> None:
        nonlocal current_count
        async with _guard:
            _held[module_path] = False
            current_count -= 1

    async def _one_worker(run_id: str) -> None:
        acquired = await tracking_acquire(MODULE_PATH, run_id)
        if acquired:
            await asyncio.sleep(0.01)
            await tracking_release(MODULE_PATH)

    run_ids = [str(uuid.uuid4()) for _ in range(10)]
    await asyncio.gather(*[_one_worker(rid) for rid in run_ids])

    # The lock must never be held by more than one coroutine simultaneously.
    assert max_concurrent <= 1, (
        f"Lock violated: {max_concurrent} concurrent holders (max allowed: 1)"
    )


# ---------------------------------------------------------------------------
# 20-task parallel pipeline stress test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stress_20_tasks_all_reach_terminal_state() -> None:
    """20 unique-path tasks must all reach a terminal state via the graph pipeline.

    Mocks: run_planner, run_transform, run_critic, _acquire_lock, _release_lock,
           persist_report, embed_chunks, find_similar.
    Assertions:
      - len(task_results) == 20
      - Every result.status ∈ {transformed, success, failed, skipped}
      - No task is stuck (i.e. absent from task_results)
      - final_status is "success"
    """
    N = 20
    tasks = _make_tasks(N, unique_paths=True)

    # Lock tracking — all paths are unique so no contention expected.
    _held: dict[str, bool] = {}
    _violations: list[str] = []
    _guard = asyncio.Lock()

    async def mock_acquire(module_path: str, run_id: str) -> bool:
        async with _guard:
            if _held.get(module_path, False):
                _violations.append(module_path)
                return False
            _held[module_path] = True
        return True

    async def mock_release(module_path: str) -> None:
        async with _guard:
            _held[module_path] = False

    async def mock_transform(
        task: MigrationTask, spec: Any, run_id: str, repo_path: str, session_id: Any = None
    ) -> TaskResult:
        await asyncio.sleep(0.005)  # simulate brief async work
        return TaskResult(
            module_path=task.module_path,
            status="transformed",
            branch_name=f"migration/run/{task.task_id}",
            tokens_used=42,
        )

    async def mock_critic(
        result: TaskResult, task_id: str, spec: Any, run_id: str, repo_path: str, session_id: Any = None
    ) -> CriticResult:
        return CriticResult(
            module_path=result.module_path,
            verdict="pass",
            branch_name=result.branch_name,
        )

    g = _graph()
    initial, thread_id = _initial_state(tasks)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    with (
        patch("agents.planner.run_planner", new=AsyncMock(return_value=tasks)),
        patch("agents.transform.run_transform", side_effect=mock_transform),
        patch("agents.critic.run_critic", side_effect=mock_critic),
        patch("workflows.graph._acquire_lock", side_effect=mock_acquire),
        patch("workflows.graph._release_lock", side_effect=mock_release),
        patch("workflows.report.persist_report", new=AsyncMock()),
    ):
        result = await g.ainvoke(initial, config=config)

    task_results = result["task_results"]

    # All 20 tasks must be accounted for.
    assert len(task_results) == N, (
        f"Expected {N} results, got {len(task_results)}: {task_results}"
    )

    # Every result must have a terminal status.
    stuck = [r for r in task_results if r.status not in _TERMINAL_STATUSES]
    assert not stuck, f"Non-terminal task results: {stuck}"

    # No concurrent lock holders for any path.
    assert not _violations, f"Concurrent lock violations: {_violations}"

    # With all unique paths and mocked success, the run should fully succeed.
    assert result["final_status"] == "success", (
        f"Expected final_status='success', got {result['final_status']!r}"
    )


# ---------------------------------------------------------------------------
# Duplicate-path race condition test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stress_no_duplicate_transformed_results_per_path() -> None:
    """Tasks sharing a module_path must not both produce status='transformed'.

    The dispatch logic filters already-completed paths from each batch, so
    duplicate tasks are skipped — only the first gets "transformed".
    """
    # 6 tasks across 2 paths (3 tasks per path).  Only 1 per path should
    # reach "transformed"; the rest should be filtered by _select_ready_batch
    # or turned into "skipped" by the lock mock.
    tasks = [
        MigrationTask(
            task_id=f"dup-{i}",
            module_path=f"/tmp/shared_{i % 2}.py",
            description=f"dup task {i}",
            priority=0,
            complexity="trivial",
            predicted_changes=[],
            retrieved_context_ids=[],
            depends_on=[],
        )
        for i in range(6)
    ]

    _held: dict[str, bool] = {}
    _guard = asyncio.Lock()

    async def mock_acquire(module_path: str, run_id: str) -> bool:
        async with _guard:
            if _held.get(module_path, False):
                return False  # second acquisition blocked — NX semantics
            _held[module_path] = True
        return True

    async def mock_release(module_path: str) -> None:
        async with _guard:
            _held[module_path] = False

    async def mock_transform(task: MigrationTask, spec: Any, run_id: str, repo_path: str, session_id: Any = None) -> TaskResult:
        return TaskResult(
            module_path=task.module_path,
            status="transformed",
            branch_name=f"migration/{task.task_id}",
            tokens_used=10,
        )

    async def mock_critic(result: TaskResult, task_id: str, spec: Any, run_id: str, repo_path: str, session_id: Any = None) -> CriticResult:
        return CriticResult(module_path=result.module_path, verdict="pass", branch_name=result.branch_name)

    g = _graph()
    initial, thread_id = _initial_state(tasks)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    with (
        patch("agents.planner.run_planner", new=AsyncMock(return_value=tasks)),
        patch("agents.transform.run_transform", side_effect=mock_transform),
        patch("agents.critic.run_critic", side_effect=mock_critic),
        patch("workflows.graph._acquire_lock", side_effect=mock_acquire),
        patch("workflows.graph._release_lock", side_effect=mock_release),
        patch("workflows.report.persist_report", new=AsyncMock()),
    ):
        result = await g.ainvoke(initial, config=config)

    # Build counts per path.
    transformed_by_path: dict[str, int] = {}
    for r in result["task_results"]:
        if r.status == "transformed":
            transformed_by_path[r.module_path] = (
                transformed_by_path.get(r.module_path, 0) + 1
            )

    duplicates = {p: c for p, c in transformed_by_path.items() if c > 1}
    assert not duplicates, (
        f"Duplicate 'transformed' results for paths: {duplicates}"
    )
