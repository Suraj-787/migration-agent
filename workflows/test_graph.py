"""Unit tests for the migration orchestration graph.

Uses MemorySaver (no Postgres required) to verify:
 - graph compiles without error
 - a stub run with empty task_queue visits plan/dispatch/test/critic/finalize
 - final_status is "success" after the run
 - with tasks in the queue, transform_task is visited via the Send fan-out
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agents.critic import CriticResult
from workflows.graph import graph_builder
from workflows.state import MigrationSpec, MigrationState, MigrationTask, TaskResult

FIXTURE_REPO = str(
    Path(__file__).parent.parent / "tests" / "fixtures" / "sample_flask_app"
)


@pytest.fixture()
def compiled_graph():  # type: ignore[no-untyped-def]
    return graph_builder.compile(checkpointer=MemorySaver())


def _initial_state() -> MigrationState:
    return MigrationState(
        repo_path="/tmp/test-repo",
        target_spec=MigrationSpec(
            source_framework="flask",
            target_framework="fastapi",
            source_version="2.3",
            target_version="0.115",
        ),
        dep_graph_id=uuid.uuid4(),
        task_queue=[],
        task_results=[],
        rollback_stack=[],
        current_batch=[],
        attempt_count={},
        final_status="pending",
        thread_id="test-thread-id",
        started_at=time.time(),
        migration_report=None,
        langfuse_trace_id=None,
    )


def test_graph_compiles() -> None:
    """StateGraph must compile without errors."""
    g = graph_builder.compile()
    assert g is not None


@pytest.mark.asyncio
async def test_empty_queue_skips_transform(compiled_graph) -> None:  # type: ignore[no-untyped-def]
    """With an empty task queue, dispatch routes to test directly.

    Nodes visited: plan → dispatch → test → critic → finalize.
    transform_task is NOT visited because there are no tasks to send.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial = _initial_state()

    with patch("agents.planner.run_planner", new=AsyncMock(return_value=[])):
        result = await compiled_graph.ainvoke(initial, config=config)

    assert result["final_status"] == "success", (
        f"expected 'success', got {result['final_status']}"
    )

    history = [c async for c in compiled_graph.aget_state_history(config)]
    visited = {task.name for snapshot in history for task in snapshot.tasks}

    expected_nodes = {"plan", "dispatch", "test", "critic", "finalize"}
    assert expected_nodes.issubset(visited), (
        f"Not all nodes were visited. visited={visited}, missing={expected_nodes - visited}"
    )
    assert "transform_task" not in visited, (
        "transform_task should not be visited with an empty task queue"
    )


@pytest.mark.asyncio
async def test_with_tasks_visits_transform_task(compiled_graph) -> None:  # type: ignore[no-untyped-def]
    """With one task in the queue, dispatch fans out to transform_task via Send."""
    task = MigrationTask(
        task_id="t1",
        module_path=str(Path(FIXTURE_REPO) / "app.py"),
        description="Migrate app.py",
        priority=0,
        complexity="standard",
        predicted_changes=[],
        retrieved_context_ids=[],
        depends_on=[],
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial = _initial_state()

    canned_result = TaskResult(
        module_path=task.module_path,
        status="transformed",
        branch_name="migration/run/app",
        tokens_used=100,
    )

    canned_critic = CriticResult(
        module_path=task.module_path,
        verdict="pass",
        branch_name=canned_result.branch_name,
    )

    with patch("agents.planner.run_planner", new=AsyncMock(return_value=[task])):
        with patch("agents.transform.run_transform", new=AsyncMock(return_value=canned_result)):
            with patch("agents.critic.run_critic", new=AsyncMock(return_value=canned_critic)):
                result = await compiled_graph.ainvoke(initial, config=config)

    history = [c async for c in compiled_graph.aget_state_history(config)]
    visited = {t.name for snapshot in history for t in snapshot.tasks}

    assert "transform_task" in visited, (
        f"transform_task not in visited={visited}"
    )
    assert result["final_status"] == "success"
    assert len(result["task_results"]) == 1
    assert result["task_results"][0].status == "transformed"
