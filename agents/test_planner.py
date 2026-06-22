"""Unit tests for agents/planner.py.

No real Postgres, Qdrant, Redis, or LLM is needed — infra is patched out.
The dep graph is built directly from tests/fixtures/sample_flask_app/ so
the graph construction logic is exercised end-to-end.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agents.planner import _PlannerLLMOutput, run_planner
from rag.dep_graph import DepGraph, DepGraphBuilder
from workflows.state import MigrationSpec

FIXTURE_REPO = str(
    Path(__file__).parent.parent / "tests" / "fixtures" / "sample_flask_app"
)

_CANNED_OUTPUT = _PlannerLLMOutput(
    complexity="standard",
    predicted_changes=[
        "Replace @app.route with FastAPI APIRouter",
        "Add return type annotations to route functions",
        "Convert request.get_json() to Pydantic request body",
    ],
    reasoning="Standard route and decorator migration with minor type annotation work.",
)


class _FakeRedis:
    """In-memory Redis substitute for testing hset / hgetall."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, str]] = {}

    async def hset(self, name: str, mapping: dict[str, str] | None = None, **_: object) -> int:
        self.store[name] = mapping or {}
        return len(mapping or {})

    async def hgetall(self, name: str) -> dict[str, str]:
        return self.store.get(name, {})


@pytest.fixture()
def flask_dep_graph() -> DepGraph:
    return DepGraphBuilder(repo_path=FIXTURE_REPO).build()


@pytest.fixture()
def migration_spec() -> MigrationSpec:
    return MigrationSpec(
        source_framework="flask",
        target_framework="fastapi",
        source_version="2.3",
        target_version="0.115",
    )


@pytest.fixture()
def fake_redis() -> _FakeRedis:
    return _FakeRedis()


def _stub_infra(dep_graph: DepGraph) -> ExitStack:
    """Return an ExitStack that patches all external I/O for one test."""
    stack = ExitStack()
    stack.enter_context(
        patch("agents.planner._load_dep_graph", new=AsyncMock(return_value=dep_graph))
    )
    stack.enter_context(
        patch("agents.planner._make_retrievers", return_value=(MagicMock(), MagicMock()))
    )
    stack.enter_context(
        patch("agents.planner._retrieve_context", new=AsyncMock(return_value=([], [])))
    )
    stack.enter_context(
        patch("agents.planner._classify_module", new=AsyncMock(return_value=_CANNED_OUTPUT))
    )
    return stack


@pytest.mark.asyncio
async def test_one_task_per_module(
    flask_dep_graph: DepGraph, migration_spec: MigrationSpec, fake_redis: _FakeRedis
) -> None:
    """Planner must produce exactly one task for each module in the dep graph."""
    with _stub_infra(flask_dep_graph):
        tasks = await run_planner(
            dep_graph_id=uuid4(),
            spec=migration_spec,
            engine=AsyncMock(),
            redis=fake_redis,  # type: ignore[arg-type]
        )

    assert len(tasks) == 3, f"expected 3 tasks, got {len(tasks)}"
    assert all(t.module_path.endswith(".py") for t in tasks)
    assert len({t.task_id for t in tasks}) == 3, "task_ids must be unique"


@pytest.mark.asyncio
async def test_dep_order_respected(
    flask_dep_graph: DepGraph, migration_spec: MigrationSpec, fake_redis: _FakeRedis
) -> None:
    """config and models must be planned before app (lower priority = earlier)."""
    with _stub_infra(flask_dep_graph):
        tasks = await run_planner(
            dep_graph_id=uuid4(),
            spec=migration_spec,
            engine=AsyncMock(),
            redis=fake_redis,  # type: ignore[arg-type]
        )

    by_stem = {Path(t.module_path).stem: t for t in tasks}
    assert by_stem["config"].priority < by_stem["app"].priority
    assert by_stem["models"].priority < by_stem["app"].priority


@pytest.mark.asyncio
async def test_tasks_stored_in_redis(
    flask_dep_graph: DepGraph, migration_spec: MigrationSpec, fake_redis: _FakeRedis
) -> None:
    """Every task must appear in Redis as a hash with status='pending'."""
    with _stub_infra(flask_dep_graph):
        tasks = await run_planner(
            dep_graph_id=uuid4(),
            spec=migration_spec,
            engine=AsyncMock(),
            redis=fake_redis,  # type: ignore[arg-type]
        )

    for task in tasks:
        key = f"task:{task.task_id}"
        assert key in fake_redis.store, f"missing Redis key {key}"
        data = fake_redis.store[key]
        assert data["status"] == "pending", "status must be a plain string 'pending'"
        assert data["task_id"] == task.task_id
        assert data["complexity"] in ("trivial", "standard", "complex")


@pytest.mark.asyncio
async def test_depends_on_matches_dep_graph(
    flask_dep_graph: DepGraph, migration_spec: MigrationSpec, fake_redis: _FakeRedis
) -> None:
    """app's depends_on must contain task IDs for config and models; leaves have none."""
    with _stub_infra(flask_dep_graph):
        tasks = await run_planner(
            dep_graph_id=uuid4(),
            spec=migration_spec,
            engine=AsyncMock(),
            redis=fake_redis,  # type: ignore[arg-type]
        )

    by_stem = {Path(t.module_path).stem: t for t in tasks}
    app_task = by_stem["app"]
    assert by_stem["config"].task_id in app_task.depends_on
    assert by_stem["models"].task_id in app_task.depends_on
    assert by_stem["config"].depends_on == []
    assert by_stem["models"].depends_on == []
