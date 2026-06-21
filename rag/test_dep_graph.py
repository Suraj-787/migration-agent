"""Unit tests for rag/dep_graph.py.

No external services required — all tests are pure in-memory.
Run with: uv run pytest rag/test_dep_graph.py -xvs
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import pytest

from rag.dep_graph import (
    DepGraph,
    DepGraphBuilder,
    PythonDepExtractor,
    _file_to_module,
    _resolve_relative,
    get_migration_order,
)

FLASK_FIXTURE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "sample_flask_app")
)


# ── Helper fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def flask_graph() -> DepGraph:
    return DepGraphBuilder(repo_path=FLASK_FIXTURE).build()


@pytest.fixture
def simple_graph() -> DepGraph:
    """A→B→C (leaf C, entry A) with no cycles."""
    g: nx.DiGraph = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c")])
    return DepGraph(graph=g, repo_path="/fake")


@pytest.fixture
def cyclic_graph() -> DepGraph:
    """A→B→A cycle plus independent leaf D."""
    g: nx.DiGraph = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "a"), ("c", "d")])
    return DepGraph(graph=g, repo_path="/fake")


# ── Unit: helpers ─────────────────────────────────────────────────────────────


def test_file_to_module_simple(tmp_path: Path) -> None:
    (tmp_path / "app" / "models").mkdir(parents=True)
    f = tmp_path / "app" / "models" / "user.py"
    f.write_text("")
    assert _file_to_module(str(f), str(tmp_path)) == "app.models.user"


def test_file_to_module_init(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    f = tmp_path / "app" / "__init__.py"
    f.write_text("")
    assert _file_to_module(str(f), str(tmp_path)) == "app"


def test_resolve_relative_same_package() -> None:
    result = _resolve_relative(1, "models", "app.views.users")
    assert result == "app.views.models"


def test_resolve_relative_parent_package() -> None:
    result = _resolve_relative(2, "config", "app.views.users")
    assert result == "app.config"


def test_resolve_relative_no_module() -> None:
    result = _resolve_relative(1, None, "app.views")
    assert result == "app"


def test_resolve_relative_overflow() -> None:
    result = _resolve_relative(10, "x", "app")
    assert result is None


# ── Unit: PythonDepExtractor ──────────────────────────────────────────────────


def test_python_extractor_absolute_import(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "__init__.py").write_text("")
    f = tmp_path / "main.py"
    f.write_text("import models\n")
    result = PythonDepExtractor().extract(str(f), str(tmp_path))
    assert "models" in result


def test_python_extractor_relative_import(tmp_path: Path) -> None:
    (tmp_path / "utils.py").write_text("")
    f = tmp_path / "main.py"
    f.write_text("from .utils import helper\n")
    result = PythonDepExtractor().extract(str(f), str(tmp_path))
    assert "utils" in result


def test_python_extractor_skips_external(tmp_path: Path) -> None:
    f = tmp_path / "main.py"
    f.write_text("import flask\nfrom sqlalchemy.orm import Session\n")
    result = PythonDepExtractor().extract(str(f), str(tmp_path))
    assert not result


def test_python_extractor_from_dot_import(tmp_path: Path) -> None:
    (tmp_path / "models.py").write_text("")
    f = tmp_path / "app.py"
    f.write_text("from . import models\n")
    result = PythonDepExtractor().extract(str(f), str(tmp_path))
    assert "models" in result


def test_python_extractor_syntax_error(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("def (: pass\n")
    result = PythonDepExtractor().extract(str(f), str(tmp_path))
    assert result == set()


# ── Integration: sample Flask app ─────────────────────────────────────────────


def test_flask_leaf_modules(flask_graph: DepGraph) -> None:
    leaves = flask_graph.leaf_modules
    assert "models" in leaves, f"Expected 'models' in leaves, got {leaves}"
    assert "config" in leaves, f"Expected 'config' in leaves, got {leaves}"
    assert "app" not in leaves, "'app' should not be a leaf (it imports models and config)"


def test_flask_entry_points(flask_graph: DepGraph) -> None:
    entries = flask_graph.entry_points
    assert "app" in entries, f"Expected 'app' in entry_points, got {entries}"


def test_flask_no_cycles(flask_graph: DepGraph) -> None:
    assert flask_graph.cycles == [], f"Unexpected cycles: {flask_graph.cycles}"


def test_flask_app_imports_models_and_config(flask_graph: DepGraph) -> None:
    g = flask_graph.graph
    assert g.has_edge("app", "models"), "Expected app→models edge"
    assert g.has_edge("app", "config"), "Expected app→config edge"


# ── Leaf detection: synthetic graph ──────────────────────────────────────────


def test_leaf_modules_simple(simple_graph: DepGraph) -> None:
    assert simple_graph.leaf_modules == ["c"]


def test_entry_points_simple(simple_graph: DepGraph) -> None:
    assert simple_graph.entry_points == ["a"]


# ── Migration order ───────────────────────────────────────────────────────────


def test_migration_order_leaves_first(flask_graph: DepGraph) -> None:
    batches = get_migration_order(flask_graph)
    assert batches, "Expected at least one batch"
    first_batch = batches[0]
    assert "models" in first_batch or "config" in first_batch, (
        f"Expected leaf modules in first batch, got {first_batch}"
    )
    # app must come after models and config
    all_before_app: set[str] = set()
    for batch in batches:
        if "app" in batch:
            break
        all_before_app.update(batch)
    assert "models" in all_before_app, "models must be scheduled before app"
    assert "config" in all_before_app, "config must be scheduled before app"


def test_parallel_batches_no_intra_deps(flask_graph: DepGraph) -> None:
    batches = get_migration_order(flask_graph)
    g = flask_graph.graph
    for batch in batches:
        for i, mod_a in enumerate(batch):
            for mod_b in batch[i + 1 :]:
                assert not g.has_edge(mod_a, mod_b), (
                    f"Intra-batch dependency found: {mod_a} → {mod_b}"
                )
                assert not g.has_edge(mod_b, mod_a), (
                    f"Intra-batch dependency found: {mod_b} → {mod_a}"
                )


def test_migration_order_simple(simple_graph: DepGraph) -> None:
    batches = get_migration_order(simple_graph)
    flat = [m for batch in batches for m in batch]
    assert flat.index("c") < flat.index("b") < flat.index("a")


def test_migration_order_cyclic_last(cyclic_graph: DepGraph) -> None:
    batches = get_migration_order(cyclic_graph)
    last_batch = batches[-1]
    cycle_members = {"a", "b"}
    assert cycle_members.issubset(set(last_batch)), (
        f"Expected cycle members in last batch, got {last_batch}"
    )
    # d should appear before cycle members
    all_non_cycle = [m for b in batches[:-1] for m in b]
    assert "d" in all_non_cycle, "Expected leaf 'd' before cycle batch"


def test_parallel_batches_no_intra_deps_synthetic(simple_graph: DepGraph) -> None:
    batches = get_migration_order(simple_graph)
    g = simple_graph.graph
    for batch in batches:
        for i, mod_a in enumerate(batch):
            for mod_b in batch[i + 1 :]:
                assert not g.has_edge(mod_a, mod_b)
                assert not g.has_edge(mod_b, mod_a)


# ── JSON serialization ────────────────────────────────────────────────────────


def test_graph_json_serializable(flask_graph: DepGraph) -> None:
    data = flask_graph.to_json()
    serialized = json.dumps(data)
    assert len(serialized) > 0


def test_graph_roundtrip_json(simple_graph: DepGraph) -> None:
    data = simple_graph.to_json()
    restored = DepGraph.from_json(data)
    assert set(restored.graph.nodes) == set(simple_graph.graph.nodes)
    assert set(restored.graph.edges) == set(simple_graph.graph.edges)
    assert restored.repo_path == simple_graph.repo_path


def test_json_contains_expected_keys(flask_graph: DepGraph) -> None:
    data = flask_graph.to_json()
    for key in ("nodes", "edges", "repo_path", "leaf_modules", "entry_points", "cycles"):
        assert key in data, f"Missing key '{key}' in to_json() output"
