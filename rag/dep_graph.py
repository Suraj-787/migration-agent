"""Module-level dependency graph for Python and JS/TS repos.

Strategies: PythonDepExtractor (stdlib ast) and JSDepExtractor (tree-sitter).
Both implement the DepExtractor protocol.

Usage:
    builder = DepGraphBuilder(repo_path="/abs/path/to/repo")
    graph = builder.build()
    batches = get_migration_order(graph)
    record_id = await persist_graph(graph, session)
"""
from __future__ import annotations

import ast
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx
from loguru import logger
from sqlalchemy import DateTime, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)

_PY_EXTS: frozenset[str] = frozenset({".py"})
_JS_EXTS: frozenset[str] = frozenset({".js", ".jsx", ".ts", ".tsx"})


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class DepExtractor(Protocol):
    """Strategy: extract intra-repo import dependencies from one source file."""

    def extract(self, file_path: str, repo_root: str) -> set[str]:
        """Return dotted module paths that file_path imports from within the repo."""
        ...


# ── Helpers ──────────────────────────────────────────────────────────────────


def _file_to_module(file_path: str, repo_root: str) -> str:
    """Convert an absolute file path to a dotted module path relative to repo_root."""
    rel = Path(file_path).relative_to(repo_root)
    parts = list(rel.parts)
    if not parts:
        return ""
    last = parts[-1]
    if last == "__init__.py":
        parts = parts[:-1]
    elif last.endswith(".py"):
        parts[-1] = last[:-3]
    else:
        stem = Path(last).stem
        parts[-1] = stem
        if stem == "index" and len(parts) > 1:
            parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(level: int, module_str: str | None, current_module: str) -> str | None:
    """Resolve a relative import to an absolute dotted module path.

    level=1 → same package; level=2 → parent package, etc.
    """
    parts = current_module.split(".")
    if level > len(parts):
        return None
    base_parts = parts[:-level] if level > 0 else parts[:]
    if module_str:
        base_parts = base_parts + module_str.split(".")
    return ".".join(base_parts) if base_parts else None


# ── Python extractor ─────────────────────────────────────────────────────────


class PythonDepExtractor:
    """Extract import dependencies from Python files using stdlib ast."""

    def extract(self, file_path: str, repo_root: str) -> set[str]:
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            logger.warning("Syntax error parsing {}; skipping", file_path)
            return set()

        current = _file_to_module(file_path, repo_root)
        repo_root_path = Path(repo_root)
        deps: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if self._is_internal(top, repo_root_path):
                        deps.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    self._handle_relative(node, current, deps)
                elif node.module:
                    top = node.module.split(".")[0]
                    if self._is_internal(top, repo_root_path):
                        deps.add(node.module)

        return deps

    def _handle_relative(
        self,
        node: ast.ImportFrom,
        current_module: str,
        deps: set[str],
    ) -> None:
        level = node.level or 0
        if node.module:
            resolved = _resolve_relative(level, node.module, current_module)
            if resolved:
                deps.add(resolved)
        else:
            # `from . import models, utils` — each name may be a sub-module.
            # base is None when the current module is at the repo root (no package prefix).
            base = _resolve_relative(level, None, current_module)
            for alias in node.names:
                deps.add(f"{base}.{alias.name}" if base else alias.name)

    def _is_internal(self, top_package: str, repo_root: Path) -> bool:
        return (repo_root / top_package).is_dir() or (repo_root / f"{top_package}.py").is_file()


# ── JS/TS extractor ──────────────────────────────────────────────────────────


class JSDepExtractor:
    """Extract import dependencies from JS/TS files using tree-sitter."""

    def __init__(self) -> None:
        self._ts_parser: Any = None
        self._tsx_parser: Any = None
        self._available = False
        self._init_parsers()

    def _init_parsers(self) -> None:
        try:
            import tree_sitter_typescript as tsts
            from tree_sitter import Language, Parser

            self._ts_parser = Parser(Language(tsts.language_typescript()))
            self._tsx_parser = Parser(Language(tsts.language_tsx()))
            self._available = True
        except Exception as exc:  # pragma: no cover
            logger.warning("JS extractor unavailable (tree-sitter error): {}", exc)

    def extract(self, file_path: str, repo_root: str) -> set[str]:
        if not self._available:  # pragma: no cover
            return set()

        suffix = Path(file_path).suffix.lower()
        parser = self._tsx_parser if suffix in {".tsx", ".jsx"} else self._ts_parser

        try:
            source = Path(file_path).read_bytes()
            tree = parser.parse(source)
        except Exception as exc:  # pragma: no cover
            logger.warning("tree-sitter parse error for {}: {}", file_path, exc)
            return set()

        file_dir = Path(file_path).parent
        repo_root_path = Path(repo_root)
        deps: set[str] = set()
        self._collect_specifiers(tree.root_node, file_dir, repo_root_path, deps)
        return deps

    def _collect_specifiers(
        self,
        node: Any,
        file_dir: Path,
        repo_root: Path,
        deps: set[str],
    ) -> None:
        if node.type == "import_statement":
            src = node.child_by_field_name("source")
            if src:
                specifier = src.text.decode("utf-8").strip("'\"")
                self._resolve_js(specifier, file_dir, repo_root, deps)

        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn and fn.text in (b"require",) and args:
                for child in args.children:
                    if child.type == "string":
                        specifier = child.text.decode("utf-8").strip("'\"")
                        self._resolve_js(specifier, file_dir, repo_root, deps)

        for child in node.children:
            self._collect_specifiers(child, file_dir, repo_root, deps)

    def _resolve_js(
        self, specifier: str, file_dir: Path, repo_root: Path, deps: set[str]
    ) -> None:
        if not specifier.startswith(("./", "../")):
            return  # external package

        resolved = (file_dir / specifier).resolve()
        candidates = [
            resolved,
            *[resolved.with_suffix(ext) for ext in (".ts", ".tsx", ".js", ".jsx")],
            *[resolved / f"index{ext}" for ext in (".ts", ".tsx", ".js", ".jsx")],
        ]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    module = _file_to_module(str(candidate), str(repo_root))
                    if module:
                        deps.add(module)
                    return
                except ValueError:
                    pass


# ── DepGraph ─────────────────────────────────────────────────────────────────


class DepGraph:
    """Wraps a NetworkX DiGraph (edge A→B = A imports B) with migration analytics."""

    def __init__(self, graph: nx.DiGraph, repo_path: str) -> None:
        self.graph = graph
        self.repo_path = repo_path
        self._cycles: list[list[str]] | None = None

    @property
    def cycles(self) -> list[list[str]]:
        if self._cycles is None:
            self._cycles = list(nx.simple_cycles(self.graph))
        return self._cycles

    @property
    def leaf_modules(self) -> list[str]:
        """Nodes with out_degree=0: they import nothing within the repo."""
        return sorted(n for n in self.graph.nodes if self.graph.out_degree(n) == 0)

    @property
    def entry_points(self) -> list[str]:
        """Nodes with in_degree=0: nothing in the repo imports them."""
        return sorted(n for n in self.graph.nodes if self.graph.in_degree(n) == 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": list(self.graph.nodes),
            "edges": [{"source": u, "target": v} for u, v in self.graph.edges],
            "repo_path": self.repo_path,
            "leaf_modules": self.leaf_modules,
            "entry_points": self.entry_points,
            "cycles": self.cycles,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DepGraph:
        g: nx.DiGraph = nx.DiGraph()
        g.add_nodes_from(data["nodes"])
        for edge in data["edges"]:
            g.add_edge(edge["source"], edge["target"])
        return cls(graph=g, repo_path=data["repo_path"])


# ── Builder ───────────────────────────────────────────────────────────────────


class DepGraphBuilder:
    """Build a module-level dependency graph for a Python or JS/TS repo."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self._py_extractor = PythonDepExtractor()
        self._js_extractor = JSDepExtractor()

    def build(self) -> DepGraph:
        py_files, js_files = self._collect_modules()
        g: nx.DiGraph = nx.DiGraph()
        g.add_nodes_from(py_files)
        g.add_nodes_from(js_files)
        known = set(g.nodes)
        self._add_edges(g, py_files, self._py_extractor, known)
        self._add_edges(g, js_files, self._js_extractor, known)

        n_cycles = len(list(nx.simple_cycles(g)))
        if n_cycles:
            logger.warning("Dependency graph has {} cycle(s) in {}", n_cycles, self.repo_path)

        logger.info(
            "Built dep graph: {} nodes, {} edges for {}",
            g.number_of_nodes(),
            g.number_of_edges(),
            self.repo_path,
        )
        return DepGraph(graph=g, repo_path=self.repo_path)

    def _collect_modules(self) -> tuple[dict[str, str], dict[str, str]]:
        """Walk repo and return {module_path: file_path} dicts for py and js."""
        py_files: dict[str, str] = {}
        js_files: dict[str, str] = {}
        root = Path(self.repo_path)

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
            ]
            for filename in sorted(filenames):
                full = str(Path(dirpath) / filename)
                suffix = Path(filename).suffix.lower()
                if suffix in _PY_EXTS:
                    mod = _file_to_module(full, self.repo_path)
                    if mod:
                        py_files[mod] = full
                elif suffix in _JS_EXTS:
                    mod = _file_to_module(full, self.repo_path)
                    if mod:
                        js_files[mod] = full

        return py_files, js_files

    def _add_edges(
        self,
        g: nx.DiGraph,
        files: dict[str, str],
        extractor: DepExtractor,
        known: set[str],
    ) -> None:
        for module, file_path in files.items():
            for dep in extractor.extract(file_path, self.repo_path):
                target = dep if dep in known else None
                if target and target != module:
                    g.add_edge(module, target)


# ── Migration order ───────────────────────────────────────────────────────────


def get_migration_order(graph: DepGraph) -> list[list[str]]:
    """Return parallel migration batches in leaf-first order.

    Each batch contains modules with no dependencies on others in the same batch.
    Modules in batch N depend only on modules in batches 0..N-1.
    Cyclic modules are collected in a final batch; they need manual review.
    """
    cycle_members: set[str] = {n for cycle in graph.cycles for n in cycle}

    dag = graph.graph.copy()
    for cycle in graph.cycles:
        for i, node in enumerate(cycle):
            successor = cycle[(i + 1) % len(cycle)]
            if dag.has_edge(node, successor):
                dag.remove_edge(node, successor)

    batches: list[list[str]] = []
    for generation in nx.topological_generations(dag.reverse(copy=True)):
        batch = sorted(n for n in generation if n not in cycle_members)
        if batch:
            batches.append(batch)

    if cycle_members:
        logger.warning("Cyclic modules require manual review: {}", sorted(cycle_members))
        batches.append(sorted(cycle_members))

    return batches


# ── Persistence ───────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class DependencyGraphRecord(Base):
    __tablename__ = "dependency_graphs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_path: Mapped[str] = mapped_column(String, nullable=False, index=True)
    graph_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


async def create_tables(engine: Any) -> None:
    """Create the dependency_graphs table if it does not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def persist_graph(dep_graph: DepGraph, session: AsyncSession) -> uuid.UUID:
    """Insert a DepGraph record and return its UUID."""
    record = DependencyGraphRecord(
        repo_path=dep_graph.repo_path,
        graph_json=dep_graph.to_json(),
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    logger.info(
        "Persisted dependency graph for {} (id={})", dep_graph.repo_path, record.id
    )
    return record.id  # type: ignore[return-value]
