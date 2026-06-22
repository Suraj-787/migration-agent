"""Planner agent: classifies migration complexity per module and builds the task queue.

Entry point: run_planner(dep_graph_id, spec, ...)
  1. Loads the dependency graph from Postgres.
  2. Computes leaf-first migration batches via get_migration_order().
  3. For each module, retrieves RAG context then calls Gemini Flash once to
     classify complexity and predict required changes.
  4. Stores each task in Redis under key  task:{task_id}  as a hash with
     status stored as a plain string (not a serialised enum).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from loguru import logger
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from agents.llm_router import get_router
from rag.dep_graph import DepGraph, DependencyGraphRecord, get_migration_order
from rag.retriever import HybridRetriever, SearchHit
from workflows.state import MigrationSpec, MigrationTask

# Hard cap on file content sent to the LLM; keeps total prompt under the 8K
# input token budget even with three code + three doc chunks appended.
_MAX_FILE_CHARS = 3_000
_RAG_TOP_K = 3

_SYSTEM_PROMPT = """\
You are a senior migration engineer. Analyse the Python source file below and classify \
its migration effort.

Return JSON with exactly these keys:
  complexity        – one of "trivial", "standard", "complex"
  predicted_changes – list of up to 8 short, concrete transformation statements
  reasoning         – one sentence explaining the complexity rating

trivial  = only import aliases or symbol renames needed
standard = function signatures, decorators, or helpers must change
complex  = architectural changes, new abstractions, or significant logic rewrites needed
"""


class _PlannerLLMOutput(BaseModel):
    complexity: Literal["trivial", "standard", "complex"]
    predicted_changes: list[str]
    reasoning: str


# ---------------------------------------------------------------------------
# Resource factories
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_engine() -> AsyncEngine:
    dsn = (
        f"postgresql+asyncpg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    return create_async_engine(dsn)


def _make_redis() -> Redis:
    return Redis.from_url(  # type: ignore[return-value]
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )


def _make_retrievers() -> tuple[HybridRetriever, HybridRetriever]:
    return HybridRetriever("code_chunks"), HybridRetriever("doc_chunks")


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------


async def _load_dep_graph(dep_graph_id: UUID, engine: AsyncEngine) -> DepGraph:
    """Load a DepGraph from the dependency_graphs Postgres table by UUID."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(DependencyGraphRecord).where(DependencyGraphRecord.id == dep_graph_id)
        )
        record = result.scalar_one()
    return DepGraph.from_json(record.graph_json)


# ---------------------------------------------------------------------------
# File-path helpers
# ---------------------------------------------------------------------------


def _module_to_file(module: str, repo_path: str) -> str | None:
    """Resolve a dotted module name to an absolute file path, or None."""
    base = Path(repo_path) / module.replace(".", "/")
    if (py := base.with_suffix(".py")).exists():
        return str(py)
    if (init := base / "__init__.py").exists():
        return str(init)
    return None


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------


async def _retrieve_context(
    module: str,
    file_content: str,
    spec: MigrationSpec,
    code_retriever: HybridRetriever,
    doc_retriever: HybridRetriever,
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Fetch the top-3 code chunks and top-3 migration-guide chunks in parallel.

    Returns empty lists if Qdrant is unreachable — the planner degrades gracefully
    and still classifies modules using the LLM without RAG context.
    """
    code_query = (
        f"{spec.source_framework} {module} to {spec.target_framework}\n"
        + file_content[:500]
    )
    doc_query = (
        f"{spec.source_framework} {spec.source_version} to "
        f"{spec.target_framework} {spec.target_version} migration guide"
    )
    try:
        code_hits, doc_hits = await asyncio.gather(
            code_retriever.search(code_query, top_k=_RAG_TOP_K),
            doc_retriever.search(doc_query, top_k=_RAG_TOP_K),
        )
        return code_hits, doc_hits
    except Exception as exc:
        logger.warning(
            "RAG retrieval unavailable for module={} (Qdrant down?): {}", module, exc
        )
        return [], []


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(
    module: str,
    file_content: str,
    code_hits: list[SearchHit],
    doc_hits: list[SearchHit],
    spec: MigrationSpec,
) -> list[Any]:
    """Assemble the SystemMessage + HumanMessage pair for the planner LLM call."""
    code_ctx = "\n\n".join(
        f"[Code example {i + 1}]\n{hit.payload.get('content', '')[:400]}"
        for i, hit in enumerate(code_hits)
    )
    doc_ctx = "\n\n".join(
        f"[Migration guide {i + 1}]\n{hit.payload.get('content', '')[:400]}"
        for i, hit in enumerate(doc_hits)
    )
    user_content = (
        f"## Migration: {spec.source_framework} {spec.source_version} → "
        f"{spec.target_framework} {spec.target_version}\n\n"
        f"## Module: {module}\n```python\n{file_content}\n```\n\n"
        f"## Similar code patterns\n{code_ctx or '(none retrieved)'}\n\n"
        f"## Migration guide excerpts\n{doc_ctx or '(none retrieved)'}"
    )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)]


# ---------------------------------------------------------------------------
# LLM classification (one call per module)
# ---------------------------------------------------------------------------


async def _classify_module(
    module: str,
    file_content: str,
    code_hits: list[SearchHit],
    doc_hits: list[SearchHit],
    spec: MigrationSpec,
    session_id: str | None,
) -> _PlannerLLMOutput:
    """Call the planner LLM once per module, with one automatic fallback retry.

    On the first failure (e.g. Gemini 403), records enough errors to trip the
    circuit breaker so that get_client() switches to the fallback provider on
    the second attempt.
    """
    router = get_router()
    messages = _build_prompt(module, file_content, code_hits, doc_hits, spec)
    last_exc: Exception | None = None

    for attempt in range(2):  # attempt 0 = primary, attempt 1 = fallback
        client, callbacks = router.get_client(
            "planner",
            session_id=session_id,
            tags=["planner", f"module:{module}"],
        )
        structured = client.with_structured_output(_PlannerLLMOutput)
        try:
            result = cast(
                _PlannerLLMOutput,
                await structured.ainvoke(messages, config={"callbacks": callbacks}),  # type: ignore[arg-type]
            )
            router.record_success("planner")
            logger.debug(
                "Planner classified module={} complexity={}", module, result.complexity
            )
            return result
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_rate_limit = (
                "429" in msg or "rate limit" in msg or "too many" in msg
                or "quota" in msg or "resource_exhausted" in msg
            )
            if is_rate_limit:
                # Record 3 errors to guarantee the circuit trips before the next attempt.
                for _ in range(3):
                    router.record_rate_limit_error("planner")
            logger.warning(
                "Planner attempt={} failed for module={} (rate_limit={}): {}",
                attempt + 1, module, is_rate_limit, exc,
            )

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Redis storage
# ---------------------------------------------------------------------------


async def _store_task(task: MigrationTask, redis: Redis) -> None:
    """Write a MigrationTask as a Redis hash. status is a plain string."""
    await redis.hset(  # type: ignore[misc]
        f"task:{task.task_id}",
        mapping={
            "task_id": task.task_id,
            "module_path": task.module_path,
            "priority": str(task.priority),
            "complexity": task.complexity,
            "predicted_changes": json.dumps(task.predicted_changes),
            "retrieved_context_ids": json.dumps(task.retrieved_context_ids),
            "depends_on": json.dumps(task.depends_on),
            "description": task.description,
            "status": "pending",
        },
    )
    logger.debug(
        "Stored task:{} module={} complexity={} priority={}",
        task.task_id,
        task.module_path,
        task.complexity,
        task.priority,
    )


# ---------------------------------------------------------------------------
# Per-module planner
# ---------------------------------------------------------------------------


async def _plan_one_module(
    module: str,
    priority: int,
    dep_graph: DepGraph,
    module_to_task_id: dict[str, str],
    spec: MigrationSpec,
    code_retriever: HybridRetriever,
    doc_retriever: HybridRetriever,
    session_id: str | None,
) -> MigrationTask | None:
    """Produce a MigrationTask for one module, or None if the source file is missing."""
    file_path = _module_to_file(module, dep_graph.repo_path)
    if file_path is None:
        logger.warning("Planner: no source file for module={}, skipping", module)
        return None

    file_content = await asyncio.to_thread(
        Path(file_path).read_text, encoding="utf-8", errors="replace"
    )
    file_content = file_content[:_MAX_FILE_CHARS]

    code_hits, doc_hits = await _retrieve_context(
        module, file_content, spec, code_retriever, doc_retriever
    )
    llm_out = await _classify_module(
        module, file_content, code_hits, doc_hits, spec, session_id
    )

    # Edges A→B mean A imports B, so B's task must run before A's.
    successors = list(dep_graph.graph.successors(module))
    depends_on = [module_to_task_id[s] for s in successors if s in module_to_task_id]

    return MigrationTask(
        task_id=str(uuid.uuid4()),
        module_path=file_path,
        description=(
            f"Migrate {module} from {spec.source_framework} {spec.source_version} "
            f"to {spec.target_framework} {spec.target_version}"
        ),
        priority=priority,
        complexity=llm_out.complexity,
        predicted_changes=llm_out.predicted_changes,
        retrieved_context_ids=[h.id for h in code_hits + doc_hits],
        depends_on=depends_on,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@observe(name="run_planner")
async def run_planner(
    dep_graph_id: UUID,
    spec: MigrationSpec,
    *,
    engine: AsyncEngine | None = None,
    redis: Redis | None = None,
    session_id: str | None = None,
) -> list[MigrationTask]:
    """Plan a migration run: classify every module and store tasks in Redis.

    Args:
        dep_graph_id: UUID of the DepGraph record in Postgres.
        spec:         Migration spec (source/target framework + version).
        engine:       SQLAlchemy async engine; falls back to singleton from env.
        redis:        Redis async client; falls back to REDIS_URL env var.
        session_id:   Langfuse session ID for grouping all planner traces.

    Returns:
        Ordered list of MigrationTask objects (one per resolvable module).
    """
    _engine = engine if engine is not None else _get_engine()
    _redis = redis if redis is not None else _make_redis()

    dep_graph = await _load_dep_graph(dep_graph_id, _engine)
    batches = get_migration_order(dep_graph)
    code_retriever, doc_retriever = _make_retrievers()

    module_to_task_id: dict[str, str] = {}
    tasks: list[MigrationTask] = []
    priority = 0

    for batch in batches:
        for module in batch:
            task = await _plan_one_module(
                module,
                priority,
                dep_graph,
                module_to_task_id,
                spec,
                code_retriever,
                doc_retriever,
                session_id,
            )
            if task is not None:
                module_to_task_id[module] = task.task_id
                tasks.append(task)
                await _store_task(task, _redis)
                priority += 1

    logger.info(
        "Planner complete: {} tasks for {} (dep_graph_id={})",
        len(tasks),
        dep_graph.repo_path,
        dep_graph_id,
    )
    return tasks
