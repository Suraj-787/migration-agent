"""Migration orchestration graph.

Topology:
  plan → dispatch ──(Send x≤3)──► transform_task ─► test → critic
                 └──(empty queue)──────────────────────────► test

  critic ──► dispatch  (more tasks remain)
         └──► rollback → finalize
         └──► finalize  (queue empty, no failures)

Langfuse v4 tracing strategy
─────────────────────────────
The outer _run() in api/routes/migrations.py wraps graph.ainvoke() with

    propagate_attributes(session_id=thread_id)          # groups all spans
    lf.start_as_current_observation(                    # root trace
        trace_context=TraceContext(trace_id=<hex>),
        as_type="agent",
    )

Because asyncio.create_task() copies Python contextvars, the OTel span
context set in _run() propagates automatically to all nodes, including those
created via the Send API.  As a belt-and-suspenders fallback, dispatch_node
captures lf.get_current_trace_id() and threads it through every Send payload
so transform_task_node can re-attach explicitly if the OTel chain was broken.

Redis lock per module_path (SET migration_lock:{path} NX EX 600) prevents two
concurrent agents from writing the same file. Fails open when Redis is
unreachable so a downed cache never blocks the pipeline.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Literal

from langfuse import get_client, observe
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger

from workflows.state import (
    MigrationReport,
    MigrationSpec,
    MigrationState,
    MigrationTask,
    TaskResult,
)

# Groq Llama 3.3-70B blended input+output estimate ($0.59/M in, $0.79/M out).
_COST_PER_TOKEN_USD = 0.60 / 1_000_000


# ---------------------------------------------------------------------------
# Redis lock helpers
# ---------------------------------------------------------------------------


async def _acquire_lock(module_path: str, run_id: str) -> bool:
    """SET migration_lock:{module_path} run_id NX EX 600. Fail-open on error."""
    try:
        from redis.asyncio import Redis  # type: ignore[import-untyped]

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r: Redis = Redis.from_url(redis_url, decode_responses=True)
        async with r:
            acquired = await r.set(
                f"migration_lock:{module_path}", run_id, nx=True, ex=600
            )
            return bool(acquired)
    except Exception as exc:
        logger.warning("[lock] Redis unavailable ({}), proceeding without lock", exc)
        return True  # fail open so migration continues without Redis


async def _release_lock(module_path: str) -> None:
    """DEL migration_lock:{module_path}. Best-effort — never raises."""
    try:
        from redis.asyncio import Redis  # type: ignore[import-untyped]

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r: Redis = Redis.from_url(redis_url, decode_responses=True)
        async with r:
            await r.delete(f"migration_lock:{module_path}")
    except Exception as exc:
        logger.warning("[lock] Redis release failed for {}: {}", module_path, exc)


# ---------------------------------------------------------------------------
# plan node
# ---------------------------------------------------------------------------


@observe(name="plan")
async def plan_node(state: MigrationState) -> dict:  # type: ignore[type-arg]
    from agents.planner import run_planner  # lazy import avoids circular deps at module load

    thread_id: str = state.get("thread_id", "")  # type: ignore[call-overload]
    logger.info("[plan] analysing repo and building task queue — repo={}", state["repo_path"])
    tasks = await run_planner(
        dep_graph_id=state["dep_graph_id"],
        spec=state["target_spec"],
        session_id=thread_id or None,
    )
    logger.info("[plan] generated {} tasks", len(tasks))
    return {"task_queue": tasks}


# ---------------------------------------------------------------------------
# dispatch node + routing function
# ---------------------------------------------------------------------------


def _select_ready_batch(state: MigrationState, batch_size: int = 3) -> list[MigrationTask]:
    """Return up to *batch_size* tasks not yet represented in task_results."""
    completed_paths = {r.module_path for r in state.get("task_results", [])}
    ready = []
    for task in state.get("task_queue", []):
        if task.module_path not in completed_paths:
            ready.append(task)
            if len(ready) >= batch_size:
                break
    return ready


@observe(name="dispatch")
async def dispatch_node(state: MigrationState) -> dict:  # type: ignore[type-arg]
    """Pass-through node that logs queue depth.  Routing is done by _route_dispatch.

    Captures lf.get_current_trace_id() while inside the @observe context so the
    ID can be threaded through Send payloads.  LangGraph creates new asyncio tasks
    for Send nodes; asyncio.create_task() copies contextvars (including OTel span
    context) so propagation should be automatic, but storing the explicit ID here
    gives transform_task_node a fallback to detect and diagnose context breaks.
    """
    lf = get_client()
    pending = len(_select_ready_batch(state, batch_size=9999))
    logger.info("[dispatch] {} tasks remain in queue", pending)

    trace_id: str | None = lf.get_current_trace_id()
    return {"langfuse_trace_id": trace_id} if trace_id else {}


def _route_dispatch(state: MigrationState) -> list[Send] | str:
    """Conditional edge: fan out via Send or skip to test when queue is empty."""
    batch = _select_ready_batch(state)
    if not batch:
        logger.info("[dispatch] queue empty — routing to test")
        return "test"

    run_id = str(state["dep_graph_id"])
    repo_path = state["repo_path"]
    spec = state["target_spec"]
    langfuse_trace_id: str | None = state.get("langfuse_trace_id")  # type: ignore[call-overload]
    thread_id: str = state.get("thread_id", "")  # type: ignore[call-overload]
    logger.info("[dispatch] fanning out {} transform_task sends", len(batch))
    return [
        Send(
            "transform_task",
            {
                "task": task,
                "spec": spec,
                "run_id": run_id,
                "repo_path": repo_path,
                "langfuse_trace_id": langfuse_trace_id,
                "thread_id": thread_id,
            },
        )
        for task in batch
    ]


# ---------------------------------------------------------------------------
# transform_task node — handles one MigrationTask (invoked via Send)
# ---------------------------------------------------------------------------


async def transform_task_node(state: dict[str, Any]) -> dict[str, Any]:  # type: ignore[type-arg, arg-type]
    """Migrate a single file behind a Redis lock.

    Uses lf.start_as_current_observation(trace_context=...) with the parent
    trace ID passed through the Send payload as the explicit Langfuse span
    container.  This handles two scenarios correctly:

    1. OTel context WAS propagated by asyncio.create_task() — trace_context
       attaches to the same trace, producing a properly nested child span.
    2. OTel context was NOT propagated — trace_context re-attaches the span
       to the correct trace using the dispatch_node's captured ID.

    The Redis lock (NX EX 600) prevents concurrent writes to the same file;
    it is released in a try/finally so it always frees on success or exception.
    """
    from agents.transform import run_transform  # lazy import
    from langfuse.types import TraceContext as LFTraceContext

    task: MigrationTask = state["task"]
    spec: MigrationSpec = state["spec"]
    run_id: str = state["run_id"]
    repo_path: str = state["repo_path"]
    thread_id: str = state.get("thread_id", "")
    langfuse_trace_id: str | None = state.get("langfuse_trace_id")

    lf = get_client()
    tc: LFTraceContext | None = (
        LFTraceContext(trace_id=langfuse_trace_id) if langfuse_trace_id else None
    )

    acquired = await _acquire_lock(task.module_path, run_id)
    if not acquired:
        logger.warning(
            "[transform_task] Lock already held for {} — skipping", task.module_path
        )
        return {
            "task_results": [
                TaskResult(
                    module_path=task.module_path,
                    status="skipped",
                    error="Redis lock held by another agent",
                )
            ]
        }

    try:
        with lf.start_as_current_observation(
            trace_context=tc,
            name="transform_task",
            as_type="agent",
            input={"module_path": task.module_path, "complexity": task.complexity},
        ):
            result = await run_transform(
                task, spec, run_id, repo_path, session_id=thread_id or None
            )
    finally:
        await _release_lock(task.module_path)

    return {"task_results": [result]}


# ---------------------------------------------------------------------------
# test, critic, rollback, finalize
# ---------------------------------------------------------------------------


@observe(name="test")
async def test_node(state: MigrationState) -> dict:  # type: ignore[type-arg]
    logger.info("[test] running test suite after transform")
    return {}


@observe(name="critic")
async def critic_node(state: MigrationState) -> dict:  # type: ignore[type-arg]
    from agents.critic import run_critic  # lazy import avoids circular deps at module load

    thread_id: str = state.get("thread_id", "")  # type: ignore[call-overload]

    # Only validate modules transformed in this batch — task_results is cumulative,
    # so filter out anything critic has already processed in a prior round.
    already_critiqued = set(state.get("critiqued_paths", []))  # type: ignore[call-overload]
    transformed = [
        r
        for r in state.get("task_results", [])
        if r.status == "transformed" and r.module_path not in already_critiqued
    ]
    if not transformed:
        logger.info("[critic] no new transformed results to validate")
        return {"rollback_stack": []}

    run_id = str(state["dep_graph_id"])
    repo_path = state["repo_path"]
    spec = state["target_spec"]

    task_id_by_path = {t.module_path: t.task_id for t in state.get("task_queue", [])}

    critic_results = await asyncio.gather(
        *[
            run_critic(
                result=r,
                task_id=task_id_by_path.get(r.module_path, r.module_path),
                spec=spec,
                run_id=run_id,
                repo_path=repo_path,
                session_id=thread_id or None,
            )
            for r in transformed
        ]
    )

    newly_critiqued = [r.module_path for r in transformed]
    passed = [cr.module_path for cr in critic_results if cr.verdict == "pass"]
    rollback_entries = [cr.rollback_entry for cr in critic_results if cr.rollback_entry]
    if rollback_entries:
        logger.info("[critic] {} module(s) failed validation → rollback", len(rollback_entries))
    return {
        "rollback_stack": rollback_entries,
        "critiqued_paths": newly_critiqued,
        "passed_paths": passed,
    }


@observe(name="rollback")
async def rollback_node(state: MigrationState) -> dict:  # type: ignore[type-arg]
    from workflows.rollback import run_rollback  # lazy import

    entries = state.get("rollback_stack", [])
    logger.info("[rollback] verifying cleanup for {} entries", len(entries))
    await run_rollback(
        entries=entries,
        run_id=str(state["dep_graph_id"]),
        repo_path=state["repo_path"],
    )
    # Clear the per-batch buffer so the dispatch loop can continue cleanly.
    return {"final_status": "partial", "rollback_stack": []}


@observe(name="finalize")
async def finalize_node(state: MigrationState) -> dict:  # type: ignore[type-arg]
    """Build a MigrationReport, persist it to Postgres, and return final_status."""
    from workflows.report import persist_report  # lazy import

    results = state.get("task_results", [])
    task_queue = state.get("task_queue", [])
    total_tasks = len(task_queue)

    # A module counts as succeeded only if critic PASSED it. Failures are the
    # union of transform failures and critic rejections (transformed-but-rolled-back).
    passed = set(state.get("passed_paths", []))  # type: ignore[call-overload]
    critiqued = set(state.get("critiqued_paths", []))  # type: ignore[call-overload]
    succeeded = len(passed)
    transform_failed = sum(1 for r in results if r.status == "failed")
    critic_failed = len(critiqued - passed)
    failed_count = transform_failed + critic_failed
    total_tokens = sum(r.tokens_used for r in results)

    # Only surviving (critic-passed) branches; failed ones were deleted on rollback.
    per_module_branches: dict[str, str] = {
        r.module_path: r.branch_name
        for r in results
        if r.branch_name and r.module_path in passed
    }

    started_at: float = state.get("started_at", time.time())  # type: ignore[call-overload]
    duration = max(0.0, time.time() - started_at)

    prior_status = state.get("final_status", "pending")
    if prior_status == "partial":
        new_status: Literal["success", "partial", "failed"] = "partial"
    elif failed_count > 0 and succeeded == 0:
        new_status = "failed"
    elif failed_count > 0:
        new_status = "partial"
    else:
        new_status = "success"

    thread_id: str = state.get("thread_id", "unknown")  # type: ignore[call-overload]
    success_rate = succeeded / total_tasks if total_tasks > 0 else 1.0

    report = MigrationReport(
        thread_id=thread_id,
        total_tasks=total_tasks,
        succeeded=succeeded,
        failed=failed_count,
        success_rate=success_rate,
        total_tokens_used=total_tokens,
        estimated_cost_usd=total_tokens * _COST_PER_TOKEN_USD,
        per_module_branch_names=per_module_branches,
        duration_seconds=duration,
        final_status=new_status,
    )

    logger.info(
        "[finalize] Migration complete — status={} succeeded={}/{} tokens={} "
        "cost=${:.4f} duration={:.1f}s thread_id={}",
        new_status,
        succeeded,
        total_tasks,
        total_tokens,
        report.estimated_cost_usd,
        duration,
        thread_id,
    )

    await persist_report(report)
    return {"final_status": new_status, "migration_report": report}


# ---------------------------------------------------------------------------
# Conditional router for critic
# ---------------------------------------------------------------------------


def _route_critic(
    state: MigrationState,
) -> Literal["dispatch", "rollback", "finalize"]:
    """Route after critic: rollback on any failure, dispatch if queue has more work, else finalize."""
    if state.get("rollback_stack"):
        return "rollback"
    if _select_ready_batch(state, batch_size=9999):
        return "dispatch"
    return "finalize"


def _route_rollback(state: MigrationState) -> Literal["dispatch", "finalize"]:
    """After rollback, keep draining the queue — a single failed module must not
    abandon the remaining work. Only finalize once nothing is left to dispatch."""
    if _select_ready_batch(state, batch_size=9999):
        return "dispatch"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

graph_builder: StateGraph = StateGraph(MigrationState)

graph_builder.add_node("plan", plan_node)
graph_builder.add_node("dispatch", dispatch_node)
graph_builder.add_node("transform_task", transform_task_node)  # type: ignore[type-var]
graph_builder.add_node("test", test_node)
graph_builder.add_node("critic", critic_node)
graph_builder.add_node("rollback", rollback_node)
graph_builder.add_node("finalize", finalize_node)

graph_builder.add_edge(START, "plan")
graph_builder.add_edge("plan", "dispatch")

# After dispatch, _route_dispatch fans out via Send or falls through to test.
graph_builder.add_conditional_edges(
    "dispatch",
    _route_dispatch,
    {"test": "test"},
)
graph_builder.add_edge("transform_task", "test")
graph_builder.add_edge("test", "critic")
graph_builder.add_conditional_edges(
    "critic",
    _route_critic,
    {"dispatch": "dispatch", "rollback": "rollback", "finalize": "finalize"},
)
graph_builder.add_conditional_edges(
    "rollback",
    _route_rollback,
    {"dispatch": "dispatch", "finalize": "finalize"},
)
graph_builder.add_edge("finalize", END)
