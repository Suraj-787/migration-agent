"""POST /migrations — start a migration run.
GET  /migrations/{thread_id} — poll current state from checkpointer.
GET  /migrations/{thread_id}/stream — SSE stream of checkpoint updates.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rag.dep_graph import DepGraphBuilder, persist_graph
from workflows.state import MigrationSpec, MigrationState

router = APIRouter(tags=["migrations"])

_SSE_MAX_SECONDS = 1800  # 30-minute hard stop for an SSE stream
_SSE_POLL_INTERVAL = 2   # seconds between checkpointer reads


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TargetSpec(BaseModel):
    target_framework: str
    target_version: str
    custom_rules: list[str] = []


class StartMigrationRequest(BaseModel):
    repo_path: str
    target_spec: TargetSpec
    source_framework: str = "flask"
    source_version: str = "latest"


class MigrationStartResponse(BaseModel):
    thread_id: str
    status: Literal["started"]
    dep_graph_id: str


class MigrationStateResponse(BaseModel):
    thread_id: str
    repo_path: str
    final_status: Literal["pending", "success", "partial", "failed", "cost_ceiling_exceeded"]
    task_count: int
    result_count: int
    current_batch: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_state(
    req: StartMigrationRequest, dep_graph_id: uuid.UUID, thread_id: str
) -> MigrationState:
    return MigrationState(
        repo_path=req.repo_path,
        target_spec=MigrationSpec(
            source_framework=req.source_framework,
            target_framework=req.target_spec.target_framework,
            source_version=req.source_version,
            target_version=req.target_spec.target_version,
        ),
        dep_graph_id=dep_graph_id,
        task_queue=[],
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


def _state_from_checkpoint(
    thread_id: str, channel_values: dict[str, Any]
) -> MigrationStateResponse:
    return MigrationStateResponse(
        thread_id=thread_id,
        repo_path=channel_values.get("repo_path", ""),
        final_status=channel_values.get("final_status", "pending"),
        task_count=len(channel_values.get("task_queue", [])),
        result_count=len(channel_values.get("task_results", [])),
        current_batch=channel_values.get("current_batch", []),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/migrations", response_model=MigrationStartResponse, status_code=202)
async def start_migration(
    req: StartMigrationRequest, request: Request
) -> MigrationStartResponse:
    """Build the dependency graph, persist it, then kick off the migration graph."""
    if getattr(request.app.state, "shutdown_event", None) and request.app.state.shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down — no new migrations accepted")

    if not req.repo_path.strip():
        raise HTTPException(status_code=422, detail="repo_path must not be empty")

    engine = request.app.state.db_engine
    try:
        dep_graph = await asyncio.to_thread(
            lambda: DepGraphBuilder(req.repo_path).build()
        )
        async with AsyncSession(engine) as session:
            dep_graph_id = await persist_graph(dep_graph, session)
    except Exception as exc:
        logger.error("Dep graph build failed for repo={}: {}", req.repo_path, exc)
        raise HTTPException(
            status_code=422, detail=f"Dependency graph error: {exc}"
        ) from exc

    graph = request.app.state.graph
    thread_id = str(uuid.uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    initial = _initial_state(req, dep_graph_id, thread_id)

    async def _run() -> None:
        import contextlib

        from langfuse import get_client
        from langfuse.types import TraceContext

        lf = get_client()
        # OTel trace IDs must be 32-char lowercase hex; strip UUID dashes.
        otel_trace_id = thread_id.replace("-", "")
        try:
            obs_cm: contextlib.AbstractContextManager[object] = lf.start_as_current_observation(
                trace_context=TraceContext(trace_id=otel_trace_id),
                name="migration",
                as_type="agent",
                input={
                    "repo_path": req.repo_path,
                    "spec": (
                        f"{req.source_framework}→"
                        f"{req.target_spec.target_framework}"
                    ),
                },
            )
        except Exception as lf_exc:
            logger.warning("[tracing] Langfuse span setup failed (non-fatal): {}", lf_exc)
            obs_cm = contextlib.nullcontext()

        try:
            with obs_cm:
                await graph.ainvoke(initial, config=config)
            logger.info("Migration run complete — thread_id={}", thread_id)
        except Exception as exc:
            logger.error(
                "Migration run failed — thread_id={} error={}", thread_id, exc
            )

    task = asyncio.create_task(_run())
    active_tasks: set[asyncio.Task] = getattr(request.app.state, "active_tasks", set())  # type: ignore[type-arg]
    active_tasks.add(task)
    task.add_done_callback(active_tasks.discard)

    logger.info(
        "Migration started — thread_id={} dep_graph_id={} repo={} {}→{}",
        thread_id,
        dep_graph_id,
        req.repo_path,
        req.source_framework,
        req.target_spec.target_framework,
    )
    return MigrationStartResponse(
        thread_id=thread_id, status="started", dep_graph_id=str(dep_graph_id)
    )


@router.get("/migrations/{thread_id}", response_model=MigrationStateResponse)
async def get_migration(thread_id: str, request: Request) -> MigrationStateResponse:
    """Return the latest checkpointed state for a migration run."""
    checkpointer = request.app.state.checkpointer
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception as exc:
        logger.error(
            "Checkpointer read failed — thread_id={} error={}", thread_id, exc
        )
        raise HTTPException(status_code=502, detail=f"Checkpointer error: {exc}") from exc

    if checkpoint_tuple is None:
        raise HTTPException(
            status_code=404,
            detail=f"No migration found for thread_id={thread_id}",
        )

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    return _state_from_checkpoint(thread_id, channel_values)


@router.get("/migrations/{thread_id}/stream")
async def stream_migration(thread_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events stream that emits a checkpoint event every time the
    graph advances to a new step.  Polls the checkpointer every 2 seconds.

    SSE format: ``data: {json}\\n\\n``  (double newline is mandatory per spec).

    Event types:
    - ``waiting``    — migration not yet in checkpointer
    - ``checkpoint`` — graph advanced; includes node name, counts, status
    - ``cost``       — current accumulated estimated_cost_usd (emitted every 10 s)
    - ``done``       — terminal status reached (success / partial / failed)
    - ``timeout``    — 30-minute hard stop
    - ``error``      — checkpointer read failure
    """
    checkpointer = request.app.state.checkpointer
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    terminal_statuses = {"success", "partial", "failed", "cost_ceiling_exceeded"}

    async def _event_generator() -> AsyncIterator[str]:
        from workflows.cost import get_run_cost

        last_step = -1
        stream_start = time.monotonic()
        last_cost_event = time.monotonic() - 10.0  # fire immediately on first tick

        while True:
            if await request.is_disconnected():
                break

            if time.monotonic() - stream_start > _SSE_MAX_SECONDS:
                payload: dict[str, Any] = {"event": "timeout", "thread_id": thread_id}
                yield f"data: {json.dumps(payload)}\n\n"
                break

            now = time.monotonic()
            if now - last_cost_event >= 10.0:
                current_cost = await get_run_cost(thread_id)
                cost_payload: dict[str, Any] = {
                    "event": "cost",
                    "thread_id": thread_id,
                    "estimated_cost_usd": current_cost,
                }
                yield f"data: {json.dumps(cost_payload)}\n\n"
                last_cost_event = now

            try:
                checkpoint_tuple = await checkpointer.aget_tuple(config)
            except Exception as exc:
                logger.error(
                    "[stream] Checkpointer read failed thread_id={}: {}", thread_id, exc
                )
                payload = {"event": "error", "thread_id": thread_id, "detail": str(exc)}
                yield f"data: {json.dumps(payload)}\n\n"
                break

            if checkpoint_tuple is None:
                payload = {"event": "waiting", "thread_id": thread_id}
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(_SSE_POLL_INTERVAL)
                continue

            metadata = checkpoint_tuple.metadata or {}
            step: int = metadata.get("step", 0)

            if step > last_step:
                last_step = step
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})

                # Identify the last node that wrote to state.
                writes: dict[str, Any] = metadata.get("writes") or {}
                last_node: str | None = next(iter(writes), None)

                raw_results = channel_values.get("task_results") or []
                result_count = len(raw_results)
                succeeded = sum(
                    1
                    for r in raw_results
                    if (
                        r.get("status") if isinstance(r, dict) else getattr(r, "status", "")
                    )
                    in ("transformed", "success")
                )

                final_status: str = channel_values.get("final_status", "pending")
                payload = {
                    "event": "checkpoint",
                    "thread_id": thread_id,
                    "step": step,
                    "node": last_node,
                    "final_status": final_status,
                    "task_count": len(channel_values.get("task_queue") or []),
                    "result_count": result_count,
                    "succeeded": succeeded,
                    "rollback_count": len(channel_values.get("rollback_stack") or []),
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if final_status in terminal_statuses:
                    done_payload = {
                        "event": "done",
                        "thread_id": thread_id,
                        "final_status": final_status,
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    break

            await asyncio.sleep(_SSE_POLL_INTERVAL)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
