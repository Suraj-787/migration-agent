"""LangGraph state schema for the migration orchestration graph."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict
from uuid import UUID

import operator
from pydantic import BaseModel


def _replace_list(_old: list, new: list) -> list:  # type: ignore[type-arg]
    """Reducer that overwrites instead of appends.

    Used for rollback_stack, which is a per-batch buffer: critic_node writes the
    failures from the current batch, rollback_node consumes them and clears it.
    An append-only reducer would make stale entries accumulate and re-trigger
    rollback on every loop iteration.
    """
    return new


class MigrationSpec(BaseModel):
    source_framework: str
    target_framework: str
    source_version: str
    target_version: str


class MigrationTask(BaseModel):
    task_id: str
    module_path: str  # absolute path
    description: str
    priority: int
    complexity: Literal["trivial", "standard", "complex"]
    predicted_changes: list[str]
    retrieved_context_ids: list[str]
    depends_on: list[str]  # task_ids of tasks this task depends on


class TaskResult(BaseModel):
    module_path: str
    status: Literal["success", "failed", "skipped", "transformed"]
    diff: str = ""
    error: str | None = None
    branch_name: str | None = None
    tokens_used: int = 0


class RollbackEntry(BaseModel):
    module_path: str
    original_content: str
    branch_name: str


class MigrationReport(BaseModel):
    thread_id: str
    total_tasks: int
    succeeded: int
    failed: int
    success_rate: float
    total_tokens_used: int
    estimated_cost_usd: float
    per_module_branch_names: dict[str, str]
    duration_seconds: float
    final_status: Literal["success", "partial", "failed"]


class MigrationState(TypedDict):
    repo_path: str
    target_spec: MigrationSpec
    dep_graph_id: UUID
    task_queue: list[MigrationTask]
    # Append-only lists use operator.add as reducer so nodes can push items.
    task_results: Annotated[list[TaskResult], operator.add]
    # Per-batch buffer (replace, not append): critic writes this batch's failures,
    # rollback consumes and clears them so the dispatch loop can continue.
    rollback_stack: Annotated[list[RollbackEntry], _replace_list]
    # Append-only ledgers of critic outcomes across all batches. critiqued_paths
    # scopes critic to only newly-transformed modules; passed_paths drives the
    # finalize success count (a module counts as succeeded only if critic passed).
    critiqued_paths: Annotated[list[str], operator.add]
    passed_paths: Annotated[list[str], operator.add]
    current_batch: list[str]
    attempt_count: dict[str, int]
    final_status: Literal["pending", "success", "partial", "failed"]
    # Set at invocation time; used by finalize_node for timing and report identity.
    thread_id: str
    started_at: float
    migration_report: MigrationReport | None
    # Captured by dispatch_node before the Send fan-out and threaded through
    # every Send payload.  asyncio.create_task() copies Python contextvars so
    # OTel context propagates automatically, but storing the explicit trace ID
    # gives transform_task_node a fallback to detect and diagnose any break.
    langfuse_trace_id: str | None
