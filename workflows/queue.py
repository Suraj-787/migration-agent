"""Global LLM concurrency limiter and Redis-backed priority TaskQueue.

The semaphore is the single source of truth for in-flight LLM call limits.
Every LLM call made through LLMRouter.ainvoke_with_retry acquires it before
proceeding, keeping the total across all agents to at most 5 concurrent calls.

TaskQueue wraps a Redis sorted set so tasks survive process restarts and can be
inspected externally.  ZPOPMIN makes pop() atomic — no two workers race for the
same task.
"""
from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from workflows.state import MigrationTask

# ---------------------------------------------------------------------------
# Global LLM concurrency semaphore
# ---------------------------------------------------------------------------

_llm_semaphore: asyncio.Semaphore | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the singleton Semaphore(5), lazy-initialised on the running loop."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(5)
    return _llm_semaphore


# ---------------------------------------------------------------------------
# Redis-backed TaskQueue
# ---------------------------------------------------------------------------


class TaskQueue:
    """Priority queue backed by a Redis sorted set.

    Key schema:
      migration_queue:{run_id}          — sorted set; score = topological depth
      migration_queue:{run_id}:done     — set of completed task_ids
      migration_queue:{run_id}:failed   — hash of task_id → failure reason

    Each method opens its own short-lived Redis connection, so the queue is
    safe to use concurrently from multiple coroutines without sharing state.
    """

    def __init__(self, run_id: str) -> None:
        self._key = f"migration_queue:{run_id}"
        self._done_key = f"migration_queue:{run_id}:done"
        self._failed_key = f"migration_queue:{run_id}:failed"

    def _client(self):  # type: ignore[return]
        from redis.asyncio import Redis  # type: ignore[import-untyped]

        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return Redis.from_url(url, decode_responses=True)

    async def push(self, task: MigrationTask) -> None:
        """Enqueue *task* with its topological priority as the sort key."""
        payload = task.model_dump_json()
        async with self._client() as r:
            await r.zadd(self._key, {payload: float(task.priority)})
        logger.debug("[queue] pushed task_id={} priority={}", task.task_id, task.priority)

    async def pop(self) -> MigrationTask | None:
        """Atomically pop the lowest-score (highest-priority) task via ZPOPMIN.

        ZPOPMIN is a single Redis command — no race between multiple workers.
        Returns None when the queue is empty.
        """
        from workflows.state import MigrationTask as _MT

        async with self._client() as r:
            result = await r.zpopmin(self._key, 1)
        if not result:
            return None
        member, _score = result[0]
        return _MT.model_validate_json(member)

    async def peek_all(self) -> list[MigrationTask]:
        """Return all queued tasks in priority order without removing them."""
        from workflows.state import MigrationTask as _MT

        async with self._client() as r:
            members = await r.zrange(self._key, 0, -1)
        return [_MT.model_validate_json(m) for m in members]

    async def mark_done(self, task_id: str) -> None:
        """Record *task_id* as successfully completed."""
        async with self._client() as r:
            await r.sadd(self._done_key, task_id)
        logger.debug("[queue] task_id={} marked done", task_id)

    async def mark_failed(self, task_id: str, reason: str) -> None:
        """Record *task_id* as failed with a human-readable *reason*."""
        async with self._client() as r:
            await r.hset(self._failed_key, task_id, reason)
        logger.debug("[queue] task_id={} marked failed: {}", task_id, reason)
