"""AsyncPostgresSaver setup for LangGraph durable checkpointing.

Usage (FastAPI lifespan):
    async with checkpointer_context() as cp:
        graph = graph_builder.compile(checkpointer=cp)
        app.state.checkpointer = cp
        app.state.graph = graph
        yield
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger


def _dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


@asynccontextmanager
async def checkpointer_context() -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a ready AsyncPostgresSaver; tables are auto-created via setup()."""
    dsn = _dsn()
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
        logger.info("LangGraph Postgres checkpointer ready")
        yield saver
