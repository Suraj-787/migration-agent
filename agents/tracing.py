"""Langfuse singleton client and per-request callback handler factory.

All instrumentation goes through here — never instantiate Langfuse or
CallbackHandler directly in other modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langfuse import Langfuse

if TYPE_CHECKING:
    from langfuse.langchain import CallbackHandler


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    """Singleton Langfuse client.

    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env.
    Silently disables tracing when keys are absent (safe for local dev before
    the Langfuse container is configured).
    """
    return Langfuse()


def make_callback_handler(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
    tags: list[str] | None = None,
) -> CallbackHandler:
    """Return a per-request Langfuse CallbackHandler for LangChain/LangGraph.

    Pass the returned handler in the LangChain ``config={"callbacks": [...]}``
    kwarg so every generation, span, and token count is captured automatically.

    Args:
        session_id: Groups all traces for one migration run.
        user_id:    Identifies who triggered the run (e.g. API caller).
        trace_id:   Deterministic trace ID — set to the migration run UUID so
                    Langfuse traces and Postgres run records share the same ID.
        tags:       Free-form labels surfaced in the Langfuse UI for filtering.
    """
    from langfuse.langchain import CallbackHandler as _CallbackHandler  # requires langchain

    return _CallbackHandler(
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        tags=tags or [],
    )


def trace_span(
    name: str,
    *,
    trace_id: str | None = None,
    span_type: str = "span",
    input: object = None,
    output: object = None,
    metadata: dict | None = None,
) -> None:
    """Record a one-shot span on the low-level Langfuse client (SDK v4+).

    Use this for non-LangChain operations (retrieval, reranking, git ops)
    where the CallbackHandler is not available.

    `span_type` maps to Langfuse `as_type`; use "retriever" for RAG spans.
    """
    lf = get_langfuse()
    obs = lf.start_observation(  # type: ignore[attr-defined]
        name=name,
        as_type=span_type,  # type: ignore[arg-type]
        input=input,
        output=output,
        metadata=metadata or {},
    )
    obs.end()  # type: ignore[union-attr]


def flush() -> None:
    """Flush pending Langfuse events. Wire into FastAPI lifespan shutdown."""
    get_langfuse().flush()
