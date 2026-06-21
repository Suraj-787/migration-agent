"""Integration tests for agents/query_router.py.

Requires GROQ_API_KEY to be set. All tests are skipped when the key is absent.
Latency assertions are generous (< 800ms) to account for Groq's typical p99.
"""
from __future__ import annotations

import os
import time

import pytest

from agents.query_router import QueryRouterState, RouterDecision, classify_query, route_query

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping Groq integration tests",
)


# ---------------------------------------------------------------------------
# Parametrized classification tests (10 queries)
# ---------------------------------------------------------------------------

_CASES: list[tuple[str, str, list[str]]] = [
    # (query, expected_intent, expected_collections)
    # --- code_pattern (3) ---
    (
        "show me how to define a FastAPI route with path parameters and a response model",
        "code_pattern",
        ["code_chunks", "migration_patterns"],
    ),
    (
        "Python asyncio.gather example with error handling",
        "code_pattern",
        ["code_chunks", "migration_patterns"],
    ),
    (
        "TypeScript generic constraint syntax for narrowing union types",
        "code_pattern",
        ["code_chunks", "migration_patterns"],
    ),
    # --- migration_guide (2) ---
    (
        "what changed between Flask 1.x and Flask 2.0 and how do I upgrade",
        "migration_guide",
        ["doc_chunks"],
    ),
    (
        "React 16 to React 18 migration guide for concurrent rendering",
        "migration_guide",
        ["doc_chunks"],
    ),
    # --- api_reference (3) ---
    (
        "what does app.include_router() do and what arguments does it accept",
        "api_reference",
        ["doc_chunks", "code_chunks"],
    ),
    (
        "asyncio.gather signature, return type, and exception behaviour",
        "api_reference",
        ["doc_chunks", "code_chunks"],
    ),
    (
        "Pydantic v2 model_validator decorator parameters and execution order",
        "api_reference",
        ["doc_chunks", "code_chunks"],
    ),
    # --- combined (2) ---
    (
        "migrate my Flask app to FastAPI including rewriting route handlers with code examples",
        "combined",
        ["code_chunks", "doc_chunks", "migration_patterns"],
    ),
    (
        "Python 2 to Python 3 print statement migration with before and after examples",
        "combined",
        ["code_chunks", "doc_chunks", "migration_patterns"],
    ),
]


@pytest.mark.parametrize("query,expected_intent,expected_collections", _CASES)
async def test_classify_query(
    query: str, expected_intent: str, expected_collections: list[str]
) -> None:
    t0 = time.monotonic()
    decision = await classify_query(query)
    latency_ms = (time.monotonic() - t0) * 1000.0

    assert isinstance(decision, RouterDecision)
    assert decision.intent == expected_intent, (
        f"Expected intent={expected_intent!r} but got {decision.intent!r}. "
        f"Reasoning: {decision.reasoning}"
    )
    assert sorted(decision.collections) == sorted(expected_collections), (
        f"Expected collections={expected_collections} but got {decision.collections}"
    )
    assert latency_ms < 800, f"Latency {latency_ms:.0f}ms exceeded 800ms budget"


# ---------------------------------------------------------------------------
# LangGraph node smoke test
# ---------------------------------------------------------------------------

async def test_route_query_node() -> None:
    state: QueryRouterState = {
        "query": "how do I use pydantic BaseModel with FastAPI request bodies",
        "router_decision": None,
    }
    update = await route_query(state)

    assert "router_decision" in update
    decision = update["router_decision"]
    assert isinstance(decision, RouterDecision)
    assert decision.intent in {"code_pattern", "migration_guide", "api_reference", "combined"}
    assert len(decision.collections) > 0
    assert decision.reasoning
