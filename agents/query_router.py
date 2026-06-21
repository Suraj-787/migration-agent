"""LangGraph node that classifies an incoming query and selects Qdrant collections."""
from __future__ import annotations

import time
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from agents.llm_router import get_router
from rag.retriever import MetadataFilter

QueryIntent = Literal["code_pattern", "migration_guide", "api_reference", "combined"]

_COLLECTION_MAP: dict[str, list[str]] = {
    "code_pattern": ["code_chunks", "migration_patterns"],
    "migration_guide": ["doc_chunks"],
    "api_reference": ["doc_chunks", "code_chunks"],
    "combined": ["code_chunks", "doc_chunks", "migration_patterns"],
}

_SYSTEM_PROMPT = """\
You are a query classifier for a code migration assistant. Classify user queries into one of four intents:

- code_pattern: User wants to find concrete code examples, patterns, or implementation snippets.
  They're looking for HOW to write code, not conceptual documentation.
- migration_guide: User wants step-by-step migration instructions, changelogs, or upgrade guides.
  They're looking for WHAT changed between versions and HOW to upgrade.
- api_reference: User wants to understand a specific API, class, function, or module signature.
  They're looking for WHAT something is, its parameters, return type, or behavior.
- combined: The query spans multiple concerns (e.g., both code examples AND migration instructions),
  or is ambiguous enough that multiple collection types would help.

Also extract an optional language_hint (e.g., "python", "typescript") if the query explicitly
mentions a programming language. Return lowercase if set, null otherwise.

Examples:
Query: "show me how to define a FastAPI route with path parameters"
→ intent: code_pattern, language_hint: "python"

Query: "what changed in Flask 2.0 and how do I upgrade from 1.x"
→ intent: migration_guide, language_hint: "python"

Query: "what does app.include_router() do and what arguments does it accept"
→ intent: api_reference, language_hint: "python"

Query: "migrate my Flask app to FastAPI including rewriting route handlers with examples"
→ intent: combined, language_hint: "python"

Query: "how do I use useEffect in React 18 and what changed from React 16"
→ intent: combined, language_hint: null

Query: "TypeScript generic constraints syntax"
→ intent: code_pattern, language_hint: "typescript"

Query: "React 18 concurrent rendering migration guide"
→ intent: migration_guide, language_hint: null

Query: "asyncio.gather signature and return type"
→ intent: api_reference, language_hint: "python"

Now classify the following query.\
"""


class _ClassifierOutput(BaseModel):
    intent: QueryIntent
    language_hint: str | None = None
    reasoning: str


class RouterDecision(BaseModel):
    intent: QueryIntent
    collections: list[str]
    metadata_filter: MetadataFilter | None
    reasoning: str


class QueryRouterState(TypedDict):
    query: str
    router_decision: RouterDecision | None


async def classify_query(
    query: str,
    *,
    session_id: str | None = None,
    run_id: str | None = None,
) -> RouterDecision:
    """Classify a query and return which Qdrant collections to search.

    Calls Groq Llama 3.3 70B via the LLM router using structured output.
    Latency is typically under 800ms on Groq.
    """
    router = get_router()
    client, callbacks = router.get_client(
        "classifier",
        session_id=session_id,
        run_id=run_id,
        tags=["query_router"],
    )
    structured = client.with_structured_output(_ClassifierOutput)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    t0 = time.monotonic()
    result: _ClassifierOutput = cast(
        _ClassifierOutput,
        await structured.ainvoke(
            messages, config={"callbacks": callbacks}  # type: ignore[arg-type]
        ),
    )
    latency_ms = (time.monotonic() - t0) * 1000.0

    router.record_success("classifier")
    logger.debug(
        "Query routed: intent={} latency={:.0f}ms query={!r}",
        result.intent,
        latency_ms,
        query[:80],
    )

    mf: MetadataFilter | None = None
    if result.language_hint:
        mf = MetadataFilter(language=result.language_hint.lower())

    return RouterDecision(
        intent=result.intent,
        collections=_COLLECTION_MAP[result.intent],
        metadata_filter=mf,
        reasoning=result.reasoning,
    )


async def route_query(state: QueryRouterState) -> dict[str, Any]:
    """LangGraph node: read query from state, write RouterDecision back."""
    decision = await classify_query(state["query"])
    return {"router_decision": decision}
