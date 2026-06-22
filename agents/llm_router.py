"""Central LLM client factory with Langfuse tracing and circuit-breaking fallback.

Every LLM call in the system must go through ``get_router().get_client()``.
Never instantiate LangChain chat models elsewhere.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from loguru import logger

from agents.tracing import make_callback_handler

Role = Literal["planner", "transform", "critic", "classifier"]

# Hard budget enforced at the router level. Callers must stay within these.
INPUT_TOKEN_LIMIT = 8_000
OUTPUT_TOKEN_LIMIT = 4_000

# Per-role output cap. Providers (notably Groq) bill the *reserved* max_tokens
# against rate limits, so roles that emit small structured replies must not
# reserve the full 4K. Planner/classifier return short JSON; transform/critic
# must emit whole corrected files, so they keep the full budget.
_ROLE_OUTPUT_TOKENS: dict[Role, int] = {
    "planner": 512,
    "classifier": 512,
    "transform": OUTPUT_TOKEN_LIMIT,
    "critic": OUTPUT_TOKEN_LIMIT,
}

# Role → (provider, model_id)
# NOTE: planner and critic temporarily routed to Groq — Gemini project access is
# currently denied (403) and OpenRouter free tier is rate-limited.
_ROLE_MODEL: dict[Role, tuple[str, str]] = {
    "planner": ("groq", "llama-3.3-70b-versatile"),
    "transform": ("openrouter", "qwen/qwen3-coder:free"),
    "critic": ("groq", "llama-3.3-70b-versatile"),
    "classifier": ("groq", "llama-3.3-70b-versatile"),
}

# Per-role fallback cascade — each entry is tried in order when the previous
# provider's circuit trips.  Must use a DIFFERENT provider than the primary.
# Groq roles: Groq → OpenRouter (free) → stop
# OpenRouter roles: OpenRouter → Groq → stop
# NOTE: qwen/qwen3-235b-a22b:free was discontinued (now 404 — "unavailable for
# free"). Replaced with meta-llama/llama-3.3-70b-instruct:free, which is live and
# returns proper 429s so the rate-limit/circuit logic works instead of hard-failing.
_ROLE_FALLBACK: dict[Role, list[tuple[str, str]]] = {
    "planner": [
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("gemini", "gemini-2.0-flash"),
    ],
    "transform": [
        ("groq", "llama-3.3-70b-versatile"),
        ("gemini", "gemini-2.0-flash"),
    ],
    "critic": [
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("gemini", "gemini-2.0-flash"),
    ],
    "classifier": [
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("gemini", "gemini-2.0-flash"),
    ],
}

_CIRCUIT_THRESHOLD = 3         # consecutive rate-limit errors before trip
_CIRCUIT_COOLDOWN_SECS = 300.0  # 5 minutes


@dataclass
class _CircuitBreaker:
    _errors: int = field(default=0, init=False, repr=False)
    _tripped_at: float | None = field(default=None, init=False, repr=False)

    def is_open(self) -> bool:
        if self._tripped_at is None:
            return False
        if time.monotonic() - self._tripped_at > _CIRCUIT_COOLDOWN_SECS:
            self._errors = 0
            self._tripped_at = None
            return False
        return True

    def record_rate_limit_error(self) -> None:
        self._errors += 1
        if self._errors >= _CIRCUIT_THRESHOLD:
            self._tripped_at = time.monotonic()
            logger.warning(
                "Circuit breaker tripped — provider will use fallback for {}s",
                _CIRCUIT_COOLDOWN_SECS,
            )

    def record_success(self) -> None:
        self._errors = 0


_breakers: dict[str, _CircuitBreaker] = defaultdict(lambda: _CircuitBreaker())


def _build_client(
    provider: str, model: str, max_output_tokens: int = OUTPUT_TOKEN_LIMIT
) -> BaseChatModel:
    kwargs: dict = {"max_tokens": max_output_tokens}
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.environ["GEMINI_API_KEY"],
            **kwargs,
        )
    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )
    if provider == "groq":
        return ChatGroq(
            model=model,
            groq_api_key=os.environ["GROQ_API_KEY"],
            **kwargs,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


class LLMRouter:
    """Resolves a role name to a (LangChain client, Langfuse callback) pair.

    Usage::

        client, callbacks = get_router().get_client("planner", session_id=run_id)
        response = await client.ainvoke(messages, config={"callbacks": callbacks})
        get_router().record_success("planner")
    """

    def get_client(
        self,
        role: Role,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[BaseChatModel, list[Any]]:
        """Return a configured chat client and its Langfuse callback handler.

        Args:
            role:       Which agent is calling (determines model selection).
            session_id: Migration run ID — groups all traces in Langfuse.
            run_id:     Deterministic Langfuse trace ID (use the run UUID).
            tags:       Extra labels for Langfuse filtering.
        """
        provider, model = _ROLE_MODEL[role]

        if _breakers[provider].is_open():
            # Walk the fallback cascade until we find an open provider.
            for fb_provider, fb_model in _ROLE_FALLBACK[role]:
                if not _breakers[fb_provider].is_open():
                    logger.warning(
                        "Circuit open for provider={}, routing role={} to fallback {}/{}",
                        provider, role, fb_provider, fb_model,
                    )
                    provider, model = fb_provider, fb_model
                    break
            else:
                # All fallbacks also tripped — use the last one and let it fail loudly.
                fb_provider, fb_model = _ROLE_FALLBACK[role][-1]
                logger.error(
                    "All providers tripped for role={}, forcing last fallback {}/{}",
                    role, fb_provider, fb_model,
                )
                provider, model = fb_provider, fb_model

        max_output = _ROLE_OUTPUT_TOKENS.get(role, OUTPUT_TOKEN_LIMIT)
        client = _build_client(provider, model, max_output)
        try:
            cb = make_callback_handler(
                session_id=session_id,
                trace_id=run_id,
                tags=(tags or []) + [f"role:{role}", f"provider:{provider}"],
            )
            callbacks: list[Any] = [cb]
        except Exception:
            # langchain not installed — tracing disabled, LLM calls still work
            callbacks = []
        return client, callbacks

    def record_rate_limit_error(self, role: Role) -> None:
        """Call when a 429 is returned for this role's provider."""
        provider, _ = _ROLE_MODEL[role]
        _breakers[provider].record_rate_limit_error()

    def record_success(self, role: Role) -> None:
        """Call after a successful LLM response to reset the error counter."""
        provider, _ = _ROLE_MODEL[role]
        _breakers[provider].record_success()


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    """Singleton LLMRouter — safe to call from any async context."""
    return LLMRouter()
