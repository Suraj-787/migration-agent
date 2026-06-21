"""Central LLM client factory with Langfuse tracing and circuit-breaking fallback.

Every LLM call in the system must go through ``get_router().get_client()``.
Never instantiate LangChain chat models elsewhere.
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from loguru import logger

from agents.tracing import make_callback_handler

Role = Literal["planner", "transform", "critic", "classifier"]

# Hard budget enforced at the router level. Callers must stay within these.
INPUT_TOKEN_LIMIT = 8_000
OUTPUT_TOKEN_LIMIT = 4_000

# Role → (provider, model_id)
_ROLE_MODEL: dict[Role, tuple[str, str]] = {
    "planner": ("gemini", "gemini-2.5-flash"),
    "transform": ("openrouter", "qwen/qwen3-coder:free"),
    "critic": ("gemini", "gemini-2.5-flash"),
    "classifier": ("groq", "llama-3.3-70b-versatile"),
}

# Fallback when a primary provider circuit trips
_FALLBACK_PROVIDER = "openrouter"
_FALLBACK_MODEL = "qwen/qwen3-coder:free"

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


def _build_client(provider: str, model: str) -> BaseChatModel:
    kwargs: dict = {"max_tokens": OUTPUT_TOKEN_LIMIT}
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
    ) -> tuple[BaseChatModel, list[CallbackHandler]]:
        """Return a configured chat client and its Langfuse callback handler.

        Args:
            role:       Which agent is calling (determines model selection).
            session_id: Migration run ID — groups all traces in Langfuse.
            run_id:     Deterministic Langfuse trace ID (use the run UUID).
            tags:       Extra labels for Langfuse filtering.
        """
        provider, model = _ROLE_MODEL[role]

        if _breakers[provider].is_open():
            logger.warning(
                "Circuit open for provider={}, routing role={} to fallback {}",
                provider,
                role,
                _FALLBACK_MODEL,
            )
            provider, model = _FALLBACK_PROVIDER, _FALLBACK_MODEL

        client = _build_client(provider, model)
        cb = make_callback_handler(
            session_id=session_id,
            trace_id=run_id,
            tags=(tags or []) + [f"role:{role}", f"provider:{provider}"],
        )
        return client, [cb]

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
