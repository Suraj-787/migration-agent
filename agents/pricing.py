"""Model pricing table and cost estimation for migration runs.

All rates are USD per 1,000 tokens (input, output).
PRICING_NOTE: reference rates for benchmarking; actual charges are $0 on
free tier for all currently-configured providers, but the ceiling logic is
exercised using these reference rates so it works correctly when paid models
are enabled.
"""
from __future__ import annotations

PRICING_NOTE = (
    "Reference rates for benchmarking. "
    "Actual charges are $0 on free tier for all configured providers."
)

COST_CEILING_USD: float = 5.0

# (input_per_1k_usd, output_per_1k_usd)
_PRICING: dict[str, tuple[float, float]] = {
    # Free-tier providers — always $0
    "groq/llama-3.3-70b-versatile": (0.0, 0.0),
    "openrouter/qwen/qwen3-coder:free": (0.0, 0.0),
    "openrouter/meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "voyage/voyage-code-3": (0.0, 0.0),
    # Paid reference rates (unused on free tier; exercised by ceiling logic)
    "gemini/gemini-2.5-flash": (0.00075, 0.003),
    "gemini/gemini-2.0-flash": (0.00010, 0.00040),
}


class CostCeilingExceeded(Exception):
    """Raised when a run's accumulated estimated cost exceeds COST_CEILING_USD."""


def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for one LLM call.

    Args:
        model_key:     "{provider}/{model}" e.g. "groq/llama-3.3-70b-versatile".
        input_tokens:  Prompt token count.
        output_tokens: Completion token count.
    """
    rate_in, rate_out = _PRICING.get(model_key, (0.0, 0.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000.0
