"""LLM list prices used for COST ESTIMATES (Phase 6+).

Source: Groq models documentation, https://console.groq.com/docs/models, "Production Models" table,
row ``openai/gpt-oss-120b``: "$0.15 input" / "$0.60 output" per 1M tokens; retrieved 2026-11-26.
The figures were read through an automated page extraction, so check them against the live page
before relying on them. https://groq.com/pricing showed no price table when fetched.

These are estimates of what the same token usage would cost at the published on-demand price.
They are NOT the actual charge: this project uses the user's own Groq account plan (free /
developer tier), whose actual billed amount must be read from the Groq console.
For reasoning models (gpt-oss), ``completion_tokens`` reported by the API include the hidden
reasoning tokens, which are billed as output.
"""

from __future__ import annotations

PRICING_USD_PER_MILLION = {
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.60,
        "source": "https://console.groq.com/docs/models",
        "retrieved": "2026-11-26",
        "status": "published on-demand list price; verify against the live page",
    },
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimated USD cost at the published list price; None if the model has no recorded price."""
    price = PRICING_USD_PER_MILLION.get(model)
    if price is None:
        return None
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000
