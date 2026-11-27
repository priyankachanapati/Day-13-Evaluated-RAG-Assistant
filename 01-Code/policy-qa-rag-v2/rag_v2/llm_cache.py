"""Response cache around an LLM client - avoids repeating identical calls.

The cache key is SHA-256 over (model, temperature, max_tokens, system prompt, user prompt).
If two configurations give the LLM exactly the same prompt (e.g. the same four context
chunks), the second one reuses the saved response instead of spending quota. Each completion
exposes ``last_usage``: token counts and whether the response came from the cache (cached
responses are excluded from LLM latency statistics by the callers).

Nothing secret is stored: entries contain prompts, responses and token counts only.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class CachingLLM:
    def __init__(self, inner, cache_dir: Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_usage: dict = {}
        self.real_calls = 0
        self.cache_hits = 0

    def key(self, system_prompt: str, user_prompt: str) -> str:
        material = json.dumps(
            [
                self.model,
                getattr(self.inner, "temperature", None),
                getattr(self.inner, "max_tokens", None),
                system_prompt,
                user_prompt,
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        path = self.cache_dir / f"{self.key(system_prompt, user_prompt)}.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            self.last_usage = {
                "cached": True,
                "prompt_tokens": entry["prompt_tokens"],
                "completion_tokens": entry["completion_tokens"],
                "latency_ms": None,
                "original_latency_ms": entry["latency_ms"],
            }
            return entry["response"]
        before = (getattr(self.inner, "prompt_tokens", 0), getattr(self.inner, "completion_tokens", 0))
        t0 = time.perf_counter()
        response = self.inner.complete(system_prompt, user_prompt)
        latency = (time.perf_counter() - t0) * 1000
        usage = {
            "prompt_tokens": getattr(self.inner, "prompt_tokens", 0) - before[0],
            "completion_tokens": getattr(self.inner, "completion_tokens", 0) - before[1],
        }
        self.real_calls += 1
        path.write_text(
            json.dumps(
                {"model": self.model, "response": response, "latency_ms": latency, **usage},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.last_usage = {"cached": False, "latency_ms": latency, **usage}
        return response
