from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import hashlib
import json

from ..models import NewsItem, FilterResult
from ..utils.cache import LLMResponseCache


LLM_SYSTEM_PROMPT = (
    "You are an assistant that classifies if a news headline/summary is about the NFL. "
    "Respond with a short JSON: {\"is_relevant\": true|false, \"confidence\": 0..1, \"reason\": \"...\"}. "
    "Only use the title, optional description, and URL (domain/path hints)."
)


def _default_openai_client(timeout_s: float | None = None):
    """Create an OpenAI client only if an API key is configured; else None.

    This prevents crashes when running the demo without credentials and allows
    the fallback heuristic to be used instead.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        # Respect optional timeout (defaults applied by caller)
        if timeout_s is not None:
            return OpenAI(api_key=api_key, timeout=timeout_s)
        return OpenAI(api_key=api_key)
    except Exception:  # pragma: no cover
        return None


@dataclass
class LLMFilter:
    model: str = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    client: Optional[object] = None
    # Global timeout for LLM calls (seconds); can be overridden via OPENAI_TIMEOUT env
    timeout_s: float = float(os.environ.get("OPENAI_TIMEOUT", "10"))
    # Caching
    cache_ttl_s: int = int(os.environ.get("NEWS_PIPELINE_LLM_CACHE_TTL", "86400"))
    cache_path: Optional[str] = os.environ.get("NEWS_PIPELINE_LLM_CACHE_PATH") or None
    cache_enabled: bool = os.environ.get("NEWS_PIPELINE_LLM_CACHE", "1").lower() not in {"0", "false", "no"}

    def __post_init__(self):
        self._cache = LLMResponseCache(ttl_s=self.cache_ttl_s, sqlite_path=self.cache_path) if self.cache_enabled else None

    def _get_client(self):
        return self.client or _default_openai_client(self.timeout_s)

    def filter(self, item: NewsItem) -> FilterResult:
        prompt = (
            f"Title: {item.title}\n"
            f"Description: {item.description or '-'}\n"
            f"URL: {item.url}\n"
        )
        # Cache key based on model + compact JSON of inputs
        cache_key = None
        if self._cache is not None:
            try:
                payload = {
                    "m": self.model,
                    "t": item.title,
                    "d": item.description or "",
                    "u": item.url,
                }
                cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return FilterResult(
                        is_relevant=bool(cached.get("is_relevant", False)),
                        confidence_score=float(cached.get("confidence", 0.0)),
                        reasoning=str(cached.get("reason", "cache")),
                        method="llm-cache",
                    )
            except Exception:
                cache_key = None
        client = self._get_client()
        if client is None:
            # Fallback heuristic if no client (keeps tests deterministic)
            text = f"{item.title} {item.description or ''} {item.url}"
            is_rel = any(w in text.lower() for w in ["nfl", "nfl.com"])  # simple fallback
            return FilterResult(
                is_relevant=is_rel,
                confidence_score=0.55 if is_rel else 0.2,
                reasoning="fallback heuristic (no client)",
                method="llm",
            )

        # Minimal JSON-style instruction; callers should mock client in tests
        try:
            msg = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            # OpenAI Python v1 style
            # Try with timeout; if client doesn't accept it (e.g., tests' dummy client), retry without
            try:
                resp = client.chat.completions.create(model=self.model, messages=msg, timeout=self.timeout_s)
            except TypeError:
                resp = client.chat.completions.create(model=self.model, messages=msg)
            content = resp.choices[0].message.content.strip()
            # Very permissive parse; expect a JSON-like string
            # Note: use module-level json import to avoid shadowing earlier usage
            import re

            m = re.search(r"\{.*\}", content, re.S)
            data = json.loads(m.group(0)) if m else {}
            is_rel = bool(data.get("is_relevant", False))
            conf = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "")) or "llm"
            # Write-through cache
            if self._cache is not None and cache_key:
                try:
                    self._cache.set(cache_key, {"is_relevant": is_rel, "confidence": conf, "reason": reason})
                except Exception:
                    pass
            return FilterResult(
                is_relevant=is_rel,
                confidence_score=max(0.0, min(conf, 1.0)),
                reasoning=reason,
                method="llm",
            )
        except Exception:
            return FilterResult(
                is_relevant=False,
                confidence_score=0.0,
                reasoning="llm error",
                method="llm",
            )
