from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..models import NewsItem, FilterResult


LLM_SYSTEM_PROMPT = (
    "You are an assistant that classifies if a news headline/summary is about the NFL. "
    "Respond with a short JSON: {\"is_relevant\": true|false, \"confidence\": 0..1, \"reason\": \"...\"}. "
    "Only use the title, optional description, and URL (domain/path hints)."
)


def _default_openai_client():
    """Create an OpenAI client only if an API key is configured; else None.

    This prevents crashes when running the demo without credentials and allows
    the fallback heuristic to be used instead.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key)
    except Exception:  # pragma: no cover
        return None


@dataclass
class LLMFilter:
    model: str = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    client: Optional[object] = None

    def _get_client(self):
        return self.client or _default_openai_client()

    def filter(self, item: NewsItem) -> FilterResult:
        prompt = (
            f"Title: {item.title}\n"
            f"Description: {item.description or '-'}\n"
            f"URL: {item.url}\n"
        )
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
            resp = client.chat.completions.create(model=self.model, messages=msg)
            content = resp.choices[0].message.content.strip()
            # Very permissive parse; expect a JSON-like string
            import json
            import re

            m = re.search(r"\{.*\}", content, re.S)
            data = json.loads(m.group(0)) if m else {}
            is_rel = bool(data.get("is_relevant", False))
            conf = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "")) or "llm"
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
