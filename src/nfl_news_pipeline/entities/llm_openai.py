from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


class OpenAIEntityExtractor:
    """Small OpenAI-backed extractor for players/teams from metadata text.

    Uses chat.completions to request a strict JSON with lists 'players' and 'teams'.
    Defaults model to OPENAI_ENTITY_MODEL or OPENAI_MODEL or 'gpt-5-nano'.
    """

    def __init__(self, client: Optional[Any] = None, *, model: Optional[str] = None, timeout_s: Optional[float] = None) -> None:
        self.model = model or os.environ.get("OPENAI_ENTITY_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-5-nano")
        self.timeout_s = float(timeout_s) if timeout_s is not None else float(os.environ.get("OPENAI_TIMEOUT", "10"))
        self.client = client or self._default_openai_client(self.timeout_s)

    def _default_openai_client(self, timeout_s: float | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
            if timeout_s is not None:
                return OpenAI(api_key=api_key, timeout=timeout_s)
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def extract_entities(self, article_text: str, max_retries: int = 2) -> Dict[str, Any]:
        if not article_text or not article_text.strip() or not self.client:
            return {"players": [], "teams": []}

        system = (
            "You are an expert NFL analyst. Extract only NFL player and team names from the text.\n"
            "Respond with ONLY JSON: {\"players\": [...], \"teams\": [...]} where elements are strings or objects {name, confidence}."
        )
        user = f"Article:\n'''{article_text.strip()}'''\n\nJSON Output:"

        for _ in range(max_retries):
            try:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=600,
                        temperature=0.1,
                        timeout=self.timeout_s,
                    )
                except TypeError:
                    # Some mock clients don't accept timeout
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=600,
                        temperature=0.1,
                    )

                content = (resp.choices[0].message.content or "").strip()
                m = re.search(r"\{.*\}", content, re.S)
                data = json.loads(m.group(0)) if m else {}

                players = data.get("players", [])
                teams = data.get("teams", [])
                if isinstance(players, list) and isinstance(teams, list):
                    return {"players": players, "teams": teams}
            except Exception:
                # Try again or fall through
                pass

        return {"players": [], "teams": []}
