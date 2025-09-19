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

        is_gpt5 = self.model.lower().startswith("gpt-5")
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"

        for _ in range(max_retries):
            try:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=1.0 if is_gpt5 else 0.1,
                        timeout=self.timeout_s,
                        **{token_param: 600},
                    )
                except TypeError:
                    # Some mock clients don't accept timeout
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=1.0 if is_gpt5 else 0.1,
                        **{token_param: 600},
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

    def disambiguate_player(self, last_name: str, article_text: str, candidates: list[dict], max_retries: int = 2) -> Optional[str]:
        """Given a last name and candidate list [{name, id}], return the best-matching player's id or full name.

        Returns None on failure. Prefers returning the canonical id when possible.
        """
        if not self.client or not candidates:
            return None
        system = (
            "You are an NFL entity linker. Given a short article and a list of candidate players with the same last name, "
            "pick the one referenced by the article. Respond with ONLY JSON: {\"id\": <id-or-blank>, \"name\": <full-name-or-blank>}"
        )
        cand_json = json.dumps(candidates, ensure_ascii=False)
        user = (
            f"Last name: {last_name}\n"
            f"Candidates: {cand_json}\n"
            f"Article:\n'''{article_text.strip()}'''\n\n"
            "Return strictly the JSON payload."
        )
        for _ in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=1.0 if is_gpt5 else 0.1,
                    timeout=self.timeout_s,
                    **{token_param: 200},
                )
                content = (resp.choices[0].message.content or "").strip()
                m = re.search(r"\{.*\}", content, re.S)
                data = json.loads(m.group(0)) if m else {}
                pid = (data.get("id") or "").strip()
                pname = (data.get("name") or "").strip()
                if pid:
                    return pid
                if pname:
                    return pname
            except Exception:
                continue
        return None
