from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Protocol
import re
import os

from ..models import NewsItem


@dataclass
class ExtractedEntities:
    teams: Set[str] = field(default_factory=set)
    players: Set[str] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "teams": sorted(self.teams),
            "players": sorted(self.players),
            "topics": sorted(self.topics),
        }


class _LLMExtractorProto(Protocol):
    def extract_entities(self, article_text: str, max_retries: int = 3) -> Dict[str, Any]:
        ...


class EntitiesExtractor:
    def __init__(self, *, entity_dict: Optional[Dict[str, str]] = None, llm: Optional[_LLMExtractorProto] = None, llm_enabled: Optional[bool] = None) -> None:
        """
        entity_dict maps surface strings -> canonical id (player_id or team_abbr).
        If not provided, extractor will still tag topics and detect common team aliases.

        Optionally, provide an LLM client (llm.extract_entities) to enrich detection
        with model-extracted teams/players from title/description. LLM usage is gated
        by llm_enabled (defaults to env NEWS_PIPELINE_ENTITY_LLM=0).
        """
        self.entity_dict = entity_dict or {}
        self._entity_dict_lower = {k.lower(): v for k, v in self.entity_dict.items()}

        self.llm = llm
        if llm_enabled is None:
            # Enabled by default; set NEWS_PIPELINE_ENTITY_LLM=0 to turn off
            llm_enabled = os.getenv("NEWS_PIPELINE_ENTITY_LLM", "1") in ("1", "true", "True")
        self.llm_enabled = bool(llm_enabled)

        # Expanded topic taxonomy (simple regex hints)
        self.topic_keywords = {
            # Player availability / injuries
            "injury": re.compile(r"\b(acl|achilles|hamstring|ankle|concussion|injur(?:y|ed)|questionable|doubtful|out|day-?to-?day|illness)\b", re.I),
            "return": re.compile(r"\b(activated|return(?:s|ing)|cleared|designated to return|taken off ir|off pup)\b", re.I),
            "ir": re.compile(r"\b(injured reserve|\bIR\b|placed on ir|pup list|physically unable to perform)\b", re.I),

            # Transactions
            "signing": re.compile(r"\b(sign(?:s|ed|ing)|agrees? to terms|ink(?:s|ed)|one-?year deal|two-?year deal)\b", re.I),
            "release": re.compile(r"\b(release(?:s|d)|waive(?:s|d)|cut|part ways)\b", re.I),
            "trade": re.compile(r"\b(trade(?:s|d)?|acquire(?:s|d)?|deal for|sent to|in exchange for)\b", re.I),
            "contract": re.compile(r"\b(contract|extension|restructure|reworked|guarantee(?:s|d)?|salary cap)\b", re.I),

            # Team depth/role
            "depth_chart": re.compile(r"\b(starter|starting|backup|rb1|wr2|slot receiver|depth chart|snaps?|target share)\b", re.I),
            "roster_move": re.compile(r"\b(roster move|practice squad|elevate(?:s|d)?|promote(?:s|d)?|assign(?:s|ed)?|activate(?:s|d)?)\b", re.I),

            # Coaching / strategy
            "coaching": re.compile(r"\b(hire(?:s|d)|fire(?:s|d)|promote(?:s|d)|offensive coordinator|defensive coordinator|play-?caller|scheme)\b", re.I),

            # Events / meta
            "draft": re.compile(r"\b(nfl draft|mock draft|prospect|combine|pro day)\b", re.I),
            "rumor": re.compile(r"\b(rumor|report(?:s|ed)?|hearing|linked to)\b", re.I),
            "legal": re.compile(r"\b(arrest(?:ed)?|lawsuit|legal|charges?)\b", re.I),
            "retirement": re.compile(r"\b(retire(?:s|d|ment))\b", re.I),
            "record": re.compile(r"\b(franchise record|nfl record|career high|milestone)\b", re.I),
            "fantasy": re.compile(r"\b(fantasy|waiver wire|streamer|start(?:/|\s*or\s*)sit|dfs)\b", re.I),
        }

        # Common team alias normalization (synonyms -> team abbr)
        self.team_aliases: Dict[str, str] = self._default_team_aliases()

    def extract(self, item: NewsItem) -> ExtractedEntities:
        text = f"{item.title or ''} {item.description or ''}"
        ents = ExtractedEntities()

        # Topics
        for name, rx in self.topic_keywords.items():
            if rx.search(text):
                ents.topics.add(name)

        # Team aliases without requiring a dictionary
        lowered = text.lower()
        for alias, abbr in self.team_aliases.items():
            # word boundary match to avoid substrings
            if re.search(rf"\b{re.escape(alias)}\b", lowered, re.I):
                ents.teams.add(abbr)

        # Dictionary-driven mentions (safe token matching, case-insensitive)
        if self.entity_dict:
            # Sort keys longer-first to reduce substring shadowing
            for surface in sorted(self.entity_dict.keys(), key=len, reverse=True):
                s_norm = surface.strip()
                if not s_norm:
                    continue
                # Skip extremely short ambiguous surfaces (like 'NO', 'NE') to avoid noise
                if len(s_norm) <= 2 and s_norm.isalpha():
                    continue

                pat = self._surface_pattern(s_norm)
                if pat.search(text):
                    canonical = self.entity_dict[surface]
                    if self._is_team_abbr(canonical):
                        ents.teams.add(canonical)
                    else:
                        ents.players.add(canonical)

        # Optional: enrich with LLM results (validated against dictionary/aliases)
        if self.llm_enabled and self.llm:
            try:
                llm_entities = self.llm.extract_entities(text)
                players = llm_entities.get("players", []) or []
                teams = llm_entities.get("teams", []) or []

                # Normalize possible structured responses into list[str]
                def _names(seq):
                    out: List[str] = []
                    for item in seq:
                        if isinstance(item, dict) and "name" in item:
                            # Optional confidence acceptance if present
                            conf = item.get("confidence")
                            if conf is None or conf >= 0.5:
                                out.append(str(item["name"]))
                        elif isinstance(item, str):
                            out.append(item)
                    return out

                pl_names = _names(players)
                tm_names = _names(teams)

                # Validate/resolve names to canonical IDs using entity_dict and aliases
                for nm in pl_names:
                    key = nm.strip().lower()
                    if not key:
                        continue
                    if key in self._entity_dict_lower:
                        canonical = self._entity_dict_lower[key]
                        if not self._is_team_abbr(canonical):
                            ents.players.add(canonical)

                for nm in tm_names:
                    key = nm.strip().lower()
                    if not key:
                        continue
                    # direct dict lookup first
                    canon = self._entity_dict_lower.get(key)
                    if canon and self._is_team_abbr(canon):
                        ents.teams.add(canon)
                        continue
                    # fallback via alias table
                    alias_canon = self.team_aliases.get(key)
                    if alias_canon:
                        ents.teams.add(alias_canon)
            except Exception:
                # LLM is best-effort; ignore failures
                pass

        return ents

    def _is_team_abbr(self, canonical: str) -> bool:
        # Heuristic: 2-4 uppercase letters considered a team code
        return canonical.isupper() and 2 <= len(canonical) <= 4

    def _surface_pattern(self, surface: str) -> re.Pattern[str]:
        """Build a safe regex for a surface form to avoid substring hits.

        - For short alphas (<=4), ensure alpha boundaries using negative lookaround
          so 'IND' won't match 'kind'.
        - Otherwise, rely on \b boundaries.
        """
        if surface.isalpha() and len(surface) <= 4:
            # Negative lookaround for letters on both sides
            esc = re.escape(surface)
            return re.compile(rf"(?<![A-Za-z]){esc}(?![A-Za-z])", re.I)
        # Default: whole-token using word boundaries
        return re.compile(rf"\b{re.escape(surface)}\b", re.I)

    def _default_team_aliases(self) -> Dict[str, str]:
        """A lightweight alias map for NFL teams. Not exhaustive but practical.

        Maps common nicknames and location names to standard abbreviations.
        """
        aliases = {
            # NFC
            "49ers": "SF", "niners": "SF", "san francisco": "SF",
            "seahawks": "SEA", "seattle": "SEA",
            "rams": "LAR", "los angeles rams": "LAR",
            "cardinals": "ARI", "arizona": "ARI",
            "cowboys": "DAL", "dallas": "DAL",
            "eagles": "PHI", "philadelphia": "PHI",
            "giants": "NYG", "new york giants": "NYG",
            "commanders": "WAS", "washington": "WAS", "football team": "WAS",
            "lions": "DET", "detroit": "DET",
            "packers": "GB", "green bay": "GB",
            "vikings": "MIN", "minnesota": "MIN",
            "bears": "CHI", "chicago": "CHI",
            "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB",
            "falcons": "ATL", "atlanta": "ATL",
            "panthers": "CAR", "carolina": "CAR",
            "saints": "NO", "new orleans": "NO",

            # AFC
            "chiefs": "KC", "kansas city": "KC",
            "chargers": "LAC", "los angeles chargers": "LAC",
            "raiders": "LV", "las vegas": "LV",
            "broncos": "DEN", "denver": "DEN",
            "bills": "BUF", "buffalo": "BUF",
            "dolphins": "MIA", "miami": "MIA",
            "patriots": "NE", "pats": "NE", "new england": "NE",
            "jets": "NYJ", "new york jets": "NYJ",
            "ravens": "BAL", "baltimore": "BAL",
            "browns": "CLE", "cleveland": "CLE",
            "bengals": "CIN", "cincinnati": "CIN",
            "steelers": "PIT", "pittsburgh": "PIT",
            "jaguars": "JAX", "jags": "JAX", "jacksonville": "JAX",
            "colts": "IND", "indianapolis": "IND",
            "titans": "TEN", "tennessee": "TEN",
            "texans": "HOU", "houston": "HOU",
        }
        # Normalize keys to lowercase to simplify matching
        return {k.lower(): v for k, v in aliases.items()}
