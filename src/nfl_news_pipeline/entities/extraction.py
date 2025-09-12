from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re

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


class EntitiesExtractor:
    def __init__(self, *, entity_dict: Optional[Dict[str, str]] = None) -> None:
        """
        entity_dict maps surface strings -> canonical id (player_id or team_abbr).
        If not provided, extractor will still tag topics and detect common team aliases.
        """
        self.entity_dict = entity_dict or {}

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

        # Dictionary-driven mentions (exact, case-insensitive)
        if self.entity_dict:
            # Sort keys longer-first to reduce substring shadowing
            for surface in sorted(self.entity_dict.keys(), key=len, reverse=True):
                s_norm = surface.strip()
                if not s_norm:
                    continue
                # word boundary match when reasonable, else simple contains
                pat = re.compile(rf"\b{re.escape(s_norm)}\b", re.I)
                if pat.search(text) or s_norm.lower() in lowered:
                    canonical = self.entity_dict[surface]
                    # Heuristic: two-letter or three-letter uppercase => team abbr
                    if canonical.isupper() and 2 <= len(canonical) <= 4:
                        ents.teams.add(canonical)
                    else:
                        ents.players.add(canonical)

        return ents

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
