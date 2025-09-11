from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from ..models import NewsItem, FilterResult


NFL_TEAMS = [
    # NFC
    "49ers", "Cardinals", "Rams", "Seahawks",
    "Cowboys", "Giants", "Eagles", "Commanders",
    "Bears", "Lions", "Packers", "Vikings",
    "Buccaneers", "Falcons", "Panthers", "Saints",
    # AFC
    "Ravens", "Bengals", "Browns", "Steelers",
    "Bills", "Dolphins", "Jets", "Patriots",
    "Texans", "Colts", "Jaguars", "Titans",
    "Broncos", "Chargers", "Chiefs", "Raiders",
]

NFL_KEYWORDS = [
    "NFL", "Super Bowl", "Week ", "touchdown", "quarterback", "wide receiver",
    "running back", "linebacker", "cornerback", "head coach", "preseason", "regular season",
]

URL_PATTERNS = [
    r"/nfl/",
    r"^https?://(www\.)?nfl\.com/",
]


def _compile_keywords(words: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]


TEAM_PATTERNS = _compile_keywords(NFL_TEAMS)
KEYWORD_PATTERNS = _compile_keywords(NFL_KEYWORDS)
URL_REGEXES = [re.compile(p, re.IGNORECASE) for p in URL_PATTERNS]


@dataclass
class RuleBasedFilter:
    team_weight: float = 0.6
    keyword_weight: float = 0.3
    url_weight: float = 0.2

    def score(self, item: NewsItem) -> Tuple[float, List[str]]:
        text = f"{item.title or ''} {item.description or ''}"
        reasons: List[str] = []
        score = 0.0

        if any(p.search(text) for p in TEAM_PATTERNS):
            score += self.team_weight
            reasons.append("team match")

        if any(p.search(text) for p in KEYWORD_PATTERNS):
            score += self.keyword_weight
            reasons.append("keyword match")

        if any(rx.search(item.url) for rx in URL_REGEXES):
            score += self.url_weight
            reasons.append("url pattern")

        score = min(score, 1.0)
        return score, reasons

    def filter(self, item: NewsItem, threshold: float = 0.4) -> FilterResult:
        score, reasons = self.score(item)
        is_rel = score >= threshold
        return FilterResult(
            is_relevant=is_rel,
            confidence_score=score,
            reasoning=", ".join(reasons) if reasons else "no nfl signals",
            method="rule_based",
        )
