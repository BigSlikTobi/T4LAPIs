from __future__ import annotations

from typing import Tuple

from ..models import NewsItem, FilterResult
from .rule_based import RuleBasedFilter
from .llm import LLMFilter


AMBIGUOUS_LOW = 0.2
AMBIGUOUS_HIGH = 0.6


def filter_item(
    item: NewsItem,
    *,
    rule: RuleBasedFilter | None = None,
    llm: LLMFilter | None = None,
) -> Tuple[FilterResult, str]:
    """Filter a news item using rule-based first, then LLM if ambiguous.

    Returns (FilterResult, stage) where stage is 'rule' or 'llm'.
    """
    rule = rule or RuleBasedFilter()
    llm = llm or LLMFilter()

    rb = rule.filter(item)
    # Treat exact boundary values as ambiguous to allow LLM assist.
    if rb.confidence_score < AMBIGUOUS_LOW or rb.confidence_score > AMBIGUOUS_HIGH:
        return rb, "rule"

    # Ambiguous range: ask LLM
    lf = llm.filter(item)
    return lf, "llm"
