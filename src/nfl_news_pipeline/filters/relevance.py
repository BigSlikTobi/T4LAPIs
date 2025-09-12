from __future__ import annotations

import os
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

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
    allow_llm: bool | None = None,
) -> Tuple[FilterResult, str]:
    """Filter a news item with rule-based first, then always LLM as a quality gate.

    If the LLM call fails or exceeds a hard timeout, fall back to the rule-based result.
    Returns (FilterResult, stage) where stage is 'llm' when LLM result is used, otherwise 'rule'.
    """
    rule = rule or RuleBasedFilter()
    llm = llm or LLMFilter()
    # Always use LLM per quality gate; flag retained for backward-compat but ignored
    allow_llm = True

    rb = rule.filter(item)

    # Only gate items that would otherwise be included
    if not rb.is_relevant:
        return rb, "rule"

    # Hard timeout for LLM call; default to OPENAI_TIMEOUT or 10s, override via NEWS_PIPELINE_LLM_HARD_TIMEOUT
    try:
        hard_timeout_s = float(os.environ.get("NEWS_PIPELINE_LLM_HARD_TIMEOUT", os.environ.get("OPENAI_TIMEOUT", "10")))
    except ValueError:
        hard_timeout_s = 10.0

    # Run LLM call in a separate thread to enforce a hard cap; fall back to rule on timeout or error
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(llm.filter, item)
        lf = fut.result(timeout=hard_timeout_s)
        return lf, "llm"
    except FuturesTimeout:
        # Do not block waiting for the worker thread
        ex.shutdown(wait=False, cancel_futures=True)
        return rb, "rule"
    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)
        return rb, "rule"
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
