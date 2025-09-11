from datetime import datetime, timezone

import pytest

from src.nfl_news_pipeline.models import NewsItem
from src.nfl_news_pipeline.filters.rule_based import RuleBasedFilter
from src.nfl_news_pipeline.filters.llm import LLMFilter
from src.nfl_news_pipeline.filters.relevance import filter_item


def make_item(title: str, url: str = "https://example.com/") -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        description=None,
        publication_date=datetime.now(timezone.utc),
        source_name="test",
        publisher="testpub",
    )


def test_rule_based_filter_positive():
    rb = RuleBasedFilter()
    item = make_item("NFL news: 49ers sign CB")
    res = rb.filter(item)
    assert res.is_relevant is True
    assert res.confidence_score >= 0.4


def test_rule_based_filter_negative():
    rb = RuleBasedFilter()
    item = make_item("Premier League result: Arsenal win")
    res = rb.filter(item)
    assert res.is_relevant is False or res.confidence_score < 0.4


class DummyLLM:
    def __init__(self, relevant: bool, conf: float = 0.7):
        self.relevant = relevant
        self.conf = conf

    class chat:
        class completions:
            @staticmethod
            def create(model, messages):  # type: ignore
                class Msg:
                    class Choice:
                        class Message:
                            content = '{"is_relevant": true, "confidence": 0.75, "reason": "mentions nfl"}'

                        message = Message()

                    choices = [Choice()]

                return Msg()


def test_llm_filter_mocked_positive():
    item = make_item("Is this about NFL?", url="https://www.nfl.com/news/foo")
    lf = LLMFilter(client=DummyLLM(True))
    res = lf.filter(item)
    assert res.is_relevant is True
    assert 0.5 <= res.confidence_score <= 1.0


def test_relevance_orchestrator_uses_llm_when_ambiguous():
    # Craft an ambiguous case with low rule-based score
    item = make_item("Football update: preseason thoughts")
    rb = RuleBasedFilter(team_weight=0.0, keyword_weight=0.2, url_weight=0.0)
    lf = LLMFilter(client=DummyLLM(True))
    res, stage = filter_item(item, rule=rb, llm=lf)
    assert stage == "llm"
    assert res.is_relevant is True
