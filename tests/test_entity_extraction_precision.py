from datetime import datetime, timezone
from typing import Any, Dict

import types

from src.nfl_news_pipeline.entities import EntitiesExtractor
from src.nfl_news_pipeline.models import NewsItem


def make_item(title: str, desc: str = ""):
    return NewsItem(
        url="https://example.com/x",
        title=title,
        publication_date=datetime.now(timezone.utc),
        source_name="test",
        publisher="pub",
        description=desc,
        raw_metadata={},
    )


def test_team_abbr_not_matched_as_substring():
    # 'IND' should not be matched inside 'kind'
    entity_dict = {"Indianapolis Colts": "IND"}
    ex = EntitiesExtractor(entity_dict=entity_dict)

    it = make_item("If DK Metcalf is feeling some kind of way ...")
    ents = ex.extract(it).as_dict()

    assert "IND" not in ents["teams"], "IND must not match inside 'kind'"


def test_llm_enrichment_validated_against_dictionary_and_aliases(monkeypatch):
    # Provide a tiny dictionary and ensure LLM names map to canonical
    entity_dict = {
        "Patrick Mahomes": "00-0033873",
        "Kansas City Chiefs": "KC",
        "KC": "KC",
    }

    class DummyLLM:
        def extract_entities(self, article_text: str, max_retries: int = 3):
            # Mixed structured and plain outputs, with a low-confidence entry excluded
            return {
                "players": [
                    {"name": "Patrick Mahomes", "confidence": 0.9},
                    {"name": "Somebody Else", "confidence": 0.2},
                ],
                "teams": [
                    {"name": "Kansas City Chiefs", "confidence": 0.95},
                    "KC",
                ],
            }

    ex = EntitiesExtractor(entity_dict=entity_dict, llm=DummyLLM(), llm_enabled=True)

    it = make_item("Mahomes leads Chiefs to victory")
    ents = ex.extract(it).as_dict()

    assert "00-0033873" in ents["players"]
    assert "KC" in ents["teams"]
    # Low-confidence 'Somebody Else' should be ignored
    assert len(ents["players"]) == 1
