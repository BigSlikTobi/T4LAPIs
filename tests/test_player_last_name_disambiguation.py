from datetime import datetime, timezone

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


class DummyLLM:
    def __init__(self, chosen=None):
        self._chosen = chosen

    def extract_entities(self, article_text: str, max_retries: int = 3):
        # Return only the last name
        return {"players": ["Jackson"], "teams": []}

    def disambiguate_player(self, last_name: str, article_text: str, candidates):
        # Select Lamar by id or name depending on fixture
        return self._chosen


def test_last_name_disambiguation_via_llm_by_id():
    entity_dict = {
        "Lamar Jackson": "00-0033077",
        "DeSean Jackson": "00-0026169",
        "Kansas City Chiefs": "KC",
    }
    ex = EntitiesExtractor(entity_dict=entity_dict, llm=DummyLLM(chosen="00-0033077"), llm_enabled=True)
    it = make_item("Jackson shines in primetime win")
    ents = ex.extract(it).as_dict()
    assert "00-0033077" in ents["players"]
    assert "00-0026169" not in ents["players"]


def test_last_name_single_candidate_auto_resolves():
    entity_dict = {
        "Lamar Jackson": "00-0033077",
        # No other Jacksons in dict
    }
    class LLMLastNameOnly:
        def extract_entities(self, article_text: str, max_retries: int = 3):
            return {"players": ["Jackson"], "teams": []}

    ex = EntitiesExtractor(entity_dict=entity_dict, llm=LLMLastNameOnly(), llm_enabled=True)
    it = make_item("Jackson sets new rushing record")
    ents = ex.extract(it).as_dict()
    assert "00-0033077" in ents["players"]