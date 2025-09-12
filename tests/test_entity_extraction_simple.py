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


def test_entities_extractor_simple_dict_and_topics():
    entity_dict = {
        "Patrick Mahomes": "00-0033873",
        "KC": "KC",
        "Kansas City Chiefs": "KC",
        "Chiefs": "KC",
    }
    ex = EntitiesExtractor(entity_dict=entity_dict)

    it = make_item("Chiefs trade for star; Patrick Mahomes injury scare")
    ents = ex.extract(it).as_dict()

    assert "KC" in ents["teams"]
    assert "00-0033873" in ents["players"]
    # Has both trade and injury topics
    assert "trade" in ents["topics"]
    assert "injury" in ents["topics"]
