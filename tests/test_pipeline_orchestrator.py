from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from src.nfl_news_pipeline.orchestrator import NFLNewsPipeline
from src.nfl_news_pipeline.models import NewsItem, FeedConfig, DefaultsConfig, PipelineConfig


class FakeConfigManager:
    def __init__(self, feeds: List[FeedConfig]):
        self._feeds = feeds
        self._defaults = DefaultsConfig()

    def load_config(self):
        return None

    def get_defaults(self):
        return self._defaults

    def get_enabled_sources(self):
        return self._feeds


class FakeRSS:
    def __init__(self, items: List[NewsItem]):
        self.items = items

    def fetch_multiple(self, feeds: List[FeedConfig]):
        return self.items


class FakeSitemap:
    def __init__(self, items: Dict[str, List[NewsItem]]):
        self.items = items

    def fetch_sitemap(self, feed: FeedConfig):
        return self.items.get(feed.name, [])


class FakeStorage:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.watermarks: Dict[str, datetime] = {}

    def check_duplicate_urls(self, urls):
        return {}

    class _Resp:
        def __init__(self, inserted, updated):
            self.inserted_count = inserted
            self.updated_count = updated
            self.errors_count = 0
            self.ids_by_url = {}

    def store_news_items(self, items):
        self.rows.extend(items)
        return FakeStorage._Resp(len(items), 0)

    def get_watermark(self, source_name: str):
        return self.watermarks.get(source_name)

    def update_watermark(self, source_name: str, *, last_processed_date: datetime, **kwargs):
        self.watermarks[source_name] = last_processed_date
        return True


def make_item(name: str, url: str, title: str, dt: datetime):
    return NewsItem(
        url=url,
        title=title,
        publication_date=dt,
        source_name=name,
        publisher="pub",
        description=None,
        raw_metadata={},
    )


def test_pipeline_orchestrator_basic_monolithic(monkeypatch):
    # Ensure deterministic behavior: run in rule-only mode and avoid remote entity dictionary
    monkeypatch.setenv("NEWS_PIPELINE_DISABLE_LLM", "1")
    monkeypatch.setenv("NEWS_PIPELINE_DISABLE_ENTITY_DICT", "1")
    monkeypatch.setenv("NEWS_PIPELINE_DISABLE_ENTITY_LLM", "1")
    # Two sources: one RSS with two items, one sitemap with one recent item
    now = datetime.now(timezone.utc)
    feeds = [
        FeedConfig(name="feed_rss", type="rss", publisher="p1"),
        FeedConfig(name="feed_map", type="sitemap", publisher="p2"),
    ]
    rss_items = [
        make_item("feed_rss", "https://a/1", "NFL Something", now - timedelta(minutes=30)),
        make_item("feed_rss", "https://a/2", "Other Topic", now - timedelta(minutes=10)),
    ]
    site_items = {
        "feed_map": [make_item("feed_map", "https://b/1", "NFL Roundup", now - timedelta(minutes=5))]
    }

    # Force our fakes into the pipeline by monkeypatching the modules the class imports
    import src.nfl_news_pipeline.orchestrator.pipeline as pl

    cm = FakeConfigManager(feeds)
    st = FakeStorage()
    p = NFLNewsPipeline("/dev/null", storage=st, audit=None)
    pl.ConfigManager = lambda _: cm
    pl.RSSProcessor = lambda defaults: FakeRSS(rss_items)
    pl.SitemapProcessor = lambda defaults: FakeSitemap(site_items)

    summary = p.run()
    # Filter rules: only items with NFL in title should pass by default rule-based filter
    # Expect 2 kept (NFL Something, NFL Roundup)
    assert summary.filtered_in >= 1
    assert summary.fetched_items == 3
    assert summary.inserted >= 1
    # Watermark updated
    assert st.watermarks.get("feed_rss") is not None or st.watermarks.get("feed_map") is not None
