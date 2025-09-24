from datetime import datetime, timedelta

import pytest

from src.nfl_news_pipeline.models import FeedConfig, NewsItem

from scripts.news_ingestion.fetch_sources_cli import (
    SourceFetchResult,
    collect_source_results,
    fetch_items_for_source,
    filter_items_by_watermark,
    run_fetch_cli,
)


def _make_news_item(source: str, minutes_offset: int) -> NewsItem:
    return NewsItem(
        url=f"https://example.com/{source}/{minutes_offset}",
        title=f"Title {minutes_offset}",
        publication_date=datetime.utcnow() + timedelta(minutes=minutes_offset),
        source_name=source,
        publisher="Example",
        description="desc",
    )


class _DummyRSS:
    def __init__(self, items):
        self.items = items

    def fetch_multiple(self, feeds):
        return self.items


class _DummySitemap:
    def __init__(self, items):
        self.items = items

    def fetch_sitemap(self, feed):
        return self.items


class _DummyStorage:
    def __init__(self, watermark_map):
        self.watermark_map = watermark_map

    def get_watermark(self, name: str):
        return self.watermark_map.get(name)


def test_filter_items_by_watermark_respects_timestamp():
    watermark = datetime.utcnow()
    newer = _make_news_item("espn", 5)
    older = _make_news_item("espn", -5)

    filtered = filter_items_by_watermark([newer, older], watermark, ignore_watermark=False)

    assert filtered == [newer]


@pytest.mark.parametrize("ignore", [True, False])
def test_collect_source_results_applies_limit_and_watermark(ignore):
    feed = FeedConfig(name="espn", type="rss", publisher="ESPN")
    newer = _make_news_item("espn", 10)
    older = _make_news_item("espn", -10)

    rss = _DummyRSS([newer, older])
    sitemap = _DummySitemap([])
    storage = _DummyStorage({"espn": datetime.utcnow()})

    results, total = collect_source_results(
        sources=[feed],
        rss=rss,  # type: ignore[arg-type]
        sitemap=sitemap,  # type: ignore[arg-type]
        storage=storage,
        ignore_watermark=ignore,
        per_source_limit=1,
    )

    assert isinstance(results[0], SourceFetchResult)
    if ignore:
        assert results[0].returned == 1
        assert total == 1
        assert results[0].items[0].url == newer.url
    else:
        assert results[0].returned == 1  # limit keeps one
        assert total == 1
        # watermark should drop older item; limit then applies to remaining list
        assert results[0].items[0].url == newer.url


def test_fetch_items_for_source_prefers_rss_for_rss_feed():
    feed = FeedConfig(name="espn", type="rss", publisher="ESPN")
    items = [_make_news_item("espn", 0)]
    rss = _DummyRSS(items)
    sitemap = _DummySitemap([])

    fetched = fetch_items_for_source(feed, rss, sitemap)  # type: ignore[arg-type]
    assert fetched == items


def test_run_fetch_cli_writes_supabase(monkeypatch):
    result = SourceFetchResult(
        source=FeedConfig(name="espn", type="rss", publisher="ESPN"),
        fetched=1,
        returned=1,
        watermark=None,
        items=[_make_news_item("espn", 0)],
    )

    monkeypatch.setattr(
        "scripts.news_ingestion.fetch_sources_cli.fetch_sources",
        lambda **kwargs: ([result], 1),
    )

    stored = {}

    def fake_store(results):
        stored["count"] = sum(len(r.items) for r in results)
        return (stored["count"], 0)

    monkeypatch.setattr(
        "scripts.news_ingestion.fetch_sources_cli._store_to_supabase",
        fake_store,
    )

    rc = run_fetch_cli(
        cfg_path="feeds.yaml",
        source=None,
        use_watermarks=False,
        ignore_watermark=False,
        per_source_limit=None,
        output_path=None,
        write_supabase=True,
    )

    assert rc == 0
    assert stored.get("count") == 1
