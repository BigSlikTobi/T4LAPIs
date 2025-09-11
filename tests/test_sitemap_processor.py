from datetime import datetime, timezone, timedelta

import pytest

from src.nfl_news_pipeline.models import DefaultsConfig, FeedConfig
from src.nfl_news_pipeline.processors.sitemap import SitemapProcessor


SITEMAP_XML = b"""
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/article/3</loc>
    <lastmod>2025-09-11T12:00:00Z</lastmod>
  </url>
  <url>
    <loc>https://example.com/article/2</loc>
    <lastmod>2025-09-10T15:00:00+00:00</lastmod>
  </url>
  <url>
    <loc>https://example.com/article/1</loc>
    <lastmod>2025-08-31</lastmod>
  </url>
</urlset>
"""


@pytest.mark.asyncio
async def test_fetch_sitemap_async_with_filters(monkeypatch):
    defaults = DefaultsConfig(user_agent="UA", timeout_seconds=5, max_parallel_fetches=3)
    feed = FeedConfig(
        name="NFL Sitemap",
        type="sitemap",
        publisher="NFL.com",
        url_template="https://www.nfl.com/sitemap/html/articles/{YYYY}/{MM}",
        max_articles=2,
        days_back=7,
    )
    sp = SitemapProcessor(defaults)

    class FakeResponse:
        status_code = 200
        content = SITEMAP_XML
        request = None
        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: FakeClient())

    now = datetime(2025, 9, 11, 13, 0, 0, tzinfo=timezone.utc)
    items = await sp.fetch_sitemap_async(feed, now=now)

    # days_back=7 should filter out 2025-08-31
    assert [i.url for i in items] == [
        "https://example.com/article/3",
        "https://example.com/article/2",
    ]


def test_construct_sitemap_url():
    defaults = DefaultsConfig(user_agent="UA", timeout_seconds=5, max_parallel_fetches=3)
    feed = FeedConfig(
        name="NFL Sitemap",
        type="sitemap",
        publisher="NFL.com",
        url_template="https://www.nfl.com/sitemap/html/articles/{YYYY}/{MM}",
    )
    sp = SitemapProcessor(defaults)
    now = datetime(2025, 9, 11, tzinfo=timezone.utc)
    out = sp.construct_sitemap_url(feed, now=now)
    assert out == "https://www.nfl.com/sitemap/html/articles/2025/09"


@pytest.mark.asyncio
async def test_fetch_sitemap_async_parsing_errors(monkeypatch):
    defaults = DefaultsConfig(user_agent="UA", timeout_seconds=5, max_parallel_fetches=3)
    feed = FeedConfig(
        name="NFL Sitemap",
        type="sitemap",
        publisher="NFL.com",
        url_template="https://www.nfl.com/sitemap/html/articles/{YYYY}/{MM}",
    )
    sp = SitemapProcessor(defaults)

    class BadResponse:
        status_code = 200
        content = b"<not-xml"
        request = None
        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def get(self, url):
            return BadResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: FakeClient())

    items = await sp.fetch_sitemap_async(feed)
    assert items == []
