from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from src.nfl_news_pipeline.models import DefaultsConfig, FeedConfig
from src.nfl_news_pipeline.processors.rss import RSSProcessor


RSS_XML = b"""
<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <title>Test RSS</title>
    <link>https://example.com</link>
    <description>Example feed</description>
    <item>
      <title>NFL: Team signs player</title>
      <link>https://example.com/news/1</link>
      <description>Some summary</description>
      <pubDate>Thu, 11 Sep 2025 12:34:56 GMT</pubDate>
    </item>
    <item>
      <title>Another headline</title>
      <link>https://example.com/news/2</link>
      <description>More summary</description>
      <pubDate>Thu, 11 Sep 2025 09:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.mark.asyncio
async def test_fetch_feed_async(monkeypatch):
    defaults = DefaultsConfig(user_agent="TestUA", timeout_seconds=5, max_parallel_fetches=2)
    feed = FeedConfig(name="Test RSS", type="rss", publisher="X", url="https://example.com/rss.xml")
    rp = RSSProcessor(defaults)

    class FakeResponse:
        status_code = 200
        content = RSS_XML
        request = None
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)

    items = await rp.fetch_feed_async(feed)
    assert len(items) == 2
    assert items[0].title.startswith("NFL:")
    assert items[0].url == "https://example.com/news/1"
    assert items[0].source_name == "Test RSS"
    assert items[0].publisher == "X"
    assert items[0].publication_date.tzinfo is not None


def test_fetch_feed_sync_with_retry(monkeypatch):
    defaults = DefaultsConfig(user_agent="TestUA", timeout_seconds=5, max_parallel_fetches=2)
    feed = FeedConfig(name="Test RSS", type="rss", publisher="X", url="https://example.com/rss.xml")
    rp = RSSProcessor(defaults, max_retries=2, base_delay=0)

    class FailThenSuccess:
        def __init__(self):
            self.calls = 0
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def get(self, url):
            self.calls += 1
            if self.calls == 1:
                class R:
                    status_code = 503
                    request = None
                    def raise_for_status(self):
                        import httpx
                        raise httpx.HTTPStatusError("server error", request=None, response=self)
                return R()
            class OK:
                status_code = 200
                content = RSS_XML
                request = None
                def raise_for_status(self):
                    return None
            return OK()

    class FakeClientFactory:
        def __init__(self, *a, **k):
            pass
        def __call__(self, *a, **k):
            return self
        async def __aenter__(self):
            return FailThenSuccess()
        async def __aexit__(self, *exc):
            return False

    # Monkeypatch the context manager to our special factory
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return FailThenSuccess()
        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    items = rp.fetch_feed(feed)
    assert len(items) == 2


@pytest.mark.asyncio
async def test_fetch_multiple_async_handles_exceptions(monkeypatch):
    defaults = DefaultsConfig(user_agent="TestUA", timeout_seconds=5, max_parallel_fetches=4)
    feeds = [
        FeedConfig(name="Good", type="rss", publisher="X", url="https://ok"),
        FeedConfig(name="Bad", type="rss", publisher="X", url="https://bad"),
    ]
    rp = RSSProcessor(defaults)

    class GoodResp:
        status_code = 200
        content = RSS_XML
        request = None
        def raise_for_status(self):
            return None

    class BadResp:
        status_code = 404
        request = None
        def raise_for_status(self):
            import httpx
            raise httpx.HTTPStatusError("not found", request=None, response=self)

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def get(self, url):
            if url == "https://ok":
                return GoodResp()
            return BadResp()

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: FakeClient())

    items = await rp.fetch_multiple_async(feeds)
    assert len(items) == 2
