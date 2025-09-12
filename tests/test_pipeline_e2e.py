from __future__ import annotations

"""
End-to-end tests for the NFL News Pipeline (Task 12.1)

- Runs the full orchestrator against mocked RSS and Sitemap inputs
- Uses dry-run/in-memory storage to avoid DB writes
- Also exercises the CLI entrypoint in dry-run mode
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import os
import textwrap
import types

import pytest

from src.nfl_news_pipeline.orchestrator import NFLNewsPipeline
from src.nfl_news_pipeline.config import ConfigManager


class DryRunStorage:
    def __init__(self):
        self.rows = []
        self.watermarks: Dict[str, datetime] = {}

    def add_audit_event(self, *args, **kwargs):
        return True

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
        return DryRunStorage._Resp(len(items), 0)

    def get_watermark(self, source_name: str):
        return self.watermarks.get(source_name)

    def update_watermark(self, source_name: str, *, last_processed_date: datetime, **kwargs):
        self.watermarks[source_name] = last_processed_date
        return True


def _write_temp_config(tmp_path: Path, rss_url: str, sitemap_url_template: str) -> Path:
    cfg = textwrap.dedent(
        f"""
        defaults:
          user_agent: "pytest"
          timeout_seconds: 5
          max_parallel_fetches: 3
        sources:
          - name: Test RSS
            type: rss
            publisher: ESPN - NFL News
            enabled: true
            url: {rss_url}
          - name: Test Sitemap
            type: sitemap
            publisher: NFL.com
            enabled: true
            url_template: {sitemap_url_template}
            max_articles: 10
        """
    ).strip()
    p = tmp_path / "feeds.yaml"
    p.write_text(cfg)
    return p


@pytest.fixture
def rss_bytes() -> bytes:
    return textwrap.dedent(
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>NFL roundup week 1</title>
              <link>https://example.com/nfl/1</link>
              <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
              <description>Summary</description>
            </item>
            <item>
              <title>Weather update</title>
              <link>https://example.com/weather</link>
              <pubDate>Mon, 01 Jan 2024 01:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """
    ).encode("utf-8")


@pytest.fixture
def sitemap_bytes() -> bytes:
    # Title will be the URL in sitemap processor; embed 'NFL' in path to trigger keyword match
    return textwrap.dedent(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url>
            <loc>https://www.nfl.com/NFL-Roundup</loc>
            <lastmod>2025-09-12T00:00:00Z</lastmod>
          </url>
          <url>
            <loc>https://example.com/other</loc>
            <lastmod>2025-09-12T00:00:00Z</lastmod>
          </url>
        </urlset>
        """
    ).encode("utf-8")


def test_e2e_pipeline_rss_and_sitemap(tmp_path, monkeypatch, rss_bytes, sitemap_bytes):
    # Arrange: temp feeds and monkeypatch network fetchers
    cfg_path = _write_temp_config(
        tmp_path,
        rss_url="https://test.local/rss.xml",
        sitemap_url_template="https://test.local/sitemap-{YYYY}-{MM}-{DD}.xml",
    )

    # Force ONLY these two sources by not depending on project feeds.yaml
    from src.nfl_news_pipeline.processors import rss as rss_mod
    from src.nfl_news_pipeline.processors import sitemap as site_mod

    monkeypatch.setenv("NEWS_PIPELINE_DISABLE_LLM", "1")

    async def fake_rss_get(self, url: str):  # type: ignore[override]
        assert url == "https://test.local/rss.xml"
        return rss_bytes

    async def fake_site_get(self, url: str):  # type: ignore[override]
        assert url.startswith("https://test.local/sitemap-")
        return sitemap_bytes

    monkeypatch.setattr(rss_mod.RSSProcessor, "_http_get_with_retry", fake_rss_get, raising=True)
    monkeypatch.setattr(site_mod.SitemapProcessor, "_http_get_with_retry", fake_site_get, raising=True)

    storage = DryRunStorage()
    pipeline = NFLNewsPipeline(str(cfg_path), storage=storage, audit=None)

    # Act
    summary = pipeline.run()

    # Assert
    # Fetched: 2 RSS + 2 Sitemap = 4
    assert summary.sources == 2
    assert summary.fetched_items == 4
    # Kept: "NFL roundup week 1" (keywords + url path) and sitemap nfl.com/NFL-Roundup => 2
    assert summary.filtered_in >= 2
    # Inserted count should match kept in dry-run
    assert summary.inserted == summary.filtered_in
    # Watermark updated for at least one source
    assert storage.watermarks.get("Test RSS") or storage.watermarks.get("Test Sitemap")


def test_e2e_cli_run_single_source_dry_run(tmp_path, monkeypatch, rss_bytes, capsys):
    # Arrange: temp feeds with only RSS; run CLI with --source matching publisher/name loosely
    cfg_path = _write_temp_config(
        tmp_path,
        rss_url="https://test.local/rss.xml",
        sitemap_url_template="https://test.local/sitemap-{YYYY}-{MM}-{DD}.xml",
    )
    monkeypatch.setenv("NEWS_PIPELINE_DISABLE_LLM", "1")

    from src.nfl_news_pipeline.processors import rss as rss_mod
    async def fake_rss_get(self, url: str):  # type: ignore[override]
        return rss_bytes
    monkeypatch.setattr(rss_mod.RSSProcessor, "_http_get_with_retry", fake_rss_get, raising=True)

    # Import CLI main and run
    import scripts.pipeline_cli as cli
    rc = cli.main(["run", "--config", str(cfg_path), "--source", "espn", "--dry-run", "--disable-llm"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Run summary:" in out
    assert "sources=1" in out
    assert "Dry-run stored items" in out
