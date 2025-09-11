import os
from pathlib import Path
from datetime import datetime, timezone

import pytest

from src.nfl_news_pipeline.config import ConfigManager, ConfigError
from src.nfl_news_pipeline.models import FeedConfig


TEST_YAML = """
version: 1

defaults:
  user_agent: TestAgent/1.0
  timeout_seconds: 12
  max_parallel_fetches: 3

sources:
  - name: ESPN - NFL News
    type: rss
    url: https://www.espn.com/espn/rss/nfl/news
    publisher: ESPN
    nfl_only: true
    enabled: true

  - name: NFL.com - Articles Monthly Sitemap
    type: sitemap
    url_template: https://www.nfl.com/sitemap/html/articles/{YYYY}/{MM}
    publisher: NFL.com
    nfl_only: true
    enabled: true
    max_articles: 30
    days_back: 7
    extract_content: true

  - name: Fox Sports - NFL News HTML
    type: html
    url: https://www.foxsports.com/nfl/news
    publisher: Fox Sports
    nfl_only: true
    enabled: false
"""


def write_tmp_feeds(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "feeds.yaml"
    p.write_text(content)
    return p


def test_load_and_validate(tmp_path: Path):
    cfg_path = write_tmp_feeds(tmp_path, TEST_YAML)
    cm = ConfigManager(cfg_path)
    config = cm.load_config()

    # defaults
    d = cm.get_defaults()
    assert d.user_agent == "TestAgent/1.0"
    assert d.timeout_seconds == 12
    assert d.max_parallel_fetches == 3

    # sources
    enabled = cm.get_enabled_sources()
    assert len(enabled) == 2  # ESPN + NFL.com (HTML disabled)
    types = {s.type for s in enabled}
    assert types == {"rss", "sitemap"}

    # warnings: extract_content forced to False
    warns = cm.get_warnings()
    assert any("extract_content" in w for w in warns)


def test_construct_sitemap_url(tmp_path: Path):
    cfg_path = write_tmp_feeds(tmp_path, TEST_YAML)
    cm = ConfigManager(cfg_path)
    config = cm.load_config()

    sm = next(s for s in cm.get_enabled_sources() if s.type == "sitemap")
    fixed = datetime(2025, 9, 11, tzinfo=timezone.utc)
    url = cm.construct_url(sm, now=fixed)
    assert url == "https://www.nfl.com/sitemap/html/articles/2025/09"


def test_invalid_entries(tmp_path: Path):
    bad_yaml = """
    version: 1
    sources:
      - name: Bad RSS
        type: rss
      - name: Bad Sitemap
        type: sitemap
        url_template: 123
      - name: Bad HTML
        type: html
        url: ftp://example.com
    """
    cfg_path = write_tmp_feeds(tmp_path, bad_yaml)
    cm = ConfigManager(cfg_path)

    with pytest.raises(ConfigError):
        cm.load_config()
