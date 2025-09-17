from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import yaml

from .models import DefaultsConfig, FeedConfig, PipelineConfig


class ConfigError(Exception):
    pass


class ConfigManager:
    """Parse and minimally validate feeds.yaml.

    Extended for Task 2: validation, error handling, and url_template
    construction with date placeholders.
    """

    def __init__(self, config_path: str | Path = "feeds.yaml") -> None:
        self.config_path = Path(config_path)
        self._config: PipelineConfig | None = None
        self._warnings: List[str] = []

    def load_config(self) -> PipelineConfig:
        if not self.config_path.exists():
            raise ConfigError(f"Config file not found: {self.config_path}")

        try:
            raw = yaml.safe_load(self.config_path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {self.config_path}: {e}") from e

        defaults = raw.get("defaults", {})
        sources = raw.get("sources", [])
        if not isinstance(sources, list):
            raise ConfigError("'sources' must be a list in feeds.yaml")

        defaults_cfg = DefaultsConfig(
            user_agent=str(defaults.get("user_agent", "Mozilla/5.0")),
            timeout_seconds=int(defaults.get("timeout_seconds", 10)),
            max_parallel_fetches=int(defaults.get("max_parallel_fetches", 5)),
            enable_story_grouping=bool(defaults.get("enable_story_grouping", False)),
            story_grouping_max_parallelism=int(defaults.get("story_grouping_max_parallelism", 4)),
            story_grouping_max_stories_per_run=defaults.get("story_grouping_max_stories_per_run"),
            story_grouping_reprocess_existing=bool(defaults.get("story_grouping_reprocess_existing", False)),
        )

        feed_cfgs: List[FeedConfig] = []
        for idx, s in enumerate(sources):
            if not isinstance(s, dict):
                raise ConfigError(f"source[{idx}] must be an object")
            try:
                feed_cfgs.append(self._validate_and_build_feed_config(s, idx))
            except KeyError as e:
                raise ConfigError(f"Missing required field in source[{idx}]: {e}") from e

        self._config = PipelineConfig(defaults=defaults_cfg, sources=feed_cfgs)
        return self._config

    def get_enabled_sources(self) -> List[FeedConfig]:
        cfg = self._ensure()
        return [s for s in cfg.sources if s.enabled]

    def get_defaults(self) -> DefaultsConfig:
        return self._ensure().defaults

    def to_dict(self) -> Dict[str, Any]:
        cfg = self._ensure()
        return {
            "defaults": asdict(cfg.defaults),
            "sources": [asdict(s) for s in cfg.sources],
        }

    def get_warnings(self) -> List[str]:
        return list(self._warnings)

    # --- Validation and helpers ---
    def _validate_and_build_feed_config(self, s: Dict[str, Any], idx: int) -> FeedConfig:
        name = str(s["name"]).strip()
        if not name:
            raise ConfigError(f"source[{idx}] 'name' cannot be empty")

        type_val = str(s["type"]).lower().strip()
        if type_val not in {"rss", "sitemap", "html"}:
            raise ConfigError(
                f"source[{idx}] 'type' must be one of ['rss','sitemap','html'], got '{type_val}'"
            )

        url = s.get("url")
        url_template = s.get("url_template")

        # Type-specific requirements
        if type_val == "rss":
            if not url or not self._looks_like_url(url):
                raise ConfigError(f"source[{idx}] rss requires valid 'url' (http/https)")
            if url_template:
                self._warnings.append(
                    f"source[{idx}] rss ignores 'url_template' when 'url' is provided"
                )
        elif type_val == "sitemap":
            if not url_template or not isinstance(url_template, str):
                raise ConfigError(f"source[{idx}] sitemap requires 'url_template' (string)")
            if url and not self._looks_like_url(url):
                raise ConfigError(f"source[{idx}] invalid 'url' format: {url}")
        elif type_val == "html":
            if not url or not self._looks_like_url(url):
                raise ConfigError(f"source[{idx}] html requires valid 'url' (http/https)")

        # Numeric validations
        max_articles = s.get("max_articles")
        if max_articles is not None:
            try:
                max_articles = int(max_articles)
                if max_articles < 0:
                    raise ValueError
            except Exception:
                raise ConfigError(f"source[{idx}] 'max_articles' must be a non-negative integer")

        days_back = s.get("days_back")
        if days_back is not None:
            try:
                days_back = int(days_back)
                if days_back < 0:
                    raise ValueError
            except Exception:
                raise ConfigError(f"source[{idx}] 'days_back' must be a non-negative integer")

        # Compliance: never extract full article content
        extract_content = bool(s.get("extract_content", False))
        if extract_content:
            self._warnings.append(
                f"source[{idx}] 'extract_content' is not permitted by policy; forcing to False"
            )
            extract_content = False

        return FeedConfig(
            name=name,
            type=type_val,
            publisher=str(s.get("publisher", name)),
            enabled=bool(s.get("enabled", True)),
            nfl_only=bool(s.get("nfl_only", False)),
            url=url,
            url_template=url_template,
            max_articles=max_articles,
            days_back=days_back,
            extract_content=extract_content,
        )

    @staticmethod
    def _looks_like_url(u: str) -> bool:
        return isinstance(u, str) and (u.startswith("http://") or u.startswith("https://"))

    def construct_url(self, feed: FeedConfig, now: Optional[datetime] = None) -> str:
        """Construct the concrete URL for a feed.

        - rss/html: returns the static 'url'
        - sitemap: replaces {YYYY}, {MM}, {DD} in 'url_template' using UTC
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if feed.type in {"rss", "html"}:
            if not feed.url:
                raise ConfigError(f"Feed '{feed.name}' missing 'url'")
            return feed.url

        if feed.type == "sitemap":
            if not feed.url_template:
                raise ConfigError(f"Feed '{feed.name}' missing 'url_template'")
            yyyy = f"{now.year:04d}"
            mm = f"{now.month:02d}"
            dd = f"{now.day:02d}"
            return (
                feed.url_template
                .replace("{YYYY}", yyyy)
                .replace("{MM}", mm)
                .replace("{DD}", dd)
            )

        raise ConfigError(f"Unsupported feed type: {feed.type}")

    # internal
    def _ensure(self) -> PipelineConfig:
        if self._config is None:
            raise ConfigError("Configuration not loaded. Call load_config() first.")
        return self._config
