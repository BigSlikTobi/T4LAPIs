from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .models import DefaultsConfig, FeedConfig, PipelineConfig


class ConfigError(Exception):
    pass


class ConfigManager:
    """Parse and minimally validate feeds.yaml.

    Task 1 scope: structure + core models + basic loading. Full validation
    and url_template logic will be delivered in Task 2.
    """

    def __init__(self, config_path: str | Path = "feeds.yaml") -> None:
        self.config_path = Path(config_path)
        self._config: PipelineConfig | None = None

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
        )

        feed_cfgs: List[FeedConfig] = []
        for idx, s in enumerate(sources):
            if not isinstance(s, dict):
                raise ConfigError(f"source[{idx}] must be an object")
            try:
                feed_cfgs.append(
                    FeedConfig(
                        name=str(s["name"]),
                        type=str(s["type"]).lower(),
                        publisher=str(s.get("publisher", s["name"])),
                        enabled=bool(s.get("enabled", True)),
                        nfl_only=bool(s.get("nfl_only", False)),
                        url=s.get("url"),
                        url_template=s.get("url_template"),
                        max_articles=s.get("max_articles"),
                        days_back=s.get("days_back"),
                        extract_content=bool(s.get("extract_content", False)),
                    )
                )
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

    # internal
    def _ensure(self) -> PipelineConfig:
        if self._config is None:
            raise ConfigError("Configuration not loaded. Call load_config() first.")
        return self._config
