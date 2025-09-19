#!/usr/bin/env python3
"""
End-to-end pipeline runner that chains ingestion and story grouping.

This CLI fetches sources, stores new stories, extracts URL context,
creates embeddings, runs similarity calculations, and forms story groups
whenever new articles (or brand-new sources) are discovered during fetch.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

from .pipeline_cli import _build_storage, _filter_sources  # reuse helpers
from ._tracking import SourceBatch, TrackingStorageAdapter

# --------------------------------------------------------------------------------------
# Repository bootstrap
# --------------------------------------------------------------------------------------


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    return start.parents[0]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(str(ROOT / ".env"), override=False)

# --------------------------------------------------------------------------------------
# Domain imports (after sys.path adjustments)
# --------------------------------------------------------------------------------------
from src.nfl_news_pipeline.config import ConfigManager
from src.nfl_news_pipeline.logging import AuditLogger
from src.nfl_news_pipeline.models import ProcessedNewsItem
from src.nfl_news_pipeline.orchestrator import NFLNewsPipeline
from src.nfl_news_pipeline.orchestrator.story_grouping import (
    StoryGroupingOrchestrator,
    StoryGroupingSettings,
)
from src.nfl_news_pipeline.group_manager import GroupManager, GroupStorageManager
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingErrorHandler
from src.nfl_news_pipeline.similarity import SimilarityCalculator
from src.nfl_news_pipeline.story_grouping import URLContextExtractor
from src.nfl_news_pipeline.storage.protocols import (
    get_grouping_client,
    has_story_grouping_capability,
)

# --------------------------------------------------------------------------------------
# Utility contexts
# --------------------------------------------------------------------------------------


def _resolve_config_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)
    candidate = ROOT / path_str
    if candidate.exists():
        return str(candidate)
    return str(path)


@contextmanager
def temporary_env(updates: Dict[str, Optional[str]]):
    """Temporarily set environment variables and restore afterwards."""

    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# --------------------------------------------------------------------------------------
# Story grouping setup
# --------------------------------------------------------------------------------------


def _build_story_grouping_orchestrator(
    storage: TrackingStorageAdapter,
    *,
    defaults,
    audit: AuditLogger,
) -> Optional[StoryGroupingOrchestrator]:
    if not has_story_grouping_capability(storage):
        return None

    settings = StoryGroupingSettings()
    if defaults:
        settings.max_parallelism = defaults.story_grouping_max_parallelism
        settings.max_stories_per_run = defaults.story_grouping_max_stories_per_run
        settings.reprocess_existing = defaults.story_grouping_reprocess_existing
    settings.validate()

    context_extractor = URLContextExtractor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    embedding_generator = EmbeddingGenerator(openai_api_key=os.getenv("OPENAI_API_KEY"))
    similarity_calculator = SimilarityCalculator()
    error_handler = EmbeddingErrorHandler()

    grouping_client = get_grouping_client(storage)
    group_storage = GroupStorageManager(grouping_client)
    centroid_manager = GroupCentroidManager()
    group_manager = GroupManager(group_storage, similarity_calculator, centroid_manager)

    return StoryGroupingOrchestrator(
        context_extractor=context_extractor,
        embedding_generator=embedding_generator,
        group_manager=group_manager,
        error_handler=error_handler,
        settings=settings,
        audit_logger=audit,
    )


# --------------------------------------------------------------------------------------
# Core workflow
# --------------------------------------------------------------------------------------


def _select_source(cm: ConfigManager, source_name: Optional[str]) -> Optional[str]:
    if not source_name:
        return None
    matches = _filter_sources(cm.get_enabled_sources(), source_name)
    if not matches:
        enabled = ", ".join(sorted(s.name for s in cm.get_enabled_sources()))
        raise ValueError(f"No enabled source matched '{source_name}'. Try one of: {enabled}")
    chosen = matches[0]
    if len(matches) > 1:
        also = ", ".join(s.name for s in matches[1:3])
        suffix = "..." if len(matches) > 3 else ""
        print(f"Matched source '{chosen.name}'. Other matches: {also}{suffix}")
    return chosen.name


def run_full_pipeline(
    *,
    cfg_path: str,
    source: Optional[str],
    dry_run: bool,
    disable_llm: bool,
    llm_timeout: Optional[float],
    ignore_watermark: bool,
) -> int:
    cm = ConfigManager(cfg_path)
    cm.load_config()
    defaults = cm.get_defaults()

    base_storage = _build_storage(dry_run)
    storage = TrackingStorageAdapter(base_storage)
    audit = AuditLogger(storage=storage)

    selected_source = None
    if source:
        selected_source = _select_source(cm, source)

    env_updates: Dict[str, Optional[str]] = {
        "NEWS_PIPELINE_DISABLE_STORY_GROUPING": "1",  # ensure pipeline does not self-trigger grouping
        "NEWS_PIPELINE_ENABLE_STORY_GROUPING": None,
    }
    if disable_llm:
        env_updates["NEWS_PIPELINE_DISABLE_LLM"] = "1"
    if llm_timeout is not None:
        env_updates["OPENAI_TIMEOUT"] = str(llm_timeout)
    if ignore_watermark:
        env_updates["NEWS_PIPELINE_IGNORE_WATERMARK"] = "1"
    if selected_source:
        env_updates["NEWS_PIPELINE_ONLY_SOURCE"] = selected_source

    pipeline = NFLNewsPipeline(cfg_path, storage=storage, audit=audit)

    with temporary_env(env_updates):
        summary = pipeline.run()

    print(
        "Ingestion summary: "
        f"sources={summary.sources} fetched={summary.fetched_items} kept={summary.filtered_in} "
        f"inserted={summary.inserted} updated={summary.updated} errors={summary.errors} "
        f"store_errors={summary.store_errors} time={summary.duration_ms}ms"
    )

    orchestrator = _build_story_grouping_orchestrator(storage, defaults=defaults, audit=audit)
    if orchestrator is None:
        if dry_run:
            print("Skipping story grouping (dry-run storage has no grouping capability).")
        else:
            print("Story grouping skipped: storage backend does not support grouping operations.")
        return 0

    total_processed = 0
    total_new_groups = 0
    total_updated_groups = 0

    for batch in storage.batches:
        if not batch.has_new_material():
            continue
        items_with_ids = list(batch.iter_items_with_ids())
        if not items_with_ids:
            continue
        label = "new source" if batch.is_new_source else "updated source"
        print(
            f"Running story grouping for {len(items_with_ids)} item(s) from '{batch.source_name}' ({label})."
        )
        url_id_map = {item.url: batch.ids_by_url[item.url] for item in items_with_ids}
        result = asyncio.run(orchestrator.process_batch(items_with_ids, url_id_map))
        total_processed += result.metrics.processed_stories
        total_new_groups += result.metrics.new_groups_created
        total_updated_groups += result.metrics.existing_groups_updated
        print(
            "  Story grouping metrics: "
            f"processed={result.metrics.processed_stories} skipped={result.metrics.skipped_stories} "
            f"new_groups={result.metrics.new_groups_created} updated_groups={result.metrics.existing_groups_updated} "
            f"time={result.metrics.total_processing_time_ms}ms"
        )

    if total_processed:
        print(
            "Story grouping summary: "
            f"processed={total_processed} new_groups={total_new_groups} updated_groups={total_updated_groups}"
        )
    else:
        print("No new or updated items required story grouping.")

    return 0


# --------------------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ingestion and story grouping in series")
    parser.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    parser.add_argument("--source", help="Process only a single enabled source by name")
    parser.add_argument("--dry-run", action="store_true", help="Use in-memory storage (no writes)")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM-based relevance filtering")
    parser.add_argument("--llm-timeout", type=float, help="Override LLM timeout in seconds")
    parser.add_argument("--ignore-watermark", action="store_true", help="Ignore source watermarks for this run")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg_path = _resolve_config_path(str(getattr(args, "config", "feeds.yaml")))

    try:
        return run_full_pipeline(
            cfg_path=cfg_path,
            source=getattr(args, "source", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            disable_llm=bool(getattr(args, "disable_llm", False)),
            llm_timeout=getattr(args, "llm_timeout", None),
            ignore_watermark=bool(getattr(args, "ignore_watermark", False)),
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
