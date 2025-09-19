#!/usr/bin/env python3
"""CLI to fetch news items from configured sources without further processing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from .pipeline_cli import _build_storage, _filter_sources

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
from src.nfl_news_pipeline.entities.enrichment import (
    build_entities_extractor,
    enrich_processed_item,
)
from src.nfl_news_pipeline.models import FeedConfig, NewsItem, ProcessedNewsItem
from src.nfl_news_pipeline.processors.rss import RSSProcessor
from src.nfl_news_pipeline.processors.sitemap import SitemapProcessor
from src.nfl_news_pipeline.storage.manager import StorageManager

# --------------------------------------------------------------------------------------
# Helper data structures
# --------------------------------------------------------------------------------------


@dataclass
class SourceFetchResult:
    source: FeedConfig
    fetched: int
    returned: int
    watermark: Optional[datetime]
    items: List[NewsItem]


# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------


def _resolve_config_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)
    candidate = ROOT / path_str
    if candidate.exists():
        return str(candidate)
    return str(path)


def fetch_items_for_source(
    feed: FeedConfig,
    rss: RSSProcessor,
    sitemap: SitemapProcessor,
) -> List[NewsItem]:
    """Fetch raw items for a single feed configuration."""
    if feed.type == "rss":
        items = [it for it in rss.fetch_multiple([feed]) if it.source_name == feed.name]
    elif feed.type == "sitemap":
        items = sitemap.fetch_sitemap(feed)
    else:
        items = []
    return items


def filter_items_by_watermark(
    items: Iterable[NewsItem],
    watermark: Optional[datetime],
    *,
    ignore_watermark: bool,
) -> List[NewsItem]:
    if ignore_watermark or watermark is None:
        return list(items)
    return [it for it in items if (it.publication_date or datetime.min) > watermark]


def news_item_to_dict(item: NewsItem, *, story_id: Optional[str] = None) -> Dict[str, object]:
    return {
        "story_id": story_id,
        "url": item.url,
        "title": item.title,
        "publication_date": item.publication_date.isoformat() if item.publication_date else None,
        "source_name": item.source_name,
        "publisher": item.publisher,
        "description": item.description,
        "raw_metadata": item.raw_metadata or {},
    }


def _news_to_processed(item: NewsItem) -> ProcessedNewsItem:
    return ProcessedNewsItem(
        url=item.url,
        title=item.title,
        publication_date=item.publication_date,
        source_name=item.source_name,
        publisher=item.publisher,
        description=item.description,
        raw_metadata=item.raw_metadata,
        relevance_score=0.0,
        filter_method="fetch_sources_cli",
        filter_reasoning="fetch_sources_cli",
        entities=[],
        categories=[],
    )


def collect_source_results(
    *,
    sources: Sequence[FeedConfig],
    rss: RSSProcessor,
    sitemap: SitemapProcessor,
    storage,
    ignore_watermark: bool,
    per_source_limit: Optional[int],
) -> Tuple[List[SourceFetchResult], int]:
    results: List[SourceFetchResult] = []
    total_returned = 0

    for feed in sources:
        items = fetch_items_for_source(feed, rss, sitemap)
        fetched_count = len(items)

        watermark = None
        if storage is not None:
            watermark = storage.get_watermark(feed.name)

        filtered_items = filter_items_by_watermark(items, watermark, ignore_watermark=ignore_watermark)
        if per_source_limit is not None:
            filtered_items = filtered_items[: max(per_source_limit, 0)]

        results.append(
            SourceFetchResult(
                source=feed,
                fetched=fetched_count,
                returned=len(filtered_items),
                watermark=watermark,
                items=filtered_items,
            )
        )
        total_returned += len(filtered_items)

    return results, total_returned


def fetch_sources(
    *,
    cfg_path: str,
    source: Optional[str],
    use_watermarks: bool,
    ignore_watermark: bool,
    per_source_limit: Optional[int],
) -> Tuple[List[SourceFetchResult], int]:
    cm = ConfigManager(cfg_path)
    cm.load_config()
    defaults = cm.get_defaults()

    sources = cm.get_enabled_sources()
    if source:
        matches = _filter_sources(sources, source)
        if not matches:
            names = ", ".join(sorted(s.name for s in sources))
            raise ValueError(f"No enabled source matched '{source}'. Try one of: {names}")
        sources = matches

    rss = RSSProcessor(defaults)
    sitemap = SitemapProcessor(defaults)

    storage = None
    if use_watermarks:
        storage = _build_storage(dry_run=False)

    return collect_source_results(
        sources=sources,
        rss=rss,
        sitemap=sitemap,
        storage=storage,
        ignore_watermark=ignore_watermark,
        per_source_limit=per_source_limit,
    )


def _store_to_supabase(results: List[SourceFetchResult]) -> Tuple[int, int]:
    storage = _build_storage(dry_run=False)
    if not isinstance(storage, StorageManager):
        raise RuntimeError("Supabase storage unavailable; configure credentials or use --dry-run")

    extractor = build_entities_extractor()
    inserted = 0
    updated = 0
    for result in results:
        if not result.items:
            continue
        processed_items = [enrich_processed_item(_news_to_processed(item), extractor) for item in result.items]
        store_result = storage.store_news_items(processed_items)
        inserted += store_result.inserted_count
        updated += store_result.updated_count
    return inserted, updated


# --------------------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch news items from configured sources")
    parser.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    parser.add_argument("--source", help="Process only a single enabled source by name")
    parser.add_argument("--use-watermarks", action="store_true", help="Filter results using source watermarks")
    parser.add_argument("--ignore-watermark", action="store_true", help="Ignore stored watermark values even when available")
    parser.add_argument("--limit", type=int, help="Maximum items to return per source")
    parser.add_argument("--output", help="Optional path to write fetched items as JSON")
    parser.add_argument("--write-supabase", action="store_true", help="Persist fetched items to Supabase")
    return parser


def run_fetch_cli(
    *,
    cfg_path: str,
    source: Optional[str],
    use_watermarks: bool,
    ignore_watermark: bool,
    per_source_limit: Optional[int],
    output_path: Optional[str],
    write_supabase: bool,
) -> int:
    results, total = fetch_sources(
        cfg_path=cfg_path,
        source=source,
        use_watermarks=use_watermarks,
        ignore_watermark=ignore_watermark,
        per_source_limit=per_source_limit,
    )

    print(f"Fetched {total} item(s) across {len(results)} source(s)")
    for result in results:
        wm = result.watermark.isoformat() if result.watermark else "-"
        print(
            f"  {result.source.name}: fetched={result.fetched} returned={result.returned} "
            f"type={result.source.type} watermark={wm}"
        )

    if output_path:
        flat_stories: List[Dict[str, object]] = []

        payload_sources = []
        for result in results:
            source_payload_items = []
            for idx, item in enumerate(result.items):
                story_id = f"{result.source.name}-{idx}"
                story_dict = news_item_to_dict(item, story_id=story_id)
                source_payload_items.append(story_dict)
                flat_stories.append(
                    {
                        **story_dict,
                        "source_name": result.source.name,
                        "publisher": result.source.publisher,
                    }
                )

            payload_sources.append(
                {
                    "name": result.source.name,
                    "type": result.source.type,
                    "publisher": result.source.publisher,
                    "items": source_payload_items,
                }
            )

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "sources": payload_sources,
            "stories": flat_stories,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote fetched items to {output_path}")

    if write_supabase and results:
        inserted, updated = _store_to_supabase(results)
        print(f"Stored {inserted} new and {updated} existing items to Supabase.")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg_path = _resolve_config_path(str(getattr(args, "config", "feeds.yaml")))

    try:
        return run_fetch_cli(
            cfg_path=cfg_path,
            source=getattr(args, "source", None),
            use_watermarks=bool(getattr(args, "use_watermarks", False)),
            ignore_watermark=bool(getattr(args, "ignore_watermark", False)),
            per_source_limit=getattr(args, "limit", None),
            output_path=getattr(args, "output", None),
            write_supabase=bool(getattr(args, "write_supabase", False)),
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
