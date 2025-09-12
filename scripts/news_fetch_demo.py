#!/usr/bin/env python3
"""
Simple demo script: load feeds.yaml, fetch from RSS and Sitemap processors, and print results.

Usage:
  python scripts/news_fetch_demo.py [--config feeds.yaml] [--show 5] [--rss-only | --sitemap-only]

Notes:
- This script performs network requests to the configured feeds.
- It prints the top N items per feed (chronologically descending when possible).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# Make sure repo root is on sys.path so 'src' package resolves when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables (e.g., OPENAI_API_KEY) from repo root .env if present
load_dotenv(str(ROOT / ".env"), override=False)

# Import using the project layout (src/ as top-level package)
from src.nfl_news_pipeline.config import ConfigManager
from src.nfl_news_pipeline.models import FeedConfig, NewsItem
from src.nfl_news_pipeline.processors.rss import RSSProcessor
from src.nfl_news_pipeline.processors.sitemap import SitemapProcessor
from src.nfl_news_pipeline.filters.relevance import filter_item

DEFAULT_LLM_MODEL_ID = "default-model"


def fmt_dt(dt: datetime | None) -> str:
    if not dt:
        return "-"
    try:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(dt)


def print_items(title: str, items: List[NewsItem], show: int) -> None:
    print(f"\n=== {title} (showing up to {show}) ===")
    for i, item in enumerate(items[:show], start=1):
        print(f"{i:>2}. [{fmt_dt(item.publication_date)}] {item.title or item.url}")
        print(f"    URL: {item.url}")
        if item.source_name:
            print(f"    Source: {item.source_name} ({item.publisher or '-'})")


def apply_filter(
    items: List[NewsItem],
    verbose: bool,
    collect_decisions: bool = False,
) -> Tuple[List[NewsItem], int, int, List[Dict[str, Any]]]:
    """Filter items for NFL relevance using rule-based first, then LLM if ambiguous.

    Returns: (relevant_items, total_count, kept_count, decisions)
    """
    kept: List[NewsItem] = []
    decisions: List[Dict[str, Any]] = []
    for it in items:
        result, stage = filter_item(it)
        if verbose:
            print(
                f"      • filter: stage={stage} relevant={result.is_relevant} "
                f"score={result.confidence_score:.2f} reason={result.reasoning}"
            )
        if collect_decisions:
            decisions.append(
                {
                    "url": it.url,
                    "title": it.title,
                    "source_name": it.source_name,
                    "publisher": it.publisher,
                    "publication_date": fmt_dt(it.publication_date),
                    "method": result.method,
                    "stage": stage,
                    "confidence": result.confidence_score,
                    "reasoning": result.reasoning,
                    "model_id": None if stage != "llm" else DEFAULT_LLM_MODEL_ID,
                }
            )
        if result.is_relevant:
            kept.append(it)
    return kept, len(items), len(kept), decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and print news items from RSS and Sitemap feeds")
    parser.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    parser.add_argument("--show", type=int, default=5, help="Max items to show per feed (default: 5)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rss-only", action="store_true", help="Only fetch RSS feeds")
    group.add_argument("--sitemap-only", action="store_true", help="Only fetch Sitemap feeds")
    parser.add_argument("--verbose", action="store_true", help="Print extra details (URLs, counts)")
    parser.add_argument("--filter", action="store_true", help="Apply NFL relevance filtering to results")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="DEMO: Print what would be stored to Supabase (no writes)",
    )
    parser.add_argument(
        "--log-decisions",
        action="store_true",
        help="DEMO: Print structured filter decisions (no writes)",
    )
    args = parser.parse_args()

    # Resolve config path (support running from repo root or scripts/)
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        # Try relative to project root if not found in CWD
        alt = ROOT / args.config
        if alt.is_file():
            cfg_path = alt

    cm = ConfigManager(str(cfg_path))
    cm.load_config()
    defaults = cm.get_defaults()
    feeds = cm.get_enabled_sources()

    rss_feeds: List[FeedConfig] = [f for f in feeds if f.type == "rss"]
    sitemap_feeds: List[FeedConfig] = [f for f in feeds if f.type == "sitemap"]

    now = datetime.now(timezone.utc)

    total_candidates = 0
    total_kept = 0
    all_decisions: List[Dict[str, Any]] = []

    if not args.sitemap_only:
        if not rss_feeds:
            print("No RSS feeds enabled in config.")
        else:
            print("Fetching RSS feeds...")
            rp = RSSProcessor(defaults)
            # Use the synchronous helper which internally runs async
            rss_items: List[NewsItem] = rp.fetch_multiple(rss_feeds)
            # Group by feed for display
            for feed in rss_feeds:
                per_feed = [it for it in rss_items if it.source_name == feed.name]
                # Sort newest first
                per_feed.sort(key=lambda x: x.publication_date or now, reverse=True)
                if args.filter:
                    if args.verbose:
                        print(f"Filtering RSS: {feed.name} ({len(per_feed)} items before)")
                    per_feed, total, kept, decisions = apply_filter(
                        per_feed, args.verbose, collect_decisions=(args.log_decisions or args.persist)
                    )
                    total_candidates += total
                    total_kept += kept
                    if args.log_decisions or args.persist:
                        all_decisions.extend(decisions)
                    if args.verbose:
                        print(f"Filtered RSS: {feed.name} kept {kept}/{total}")
                print_items(f"RSS: {feed.name}", per_feed, args.show)

    if not args.rss_only:
        if not sitemap_feeds:
            print("No Sitemap feeds enabled in config.")
        else:
            print("\nFetching Sitemap feeds...")
            sp = SitemapProcessor(defaults)
            for feed in sitemap_feeds:
                try:
                    if args.verbose:
                        try:
                            constructed = sp.construct_sitemap_url(feed)
                            print(f"Fetching {feed.name} URL: {constructed}")
                        except Exception:
                            pass
                    items = sp.fetch_sitemap(feed)
                    if args.verbose:
                        print(f"Fetched {len(items)} items from {feed.name}")
                    if args.filter:
                        if args.verbose:
                            print(f"Filtering Sitemap: {feed.name} ({len(items)} items before)")
                        items, total, kept, decisions = apply_filter(
                            items, args.verbose, collect_decisions=(args.log_decisions or args.persist)
                        )
                        total_candidates += total
                        total_kept += kept
                        if args.log_decisions or args.persist:
                            all_decisions.extend(decisions)
                        if args.verbose:
                            print(f"Filtered Sitemap: {feed.name} kept {kept}/{total}")
                    # Already sorted in implementation; sort again just in case
                    items.sort(key=lambda x: x.publication_date or now, reverse=True)
                    print_items(f"Sitemap: {feed.name}", items, args.show)
                except Exception as e:
                    print(f"Failed to fetch sitemap for {feed.name}: {e}")

    # Demo-only persistence/logging output (no database writes performed)
    if args.persist or args.log_decisions:
        print("\n--- DEMO persistence summary (dry-run; no DB writes) ---")
        if args.filter:
            print(f"Would persist {total_kept} filtered items out of {total_candidates} candidates")
        else:
            print("Filtering not enabled; persistence preview limited to fetched items shown above")
        if args.log_decisions and all_decisions:
            print(f"Would log {len(all_decisions)} filter decisions (method, stage, confidence, reasoning)")
            if args.verbose:
                # Print a sample of up to 5 decisions
                for d in all_decisions[:5]:
                    print(
                        f"  • {d['source_name']} {d['title']!r} -> method={d['method']} stage={d['stage']} "
                        f"score={d['confidence']:.2f} reason={d['reasoning']}"
                    )


if __name__ == "__main__":
    main()
