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
import time
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
from src.nfl_news_pipeline.logging import AuditLogger
from src.nfl_news_pipeline.orchestrator import NFLNewsPipeline
from src.nfl_news_pipeline.storage import StorageResult
from src.nfl_news_pipeline.filters.llm import LLMFilter

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
    allow_llm: bool | None = None,
) -> Tuple[List[NewsItem], int, int, List[Dict[str, Any]]]:
    """Filter items for NFL relevance using rule-based first, then LLM if ambiguous.

    Returns: (relevant_items, total_count, kept_count, decisions)
    """
    kept: List[NewsItem] = []
    decisions: List[Dict[str, Any]] = []
    for it in items:
        result, stage = filter_item(it, allow_llm=allow_llm)
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
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run end-to-end pipeline (dry-run; uses orchestrator)",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable LLM usage (rule-based only)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for LLM calls (overrides OPENAI_TIMEOUT)",
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

    # Set up strict dry-run audit logger (in-memory sink, no DB writes)
    class _DemoAuditStorage:
        def __init__(self) -> None:
            self.events: List[Dict[str, Any]] = []

        def add_audit_event(
            self,
            event_type: str,
            *,
            pipeline_run_id: str | None = None,
            source_name: str | None = None,
            message: str | None = None,
            event_data: Dict[str, Any] | None = None,
        ) -> bool:
            self.events.append(
                {
                    "type": event_type,
                    "run": pipeline_run_id,
                    "source": source_name,
                    "message": message,
                    "data": event_data or {},
                }
            )
            return True

    _audit_store = _DemoAuditStorage()
    audit = AuditLogger(storage=_audit_store)

    total_candidates = 0
    total_kept = 0
    all_decisions: List[Dict[str, Any]] = []
    fetched_total = 0
    errors_total = 0
    run_start = time.time()

    # Apply LLM runtime controls
    allow_llm = not args.disable_llm
    if args.llm_timeout is not None:
        # Set env var to let LLMFilter pick it up
        import os
        os.environ["OPENAI_TIMEOUT"] = str(args.llm_timeout)
    if args.verbose:
        import os
        os.environ["NEWS_PIPELINE_DEBUG"] = "1"
        print(f"LLM enabled={allow_llm} timeout={os.environ.get('OPENAI_TIMEOUT', 'default')}s")

    # Pipeline mode: run orchestrator end-to-end with a dry-run storage
    if args.pipeline:
        class _DemoStorage:
            def __init__(self) -> None:
                self.rows: List[Dict[str, Any]] = []
                self.watermarks: Dict[str, datetime] = {}

            def check_duplicate_urls(self, urls):
                return {}

            def store_news_items(self, items):
                # Accumulate and return a StorageResult-like object
                self.rows.extend(items)
                ids = {it.url: f"demo_{i}" for i, it in enumerate(items)}
                return StorageResult(inserted_count=len(items), updated_count=0, errors_count=0, ids_by_url=ids)

            def get_watermark(self, source_name: str):
                return self.watermarks.get(source_name)

            def update_watermark(self, source_name: str, *, last_processed_date: datetime, **kwargs):
                self.watermarks[source_name] = last_processed_date
                return True

        # Set env flag so pipeline's filter_item disables LLM if requested
        import os as _os
        if not allow_llm:
            _os.environ["NEWS_PIPELINE_DISABLE_LLM"] = "1"

        demo_storage = _DemoStorage()
        pipeline = NFLNewsPipeline(str(cfg_path), storage=demo_storage, audit=audit)
        summary = pipeline.run()

        # Print summary and a tiny preview of would-be stored items
        print("\n--- PIPELINE (dry-run) summary ---")
        print(
            f"Sources={summary.sources} fetched={summary.fetched_items} kept={summary.filtered_in} "
            f"inserted={summary.inserted} updated={summary.updated} errors={summary.errors} store_errors={summary.store_errors} "
            f"time={summary.duration_ms}ms"
        )
        if args.persist and demo_storage.rows:
            print(f"Would store {len(demo_storage.rows)} items. Preview:")
            for it in demo_storage.rows[:5]:
                print(f"  • {it.title} -> {it.url}")
        return

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
                fetched_total += len(per_feed)
                audit.log_fetch_end(feed.name, items=len(per_feed), duration_ms=0)
                if args.filter:
                    if args.verbose:
                        print(f"Filtering RSS: {feed.name} ({len(per_feed)} items before)")
                    per_feed, total, kept, decisions = apply_filter(
                        per_feed,
                        args.verbose,
                        collect_decisions=(args.log_decisions or args.persist),
                        allow_llm=allow_llm,
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
                    t0 = time.time()
                    items = sp.fetch_sitemap(feed)
                    dt_ms = int((time.time() - t0) * 1000)
                    if args.verbose:
                        print(f"Fetched {len(items)} items from {feed.name}")
                    fetched_total += len(items)
                    audit.log_fetch_end(feed.name, items=len(items), duration_ms=dt_ms)
                    if args.filter:
                        if args.verbose:
                            print(f"Filtering Sitemap: {feed.name} ({len(items)} items before)")
                        items, total, kept, decisions = apply_filter(
                            items,
                            args.verbose,
                            collect_decisions=(args.log_decisions or args.persist),
                            allow_llm=allow_llm,
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
                    audit.log_error(context=f"fetch {feed.name}", exc=e)
                    errors_total += 1

    # Structured filter and pipeline summaries (always dry-run)
    if args.filter:
        audit.log_filter_summary(candidates=total_candidates, kept=total_kept)

    run_ms = int((time.time() - run_start) * 1000)
    audit.log_pipeline_summary(
        sources=len(rss_feeds) + len(sitemap_feeds),
        fetched_items=fetched_total,
        filtered_in=total_kept if args.filter else 0,
        errors=errors_total,
        duration_ms=run_ms,
    )

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
        print("--- DEMO audit summary (dry-run) ---")
        print(
            f"Sources={len(rss_feeds) + len(sitemap_feeds)} fetched={fetched_total} "
            f"kept={total_kept if args.filter else 0} errors={errors_total} time={run_ms}ms"
        )


if __name__ == "__main__":
    main()
