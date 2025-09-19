#!/usr/bin/env python3
"""Fetch the latest news URL and run live context extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path when run from nested directories
def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "README.md").exists():
            return p
    return start.parents[0]

ROOT_DIR = _repo_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.db.database_init import get_supabase_client
from src.nfl_news_pipeline.models import ProcessedNewsItem
from src.nfl_news_pipeline.story_grouping import URLContextExtractor


logger = logging.getLogger("live_story_context_test")


def _parse_publication_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Invalid publication_date format: %s", value)
    return datetime.now(timezone.utc)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.warning("raw_metadata is not valid JSON text")
    return {}


def _row_to_news_item(row: Dict[str, Any]) -> ProcessedNewsItem:
    return ProcessedNewsItem(
        url=row.get("url") or "",
        title=row.get("title") or "",
        publication_date=_parse_publication_date(row.get("publication_date")),
        source_name=row.get("source_name") or "",
        publisher=row.get("publisher") or "",
        description=row.get("description"),
        relevance_score=float(row.get("relevance_score") or 0.0),
        filter_method=row.get("filter_method") or "rule_based",
        filter_reasoning=row.get("filter_reasoning"),
        entities=_as_list(row.get("entities")),
        categories=_as_list(row.get("categories")),
        raw_metadata=_as_dict(row.get("raw_metadata")),
    )


async def run_live_test(provider: str = "google", verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase client is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

    logger.info("Fetching latest article from news_urls table")
    response = client.table("news_urls").select("*").order("publication_date", desc=True).limit(1).execute()
    rows = getattr(response, "data", []) or []
    if not rows:
        raise RuntimeError("No rows returned from news_urls table.")

    latest_row = rows[0]
    news_item = _row_to_news_item(latest_row)

    logger.info("Running URL context extraction for %s", news_item.url)
    extractor = URLContextExtractor(preferred_provider=provider, enable_caching=False, verbose=verbose)
    summary = await extractor.extract_context(news_item)

    print("\n=== Live URL Context Extraction Result ===")
    print(f"URL: {news_item.url}")
    print(f"Title: {news_item.title}")
    print(f"Publication Date: {news_item.publication_date.isoformat()}")
    print(f"Model Used: {summary.llm_model}")
    print(f"Fallback Used: {summary.fallback_used}")
    print(f"Confidence: {summary.confidence_score:.2f}")
    print(f"Summary: {summary.summary_text}")
    if summary.entities:
        for key, values in summary.entities.items():
            if values:
                print(f"{key.title()}: {', '.join(values)}")
    if summary.key_topics:
        print(f"Key Topics: {', '.join(summary.key_topics)}")
    if verbose and extractor.last_url_context_metadata:
        print("\n--- URL Context Metadata ---")
        print(json.dumps(extractor.last_url_context_metadata, indent=2, default=str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run live URL context extraction")
    parser.add_argument(
        "--provider",
        choices=["google", "openai"],
        default="google",
        help="LLM provider to use for URL context extraction",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed URL context metadata and enable debug logging",
    )
    args = parser.parse_args()

    asyncio.run(run_live_test(provider=args.provider, verbose=args.verbose))
