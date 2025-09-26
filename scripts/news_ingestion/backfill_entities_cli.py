#!/usr/bin/env python3
"""One-off backfill to populate news_url_entities for historical articles."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    return start.parents[0]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.news_ingestion.pipeline_cli import _build_storage  # noqa: E402

load_dotenv(str(ROOT / ".env"), override=False)

from src.nfl_news_pipeline.entities.enrichment import (  # noqa: E402
    build_entities_extractor,
    enrich_processed_item,
)
from src.nfl_news_pipeline.models import ProcessedNewsItem  # noqa: E402


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def _row_to_processed(row: Dict[str, object]) -> ProcessedNewsItem:
    return ProcessedNewsItem(
        url=str(row.get("url")),
        title=str(row.get("title") or ""),
        publication_date=_parse_datetime(row.get("publication_date")),
        source_name=str(row.get("source_name") or ""),
        publisher=str(row.get("publisher") or ""),
        description=row.get("description"),
        raw_metadata=row.get("raw_metadata") or {},
        relevance_score=float(row.get("relevance_score") or 0.0),
        filter_method=str(row.get("filter_method") or "rule_based"),
        filter_reasoning=row.get("filter_reasoning"),
        entities=list(row.get("entities") or []),
        categories=list(row.get("categories") or []),
    )


def _fetch_batch(client, table: str, offset: int, limit: int) -> List[Dict[str, object]]:
    query = (
        client.table(table)
        .select(
            "id,url,title,description,publication_date,source_name,publisher," \
            "relevance_score,filter_method,filter_reasoning,entities,categories,raw_metadata"
        )
        .order("publication_date", desc=False)
        .range(offset, offset + limit - 1)
    )
    resp = query.execute()
    return getattr(resp, "data", []) or []


def _process_batch(
    rows: Iterable[Dict[str, object]],
    extractor,
    *,
    dry_run: bool,
    storage,
) -> Tuple[int, int, int]:
    items: List[ProcessedNewsItem] = []

    for row in rows:
        url = row.get("url")
        news_id = row.get("id")
        if not url or not news_id:
            continue
        item = _row_to_processed(row)
        enriched = enrich_processed_item(item, extractor)
        tags = (enriched.raw_metadata or {}).get("entity_tags") or {}
        has_data = any(tags.get(k) for k in ("players", "teams", "topics"))
        if not has_data:
            continue
        items.append(enriched)

    if not items:
        return 0, 0, 0

    if dry_run:
        return len(items), 0, 0

    result = storage.store_news_items(items)
    return len(items), result.inserted_count, result.updated_count


def run_backfill(
    *,
    batch_size: int,
    limit: Optional[int],
    dry_run: bool,
    verbose: bool,
) -> int:
    storage = _build_storage(dry_run=False)
    try:
        client = storage.client  # type: ignore[attr-defined]
    except AttributeError:
        raise RuntimeError("Supabase client unavailable; ensure credentials are configured.")

    table = getattr(storage, "table_news", os.getenv("NEWS_URLS_TABLE", "news_urls"))
    if verbose:
        print("Building entity dictionary (this may take a moment)...")
    extractor = build_entities_extractor()
    if extractor is None:
        print("Warning: entity extractor unavailable; team/player resolution may be incomplete.")

    processed = 0
    inserted = 0
    updated = 0
    offset = 0
    remaining = limit

    try:
        while True:
            if remaining is not None and remaining <= 0:
                break
            current_batch = batch_size if remaining is None else min(batch_size, remaining)
            rows = _fetch_batch(client, table, offset, current_batch)
            if not rows:
                break
            count, ins, upd = _process_batch(rows, extractor, dry_run=dry_run, storage=storage)
            processed += count
            inserted += ins
            updated += upd
            if verbose:
                print(
                    f"Batch offset={offset}: fetched={len(rows)} processed={count} "
                    f"inserts={ins} updates={upd}"
                )
            offset += len(rows)
            if remaining is not None:
                remaining -= len(rows)
    except KeyboardInterrupt:
        print("\nBackfill interrupted by user. Partial results reported below.")

    print(
        "Backfill summary: "
        f"processed_items={processed} updates={updated} (inserted new rows={inserted}) "
        f"dry_run={dry_run}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate news_url_entities for historical records")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of rows to process per batch (default: 200)")
    parser.add_argument("--limit", type=int, help="Optional maximum number of rows to process")
    parser.add_argument("--dry-run", action="store_true", help="Compute entities without writing to Supabase")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch progress")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return run_backfill(
        batch_size=max(1, int(getattr(args, "batch_size", 200))),
        limit=getattr(args, "limit", None),
        dry_run=bool(getattr(args, "dry_run", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
