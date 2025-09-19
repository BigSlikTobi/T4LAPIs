#!/usr/bin/env python3
"""CLI to run only the URL context extraction stage for new stories."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from .pipeline_cli import _build_storage, _filter_sources
from ._tracking import TrackingStorageAdapter
from ._supabase_helpers import load_context_summaries, write_context_summaries

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
from src.nfl_news_pipeline.models import ProcessedNewsItem, ContextSummary
from src.nfl_news_pipeline.orchestrator import NFLNewsPipeline
from src.nfl_news_pipeline.story_grouping import URLContextExtractor
from src.nfl_news_pipeline.storage.protocols import (
    get_grouping_client,
    has_story_grouping_capability,
)
from src.nfl_news_pipeline.group_manager import GroupStorageManager

# --------------------------------------------------------------------------------------
# Utilities
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


async def extract_context_batch(
    extractor: URLContextExtractor,
    storage: Optional[GroupStorageManager],
    items: Iterable[ProcessedNewsItem],
    url_id_map: Dict[str, str],
    *,
    dry_run: bool = False,
) -> Tuple[int, int, List[str], List[ContextSummary]]:
    """Run context extraction for the provided items.

    Returns:
        processed_count, stored_count, errors, summaries (only returned in dry-run mode)
    """
    processed = 0
    stored = 0
    errors: List[str] = []
    summaries: List[ContextSummary] = []

    for item in items:
        news_url_id = url_id_map.get(item.url)
        if not news_url_id:
            errors.append(f"Missing news_url_id for {item.url}")
            continue

        try:
            summary = await extractor.extract_context(item)
        except Exception as exc:  # pragma: no cover - network failures depend on env
            errors.append(f"{item.url}: {exc}")
            continue

        processed += 1
        summary.news_url_id = news_url_id
        if summary.generated_at is None:
            summary.generated_at = datetime.now(timezone.utc)

        if storage is None or dry_run:
            summaries.append(summary)
            continue

        try:
            stored_ok = await storage.upsert_context_summary(summary)
        except Exception as exc:  # pragma: no cover - storage errors depend on env
            errors.append(f"Failed to store summary for {item.url}: {exc}")
        else:
            if stored_ok:
                stored += 1
            else:
                errors.append(f"Storage declined summary for {item.url}")

    return processed, stored, errors, summaries


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt


def _load_fetch_payload(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    stories = payload.get("stories")
    if stories is None:
        # Fallback: flatten from sources
        stories = []
        for source in payload.get("sources", []):
            for item in source.get("items", []):
                stories.append({
                    **item,
                    "source_name": source.get("name"),
                    "publisher": source.get("publisher"),
                })
    return stories or []


def _story_to_processed(item: Dict[str, object]) -> ProcessedNewsItem:
    publication = _parse_datetime(item.get("publication_date")) or datetime.utcnow()
    return ProcessedNewsItem(
        url=str(item.get("url")),
        title=str(item.get("title") or ""),
        publication_date=publication,
        source_name=str(item.get("source_name") or item.get("source", "unknown")),
        publisher=str(item.get("publisher") or ""),
        description=item.get("description"),
    )


def _build_context_summary_stub(story_id: str, item: Dict[str, object]) -> ContextSummary:
    title = str(item.get("title") or "")
    description = str(item.get("description") or "")
    summary_text = title or description or f"Summary for {item.get('url')}"
    return ContextSummary(
        news_url_id=story_id,
        summary_text=summary_text,
        llm_model="mock-context",
        confidence_score=0.5,
        generated_at=datetime.now(timezone.utc),
        key_topics=[],
    )


def _parse_story_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def dump_context_from_supabase(
    *,
    story_ids: Optional[List[str]],
    limit: Optional[int],
    output_path: Optional[str],
) -> int:
    summaries = load_context_summaries(
        story_ids=story_ids,
        limit=limit,
    )
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stories": [
            {
                "story_id": summary.news_url_id,
                "summary_text": summary.summary_text,
                "confidence_score": summary.confidence_score,
                "llm_model": summary.llm_model,
                "generated_at": summary.generated_at.isoformat() if summary.generated_at else None,
            }
            for summary in summaries
        ],
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {len(payload['stories'])} summaries to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    print(f"Fetched {len(payload['stories'])} summaries from Supabase.")
    return 0


def run_context_from_file(
    *,
    input_path: str,
    output_path: Optional[str],
    mock: bool,
    write_supabase: bool,
) -> int:
    stories = _load_fetch_payload(input_path)
    if not stories:
        print("No stories found in input payload.")
        return 0

    extractor = None if mock else URLContextExtractor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    processed_items: List[ProcessedNewsItem] = []
    url_id_map: Dict[str, str] = {}
    for index, story in enumerate(stories):
        story_id = str(story.get("story_id") or f"story-{index}")
        processed = _story_to_processed(story)
        processed_items.append(processed)
        url_id_map[processed.url] = story_id

    summaries: List[ContextSummary] = []
    errors: List[str] = []
    processed_count = 0

    if mock:
        for index, story in enumerate(stories):
            story_id = str(story.get("story_id") or f"story-{index}")
            summary = _build_context_summary_stub(story_id, story)
            summaries.append(summary)
            processed_count += 1
    else:
        grouping_storage = None
        processed_count, _stored, errors, summaries = asyncio.run(
            extract_context_batch(
                extractor,  # type: ignore[arg-type]
                grouping_storage,
                processed_items,
                url_id_map,
                dry_run=True,
            )
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stories": [
            {
                "story_id": summary.news_url_id,
                "url": next((it.url for it in processed_items if url_id_map[it.url] == summary.news_url_id), None),
                "summary_text": summary.summary_text,
                "confidence_score": summary.confidence_score,
                "llm_model": summary.llm_model,
            }
            for summary in summaries
        ],
        "errors": errors,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote context summaries to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    if write_supabase and summaries:
        stored = asyncio.run(write_context_summaries(summaries))
        print(f"Stored {stored} summaries to Supabase.")

    print(
        "Context extraction summary: "
        f"processed={processed_count} stored=0 errors={len(errors)}"
    )
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")

    return 0 if not errors else 1


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


def run_context_pipeline(
    *,
    cfg_path: str,
    source: Optional[str],
    dry_run: bool,
    disable_llm: bool,
    llm_timeout: Optional[float],
    ignore_watermark: bool,
    write_supabase: bool,
) -> int:
    cm = ConfigManager(cfg_path)
    cm.load_config()

    base_storage = _build_storage(dry_run)
    storage = TrackingStorageAdapter(base_storage)
    audit = AuditLogger(storage=storage)

    selected_source = None
    if source:
        selected_source = _select_source(cm, source)

    env_updates: Dict[str, Optional[str]] = {
        "NEWS_PIPELINE_DISABLE_STORY_GROUPING": "1",
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

    supports_storage = has_story_grouping_capability(storage)
    if not supports_storage and not dry_run:
        print(
            "Context extraction cannot persist summaries because the storage backend "
            "does not provide grouping capabilities. Run with --dry-run or configure Supabase access."
        )
        return 1

    grouping_storage: Optional[GroupStorageManager] = None
    if supports_storage and not dry_run:
        grouping_client = get_grouping_client(storage)
        grouping_storage = GroupStorageManager(grouping_client)

    extractor = URLContextExtractor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    total_processed = 0
    total_stored = 0
    total_errors: List[str] = []
    dry_run_outputs: List[ContextSummary] = []

    for batch in storage.batches:
        if not batch.has_new_material():
            continue
        items_with_ids = list(batch.iter_items_with_ids())
        if not items_with_ids:
            continue

        url_id_map = {item.url: batch.ids_by_url[item.url] for item in items_with_ids}
        processed, stored, errors, summaries = asyncio.run(
            extract_context_batch(
                extractor,
                grouping_storage,
                items_with_ids,
                url_id_map,
                dry_run=dry_run or not supports_storage,
            )
        )
        total_processed += processed
        total_stored += stored
        total_errors.extend(errors)
        dry_run_outputs.extend(summaries)

        label = "dry-run" if dry_run or not supports_storage else "stored"
        print(
            f"Processed {processed} item(s) from '{batch.source_name}' for context extraction "
            f"({label}; errors={len(errors)})."
        )
        if errors:
            for err in errors[:5]:
                print(f"  - {err}")
        if summaries and (dry_run or not supports_storage):
            preview = summaries[0]
            snippet = (preview.summary_text or "").strip()
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            print(f"  Sample summary ({preview.news_url_id}): {snippet}")

    if write_supabase and dry_run_outputs:
        stored_count = asyncio.run(write_context_summaries(dry_run_outputs))
        print(f"Stored {stored_count} summaries to Supabase.")

    if total_processed:
        print(
            "Context extraction summary: "
            f"processed={total_processed} stored={total_stored} errors={len(total_errors)}"
        )
    else:
        print("No new or updated items required context extraction.")

    return 0 if not total_errors else 1


# --------------------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ingestion and URL context extraction")
    parser.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    parser.add_argument("--source", help="Process only a single enabled source by name")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing context summaries")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM-based relevance filtering during ingestion")
    parser.add_argument("--llm-timeout", type=float, help="Override LLM timeout in seconds")
    parser.add_argument("--ignore-watermark", action="store_true", help="Ignore source watermarks for this run")
    parser.add_argument("--input", help="Path to fetch JSON payload for offline context extraction")
    parser.add_argument("--output", help="Destination path for context summaries JSON")
    parser.add_argument("--mock", action="store_true", help="Generate placeholder summaries without calling external services")
    parser.add_argument("--from-supabase", action="store_true", help="Read existing context summaries from Supabase instead of running extraction")
    parser.add_argument("--story-ids", help="Comma-separated news_url_ids to fetch when using --from-supabase")
    parser.add_argument("--limit", type=int, help="Maximum summaries to load from Supabase (default: 50)")
    parser.add_argument("--write-supabase", action="store_true", help="Persist generated summaries to Supabase")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg_path = _resolve_config_path(str(getattr(args, "config", "feeds.yaml")))

    from_supabase = bool(getattr(args, "from_supabase", False))
    story_ids = _parse_story_ids(getattr(args, "story_ids", None))
    limit = getattr(args, "limit", None)
    write_supabase = bool(getattr(args, "write_supabase", False))

    if from_supabase:
        if getattr(args, "input", None):
            parser.error("--input cannot be combined with --from-supabase")
        return dump_context_from_supabase(
            story_ids=story_ids,
            limit=limit,
            output_path=getattr(args, "output", None),
        )

    input_path = getattr(args, "input", None)
    if input_path:
        return run_context_from_file(
            input_path=input_path,
            output_path=getattr(args, "output", None),
            mock=bool(getattr(args, "mock", False)),
            write_supabase=write_supabase,
        )

    try:
        return run_context_pipeline(
            cfg_path=cfg_path,
            source=getattr(args, "source", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            disable_llm=bool(getattr(args, "disable_llm", False)),
            llm_timeout=getattr(args, "llm_timeout", None),
            ignore_watermark=bool(getattr(args, "ignore_watermark", False)),
            write_supabase=write_supabase,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
