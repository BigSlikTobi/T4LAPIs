#!/usr/bin/env python3
"""Sequential pipeline that chains fetch, context, embedding, similarity, and grouping."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Repository bootstrap (ensure project root is on sys.path before importing local modules)
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

# Import helpers after path/bootstrap so this works both as a module and as a script
from scripts.news_ingestion.pipeline_cli import _build_storage

# --------------------------------------------------------------------------------------
# Stage imports (after sys.path adjustments)
# --------------------------------------------------------------------------------------

from scripts.news_ingestion import fetch_sources_cli
from scripts.news_ingestion import embedding_cli
from scripts.news_ingestion import similarity_cli
from scripts.news_ingestion import grouping_cli
from scripts.news_ingestion.context_extraction_cli import extract_context_batch
from scripts.news_ingestion._supabase_helpers import (
    load_news_without_context,
)

from src.nfl_news_pipeline.group_manager import GroupStorageManager
from src.nfl_news_pipeline.models import ProcessedNewsItem
from src.nfl_news_pipeline.similarity import SimilarityMetric
from src.nfl_news_pipeline.story_grouping import URLContextExtractor
from src.nfl_news_pipeline.storage.protocols import get_grouping_client


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    code: int


@contextmanager
def _temporary_env(key: str, value: Optional[str]):
    previous = os.environ.get(key)
    if value is None:
        yield
        return
    try:
        os.environ[key] = value
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _news_row_to_processed(row: Dict[str, object]) -> Optional[ProcessedNewsItem]:
    url = row.get("url")
    if not url:
        return None
    publication = row.get("publication_date")
    if publication is None:
        from datetime import datetime

        publication = datetime.utcnow()

    return ProcessedNewsItem(
        url=str(url),
        title=str(row.get("title") or ""),
        publication_date=publication,
        source_name=str(row.get("source_name") or ""),
        publisher=str(row.get("publisher") or ""),
        description=row.get("description"),
        raw_metadata=row.get("raw_metadata") or {},
    )


def _ensure_group_storage() -> GroupStorageManager:
    storage = _build_storage(dry_run=False)
    client = get_grouping_client(storage)
    return GroupStorageManager(client)


def _run_fetch_stage(
    *,
    cfg_path: str,
    source: Optional[str],
    use_watermarks: bool,
    ignore_watermark: bool,
) -> StageResult:
    code = fetch_sources_cli.run_fetch_cli(
        cfg_path=cfg_path,
        source=source,
        use_watermarks=use_watermarks,
        ignore_watermark=ignore_watermark,
        per_source_limit=None,
        output_path=None,
        write_supabase=True,
    )
    return StageResult("fetch", code)


def _run_context_stage(
    *,
    context_provider: Optional[str],
    limit: Optional[int],
) -> StageResult:
    # Determine provider: CLI flag > existing env > default 'google'
    provider = (context_provider or os.getenv("URL_CONTEXT_PROVIDER") or "google").strip().lower()
    if provider not in {"google", "openai"}:
        provider = "google"
    print(f"Context stage: using provider '{provider}'")
    summaries_table = os.getenv("CONTEXT_SUMMARIES_TABLE", "context_summaries")
    print(f"Context stage: target summaries table '{summaries_table}'")

    rows = load_news_without_context(limit=limit)
    if not rows:
        print("Context stage: no pending stories (all have summaries).")
        return StageResult("context", 0)

    storage = _ensure_group_storage()
    extractor = URLContextExtractor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        preferred_provider=provider,
    )

    items: List[ProcessedNewsItem] = []
    url_id_map: Dict[str, str] = {}

    for row in rows:
        processed = _news_row_to_processed(row)
        if not processed:
            continue
        news_url_id = str(row.get("id"))
        if not news_url_id:
            continue
        # Avoid duplicate URLs by only keeping the first occurrence
        if processed.url in url_id_map:
            continue
        items.append(processed)
        url_id_map[processed.url] = news_url_id

    if not items:
        print("Context stage: no valid stories available after filtering.")
        return StageResult("context", 0)

    with _temporary_env("URL_CONTEXT_PROVIDER", provider):
        processed, stored, errors, _ = asyncio.run(
            extract_context_batch(
                extractor,
                storage,
                items,
                url_id_map,
                dry_run=False,
            )
        )

    print(
        "Context stage summary: "
        f"processed={processed} stored={stored} errors={len(errors)}"
    )
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")

    return StageResult("context", 0 if not errors else 1)


def _run_embedding_stage(*, limit: Optional[int]) -> StageResult:
    code = embedding_cli.run_embedding_pipeline(
        input_path=None,
        output_path=None,
        mock=False,
        from_supabase=True,
        story_ids=None,
        limit=limit,
        write_supabase=True,
        include_embedded=False,
    )
    return StageResult("embedding", code)


def _run_similarity_stage(
    *,
    threshold: float,
    output_path: str,
    limit: Optional[int],
    write_supabase: bool,
) -> StageResult:
    code = similarity_cli.run_similarity_pipeline(
        input_path=None,
        output_path=output_path,
        threshold=threshold,
        top_k=None,
        metric=SimilarityMetric.COSINE,
        from_supabase=True,
        story_ids=None,
        limit=limit,
        write_supabase=write_supabase,
    )
    return StageResult("similarity", code)


def _run_grouping_stage(
    *,
    similarity_path: str,
    threshold: Optional[float],
    supabase_limit: Optional[int],
    include_all_stories: bool,
    news_limit: Optional[int],
) -> StageResult:
    code = grouping_cli.run_grouping_pipeline(
        similarity_path=similarity_path,
        embeddings_path=None,
        output_path=None,
        threshold=threshold,
        embeddings_from_supabase=True,
        supabase_limit=supabase_limit,
        include_all_stories=include_all_stories,
        news_limit=news_limit,
        write_supabase=True,
    )
    return StageResult("grouping", code)


def run_sequential_pipeline(
    *,
    cfg_path: str,
    source: Optional[str],
    use_watermarks: bool,
    ignore_watermark: bool,
    context_provider: Optional[str],
    context_limit: Optional[int],
    embedding_limit: Optional[int],
    similarity_threshold: float,
    similarity_output: str,
    similarity_limit: Optional[int],
    similarity_write_supabase: bool,
    grouping_threshold: Optional[float],
    grouping_supabase_limit: Optional[int],
    include_all_stories: bool,
    grouping_news_limit: Optional[int],
    skip_fetch: bool,
    skip_context: bool,
    skip_embedding: bool,
    skip_similarity: bool,
    skip_grouping: bool,
) -> int:
    stages: List[StageResult] = []

    if not skip_fetch:
        print("\n[1/5] Fetching sources...")
        result = _run_fetch_stage(
            cfg_path=cfg_path,
            source=source,
            use_watermarks=use_watermarks,
            ignore_watermark=ignore_watermark,
        )
        stages.append(result)
        if result.code != 0:
            return result.code
    else:
        print("\n[1/5] Fetch stage skipped by flag.")

    if not skip_context:
        print("\n[2/5] Extracting URL context...")
        result = _run_context_stage(
            context_provider=context_provider,
            limit=context_limit,
        )
        stages.append(result)
        if result.code != 0:
            return result.code
    else:
        print("\n[2/5] Context stage skipped by flag.")

    if not skip_embedding:
        print("\n[3/5] Generating embeddings...")
        result = _run_embedding_stage(limit=embedding_limit)
        stages.append(result)
        if result.code != 0:
            return result.code
    else:
        print("\n[3/5] Embedding stage skipped by flag.")

    if not skip_similarity:
        print("\n[4/5] Calculating similarities...")
        result = _run_similarity_stage(
            threshold=similarity_threshold,
            output_path=similarity_output,
            limit=similarity_limit,
            write_supabase=similarity_write_supabase,
        )
        stages.append(result)
        if result.code != 0:
            return result.code
    else:
        print("\n[4/5] Similarity stage skipped by flag.")

    if not skip_grouping:
        print("\n[5/5] Forming story groups...")
        result = _run_grouping_stage(
            similarity_path=similarity_output,
            threshold=grouping_threshold,
            supabase_limit=grouping_supabase_limit,
            include_all_stories=include_all_stories,
            news_limit=grouping_news_limit,
        )
        stages.append(result)
        if result.code != 0:
            return result.code
    else:
        print("\n[5/5] Grouping stage skipped by flag.")

    print("\nPipeline finished successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sequential news pipeline (fetch → context → embedding → similarity → grouping)"
    )
    parser.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    parser.add_argument("--source", help="Process only a single enabled source by name")
    parser.add_argument("--use-watermarks", action="store_true", help="Filter fetches using stored watermarks")
    parser.add_argument("--ignore-watermark", action="store_true", help="Ignore watermark even when using --use-watermarks")
    parser.add_argument(
        "--context-provider",
        help="URL context provider (default: 'google'; can also be set via URL_CONTEXT_PROVIDER env var)",
    )
    parser.add_argument("--context-limit", type=int, help="Maximum stories to process during context extraction")
    parser.add_argument("--embedding-limit", type=int, help="Maximum summaries to embed from Supabase")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Similarity threshold (default: 0.8)")
    parser.add_argument("--similarity-output", default="similarities.json", help="Path for similarity output JSON")
    parser.add_argument("--similarity-limit", type=int, help="Maximum embeddings to load when computing similarities")
    parser.add_argument("--similarity-write-supabase", action="store_true", help="Log similarity pairs to Supabase audit log")
    parser.add_argument("--grouping-threshold", type=float, help="Override grouping threshold (defaults to similarity threshold)")
    parser.add_argument("--grouping-embedding-limit", type=int, help="Maximum embeddings to pull from Supabase during grouping")
    parser.add_argument("--include-all-stories", action="store_true", help="Ensure every news_url is represented as a group (singleton fallback)")
    parser.add_argument("--grouping-news-limit", type=int, help="Limit number of news URLs considered when creating singleton groups")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip the fetch stage")
    parser.add_argument("--skip-context", action="store_true", help="Skip the context extraction stage")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip the embedding stage")
    parser.add_argument("--skip-similarity", action="store_true", help="Skip the similarity stage")
    parser.add_argument("--skip-grouping", action="store_true", help="Skip the grouping stage")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_path = str((ROOT / getattr(args, "config", "feeds.yaml")).resolve())
    if Path(cfg_path).exists():
        cfg_path = str(Path(cfg_path))

    grouping_threshold = getattr(args, "grouping_threshold", None)
    if grouping_threshold is None:
        grouping_threshold = getattr(args, "similarity_threshold", 0.8)

    return run_sequential_pipeline(
        cfg_path=cfg_path,
        source=getattr(args, "source", None),
        use_watermarks=bool(getattr(args, "use_watermarks", False)),
        ignore_watermark=bool(getattr(args, "ignore_watermark", False)),
        context_provider=getattr(args, "context_provider", None),
        context_limit=getattr(args, "context_limit", None),
        embedding_limit=getattr(args, "embedding_limit", None),
        similarity_threshold=float(getattr(args, "similarity_threshold", 0.8)),
        similarity_output=str(getattr(args, "similarity_output", "similarities.json")),
        similarity_limit=getattr(args, "similarity_limit", None),
        similarity_write_supabase=bool(getattr(args, "similarity_write_supabase", False)),
        grouping_threshold=grouping_threshold,
        grouping_supabase_limit=getattr(args, "grouping_embedding_limit", None),
        include_all_stories=bool(getattr(args, "include_all_stories", False)),
        grouping_news_limit=getattr(args, "grouping_news_limit", None),
        skip_fetch=bool(getattr(args, "skip_fetch", False)),
        skip_context=bool(getattr(args, "skip_context", False)),
        skip_embedding=bool(getattr(args, "skip_embedding", False)),
        skip_similarity=bool(getattr(args, "skip_similarity", False)),
        skip_grouping=bool(getattr(args, "skip_grouping", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())

