#!/usr/bin/env python3
"""Dry-run the story grouping pipeline without writing to the database."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure repository root is available when the script runs from scripts/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.db.database_init import get_supabase_client
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.embedding.generator import EmbeddingGenerator
from src.nfl_news_pipeline.models import (
    GroupCentroid,
    GroupStatus,
    ProcessedNewsItem,
    StoryEmbedding,
    StoryGroup,
)
from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.story_grouping import URLContextExtractor

logger = logging.getLogger("story_grouping_dry_run")


def _parse_publication_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Unable to parse publication_date '%s', using current UTC time", value)
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
            logger.debug("raw_metadata string is not valid JSON: %s", value)
    return {}


def _row_to_news_item(row: Dict[str, Any]) -> Tuple[str, ProcessedNewsItem]:
    news_url_id = str(row.get("id") or row.get("news_url_id") or row.get("url") or "")
    item = ProcessedNewsItem(
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
    return news_url_id, item


def fetch_all_rows(
    client: Any,
    table: str,
    batch_size: int = 1000,
    limit: Optional[int] = None,
    order_column: str = "publication_date",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start = 0
    total = 0

    while True:
        end = start + batch_size - 1
        query = client.table(table).select("*")
        if order_column:
            query = query.order(order_column, desc=False)
        query = query.range(start, end)
        response = query.execute()
        batch = getattr(response, "data", []) or []
        rows.extend(batch)
        total += len(batch)

        if limit and total >= limit:
            return rows[:limit]
        if len(batch) < batch_size:
            break
        start += batch_size

    return rows


def build_similarity_metric(value: str) -> SimilarityMetric:
    normalized = value.strip().lower()
    if normalized == "cosine":
        return SimilarityMetric.COSINE
    if normalized == "euclidean":
        return SimilarityMetric.EUCLIDEAN
    if normalized == "dot" or normalized == "dot_product":
        return SimilarityMetric.DOT_PRODUCT
    raise argparse.ArgumentTypeError(f"Unsupported similarity metric: {value}")


async def run_dry_run(args: argparse.Namespace) -> None:
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase client is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

    logger.info("Fetching news URLs from %s", args.table)
    rows = fetch_all_rows(client, args.table, batch_size=args.batch_size, limit=args.limit)
    if not rows:
        logger.warning("No rows returned from %s", args.table)
        return

    entries: List[Tuple[str, ProcessedNewsItem]] = []
    for row in rows:
        news_url_id, item = _row_to_news_item(row)
        if not news_url_id:
            logger.warning("Skipping row without identifiable id: %s", row)
            continue
        entries.append((news_url_id, item))

    logger.info("Preparing pipeline components for %d articles", len(entries))
    extractor = URLContextExtractor(
        preferred_provider=args.provider,
        enable_caching=False,
        verbose=args.verbose,
    )

    openai_key = os.getenv("OPENAI_API_KEY") if args.use_openai else None
    embedding_generator = EmbeddingGenerator(
        openai_api_key=openai_key,
        openai_model=args.openai_model,
        sentence_transformer_model=args.sentence_model,
        batch_size=args.embedding_batch_size,
        use_openai_primary=args.use_openai,
    )

    similarity_calculator = SimilarityCalculator(
        similarity_threshold=args.similarity_threshold,
        metric=args.similarity_metric,
    )
    centroid_manager = GroupCentroidManager()

    groups: Dict[str, StoryGroup] = {}
    group_members: Dict[str, List[StoryEmbedding]] = {}
    group_order: List[str] = []
    group_assignments: Dict[str, List[Tuple[str, float]]] = {}

    total_processed = 0
    failures: List[Tuple[str, str]] = []

    for index, (news_url_id, news_item) in enumerate(entries, start=1):
        logger.info("Processing [%d/%d] %s", index, len(entries), news_item.title or news_item.url)

        try:
            summary = await extractor.extract_context(news_item)
            summary.news_url_id = str(news_url_id)
            if not summary.summary_text:
                raise ValueError("Context summary is empty")

            embedding = await embedding_generator.generate_embedding(summary, str(news_url_id))
        except Exception as exc:
            logger.exception("Failed to process news_url_id=%s: %s", news_url_id, exc)
            failures.append((str(news_url_id), str(exc)))
            continue

        existing_centroids: List[GroupCentroid] = []
        for group_id in group_order:
            group = groups[group_id]
            if group.centroid_embedding:
                existing_centroids.append(
                    GroupCentroid(
                        group_id=group.id or group_id,
                        centroid_vector=list(group.centroid_embedding),
                        member_count=group.member_count,
                        last_updated=group.updated_at,
                    )
                )

        best_match: Optional[Tuple[str, float]] = None
        if existing_centroids:
            best_match = similarity_calculator.find_best_matching_group(embedding, existing_centroids)

        if best_match:
            group_id, score = best_match
            group = groups[group_id]
            members = group_members[group_id]
            members.append(embedding)
            update_result = centroid_manager.update_group_centroid(group, members)
            group.status = GroupStatus.UPDATED
            groups[group_id] = group
            group_members[group_id] = members
            group_assignments.setdefault(group_id, []).append((str(news_url_id), score))

            print(f"→ {news_item.title}")
            print(f"   URL: {news_item.url}")
            print(f"   Summary: {summary.summary_text}")
            print(f"   Assigned to group {group_id} (similarity {score:.3f})")
            if update_result.error_message:
                print(f"   Centroid update warning: {update_result.error_message}")
        else:
            group_id = f"group-{len(groups) + 1:03d}"
            now = datetime.now(timezone.utc)
            new_group = StoryGroup(
                member_count=1,
                status=GroupStatus.NEW,
                tags=[],
                centroid_embedding=list(embedding.embedding_vector),
                created_at=now,
                updated_at=now,
                id=group_id,
            )
            groups[group_id] = new_group
            group_members[group_id] = [embedding]
            group_order.append(group_id)
            group_assignments.setdefault(group_id, []).append((str(news_url_id), 1.0))

            print(f"→ {news_item.title}")
            print(f"   URL: {news_item.url}")
            print("   Created new group {0}".format(group_id))
            print(f"   Summary: {summary.summary_text}")

        total_processed += 1
        if args.limit and total_processed >= args.limit:
            break

    print("\n=== Dry-Run Summary ===")
    print(f"Articles processed: {total_processed}")
    print(f"Groups formed: {len(groups)}")
    if failures:
        print(f"Failed items: {len(failures)}")
        for news_url_id, error in failures:
            print(f" - {news_url_id}: {error}")

    for group_id in group_order:
        group = groups[group_id]
        members = group_assignments.get(group_id, [])
        print(f"\nGroup {group_id} — members: {len(members)}, status: {group.status.value}")
        for member_id, score in members[: args.group_preview_limit]:
            print(f"   {member_id} (similarity {score:.3f})")
        if len(members) > args.group_preview_limit:
            print(f"   ... {len(members) - args.group_preview_limit} more")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the story grouping dry-run pipeline")
    parser.add_argument("--table", default=os.getenv("NEWS_URLS_TABLE", "news_urls"), help="Supabase table containing news URLs")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of rows to fetch per request")
    parser.add_argument("--limit", type=int, help="Optional limit for number of articles to process")
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        default="openai",
        help="Preferred LLM provider for context extraction",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose context extractor logging")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI embeddings as the primary method")
    parser.add_argument("--openai-model", default="text-embedding-3-small", help="OpenAI embedding model name")
    parser.add_argument(
        "--sentence-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model to use as fallback/primary",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for group assignment",
    )
    parser.add_argument(
        "--similarity-metric",
        type=build_similarity_metric,
        default="cosine",
        help="Similarity metric to use (cosine, euclidean, dot)",
    )
    parser.add_argument(
        "--group-preview-limit",
        type=int,
        default=5,
        help="Maximum number of members to show per group in the summary",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    args = parser.parse_args(argv)
    if isinstance(args.similarity_metric, str):
        args.similarity_metric = build_similarity_metric(args.similarity_metric)
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_dry_run(args))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
