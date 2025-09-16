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
from uuid import uuid4

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
    StoryGroupMember,
)
from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.story_grouping import URLContextExtractor
from src.nfl_news_pipeline.group_manager import GroupManager

logger = logging.getLogger("story_grouping_dry_run")


class InMemoryGroupStorage:
    """Minimal in-memory implementation of GroupStorageManager for dry runs."""

    def __init__(self) -> None:
        self.groups: Dict[str, StoryGroup] = {}
        self.group_members: Dict[str, List[StoryGroupMember]] = {}
        self.embeddings: Dict[str, StoryEmbedding] = {}
        self.creation_order: List[str] = []

        # Attributes referenced by GroupManager APIs
        self.client = None
        self.table_embeddings = "in_memory_story_embeddings"
        self.table_groups = "in_memory_story_groups"
        self.table_members = "in_memory_story_group_members"
        self.table_summaries = "in_memory_context_summaries"

    async def store_embedding(self, embedding: StoryEmbedding) -> bool:
        embedding.validate()
        self.embeddings[embedding.news_url_id] = embedding
        return True

    async def store_group(self, group: StoryGroup) -> bool:
        if not group.id:
            group.id = str(uuid4())
        group.validate()
        if group.id not in self.groups:
            self.creation_order.append(group.id)
        self.groups[group.id] = group
        return True

    async def add_member_to_group(self, group_id: str, news_url_id: str, similarity_score: float) -> bool:
        group = self.groups.get(group_id)
        if not group:
            return False

        member = StoryGroupMember(
            group_id=group_id,
            news_url_id=news_url_id,
            similarity_score=float(similarity_score),
            added_at=datetime.now(timezone.utc),
        )
        member.validate()

        members = self.group_members.setdefault(group_id, [])
        members.append(member)

        group.member_count = len(members)
        group.updated_at = datetime.now(timezone.utc)
        self.groups[group_id] = group
        return True

    async def get_group_centroids(self) -> List[GroupCentroid]:
        centroids: List[GroupCentroid] = []
        for group in self.groups.values():
            if group.centroid_embedding:
                centroids.append(
                    GroupCentroid(
                        group_id=group.id or "",
                        centroid_vector=list(group.centroid_embedding),
                        member_count=group.member_count,
                        last_updated=group.updated_at,
                    )
                )
        return centroids

    async def get_group_embeddings(self, group_id: str) -> List[StoryEmbedding]:
        members = self.group_members.get(group_id, [])
        embeddings: List[StoryEmbedding] = []
        for member in members:
            embedding = self.embeddings.get(member.news_url_id)
            if embedding:
                embeddings.append(embedding)
        return embeddings

    async def get_group_by_id(self, group_id: str) -> Optional[StoryGroup]:
        return self.groups.get(group_id)

    async def update_group_status(self, group_id: str, status: GroupStatus) -> bool:
        group = self.groups.get(group_id)
        if not group:
            return False
        group.status = status
        group.updated_at = datetime.now(timezone.utc)
        self.groups[group_id] = group
        return True

    async def get_embedding_by_url_id(self, news_url_id: str) -> Optional[StoryEmbedding]:
        return self.embeddings.get(news_url_id)

    async def check_membership_exists(self, group_id: str, news_url_id: str) -> bool:
        members = self.group_members.get(group_id, [])
        return any(member.news_url_id == news_url_id for member in members)

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
    storage = InMemoryGroupStorage()
    group_manager = GroupManager(
        storage_manager=storage,
        similarity_calculator=similarity_calculator,
        centroid_manager=centroid_manager,
        similarity_threshold=args.similarity_threshold,
        max_group_size=args.max_group_size,
    )

    total_processed = 0
    failures: List[Tuple[str, str]] = []
    article_info: Dict[str, Dict[str, str]] = {}

    for index, (news_url_id, news_item) in enumerate(entries, start=1):
        logger.info("Processing [%d/%d] %s", index, len(entries), news_item.title or news_item.url)

        try:
            summary = await extractor.extract_context(news_item)
            summary.news_url_id = str(news_url_id)
            if not summary.summary_text:
                raise ValueError("Context summary is empty")

            embedding = await embedding_generator.generate_embedding(summary, str(news_url_id))
            await storage.store_embedding(embedding)
        except Exception as exc:
            logger.exception("Failed to process news_url_id=%s: %s", news_url_id, exc)
            failures.append((str(news_url_id), str(exc)))
            continue

        article_id = str(news_url_id)
        article_info[article_id] = {
            "title": news_item.title,
            "url": news_item.url,
            "summary": summary.summary_text,
        }

        assignment = await group_manager.process_new_story(embedding)

        print(f"→ {news_item.title}")
        print(f"   URL: {news_item.url}")
        print(f"   Summary: {summary.summary_text}")

        if assignment.assignment_successful and assignment.group_id:
            group_id = assignment.group_id
            group = await storage.get_group_by_id(group_id)
            if group and group_id not in storage.creation_order:
                storage.creation_order.append(group_id)

            if assignment.is_new_group:
                print(f"   Created new group {group_id}")
            else:
                print(
                    "   Assigned to group {0} (similarity {1:.3f})".format(
                        group_id, assignment.similarity_score
                    )
                )
        else:
            reason = assignment.error_message or "Unknown error during grouping"
            print(f"   Group assignment failed: {reason}")
            failures.append((article_id, reason))

        total_processed += 1
        if args.limit and total_processed >= args.limit:
            break

    print("\n=== Dry-Run Summary ===")
    print(f"Articles processed: {total_processed}")
    print(f"Groups formed: {len(storage.groups)}")
    if failures:
        print(f"Failed items: {len(failures)}")
        seen_failures = set()
        for news_url_id, error in failures:
            key = (news_url_id, error)
            if key in seen_failures:
                continue
            seen_failures.add(key)
            print(f" - {news_url_id}: {error}")

    ordered_groups = list(storage.creation_order)
    for group_id in storage.groups.keys():
        if group_id not in ordered_groups:
            ordered_groups.append(group_id)

    for group_id in ordered_groups:
        group = storage.groups[group_id]
        members = storage.group_members.get(group_id, [])
        print(
            f"\nGroup {group_id} — members: {len(members)}, status: {group.status.value}"
        )
        for member in members[: args.group_preview_limit]:
            info = article_info.get(member.news_url_id, {})
            title = info.get("title") or "(title unavailable)"
            print(
                "   {0} (similarity {1:.3f}) — {2}".format(
                    member.news_url_id, member.similarity_score, title
                )
            )
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
        "--max-group-size",
        type=int,
        default=50,
        help="Maximum number of stories allowed per group before forcing a new group",
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
