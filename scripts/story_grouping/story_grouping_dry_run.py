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

# Ensure repository root is importable regardless of nesting depth
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
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.embedding.generator import EmbeddingGenerator
from src.nfl_news_pipeline.models import (
    ContextSummary,
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
from src.nfl_news_pipeline.orchestrator.story_grouping import (
    StoryGroupingOrchestrator,
    StoryGroupingSettings,
    StoryProcessingOutcome,
)

logger = logging.getLogger("story_grouping_dry_run")

class InMemoryGroupStorage:
    """Minimal in-memory implementation of GroupStorageManager for dry runs."""

    def __init__(self) -> None:
        self.groups: Dict[str, StoryGroup] = {}
        self.group_members: Dict[str, List[StoryGroupMember]] = {}
        self.embeddings: Dict[str, StoryEmbedding] = {}
        self.context_summaries: Dict[str, ContextSummary] = {}
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

    async def upsert_context_summary(self, summary: ContextSummary) -> bool:
        summary.validate()
        if summary.generated_at is None:
            summary.generated_at = datetime.now(timezone.utc)
        self.context_summaries[summary.news_url_id] = summary
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

    id_to_item: Dict[str, ProcessedNewsItem] = {
        str(news_url_id): news_item for news_url_id, news_item in entries
    }
    url_id_map: Dict[str, str] = {
        item.url: str(news_url_id) for news_url_id, item in entries if item.url
    }
    story_order: List[str] = [str(news_url_id) for news_url_id, _ in entries]

    settings = StoryGroupingSettings(
        max_parallelism=args.max_parallelism,
        max_candidates=args.max_candidates,
        candidate_similarity_floor=args.candidate_floor,
        max_total_processing_time=args.max_processing_seconds,
        max_stories_per_run=args.max_stories,
        prioritize_recent=not args.disable_recent_priority,
        prioritize_high_relevance=not args.disable_relevance_priority,
        reprocess_existing=args.reprocess_existing,
    )

    orchestrator = StoryGroupingOrchestrator(
        group_manager=group_manager,
        context_extractor=extractor,
        embedding_generator=embedding_generator,
        settings=settings,
    )

    logger.info(
        "Starting orchestration for %d candidate stories (parallelism=%d)",
        len(entries),
        settings.max_parallelism,
    )

    batch_result = await orchestrator.process_batch(
        [item for _, item in entries],
        url_id_map,
    )

    article_info: Dict[str, Dict[str, str]] = {}
    for news_url_id, item in id_to_item.items():
        summary = storage.context_summaries.get(news_url_id)
        article_info[news_url_id] = {
            "title": item.title,
            "url": item.url,
            "summary": summary.summary_text if summary else "",
        }

    outcome_map: Dict[str, StoryProcessingOutcome] = {}
    missing_outcomes: List[StoryProcessingOutcome] = []
    for outcome in batch_result.outcomes:
        if outcome.news_url_id:
            outcome_map[outcome.news_url_id] = outcome
        else:
            missing_outcomes.append(outcome)

    print("\n=== Story Processing Details ===")
    for outcome in missing_outcomes:
        print(
            f"→ {outcome.url or '(unknown url)'}\n   Status: {outcome.status} — Reason: {outcome.reason or 'n/a'}"
        )

    for news_url_id in story_order:
        item = id_to_item.get(news_url_id)
        outcome = outcome_map.get(news_url_id)
        if not item or not outcome:
            continue

        summary = article_info[news_url_id]["summary"]
        print(f"→ {item.title or item.url or news_url_id}")
        if item.url:
            print(f"   URL: {item.url}")
        if summary:
            print(f"   Summary: {summary}")

        if outcome.status == "assigned":
            print(
                "   Assigned to group {0} (similarity {1:.3f})".format(
                    outcome.group_id, outcome.similarity_score or 0.0
                )
            )
        elif outcome.status == "created":
            print(f"   Created new group {outcome.group_id}")
        elif outcome.status == "skipped":
            print(f"   Skipped — reason: {outcome.reason or 'unspecified'}")
        else:
            reason = outcome.reason or outcome.error or "unknown error"
            print(f"   Error — {reason}")

        if outcome.candidate_groups:
            print(f"   Candidate groups evaluated: {', '.join(outcome.candidate_groups)}")

    metrics = batch_result.metrics
    failures = [o for o in batch_result.outcomes if o.status == "error"]

    print("\n=== Orchestrator Metrics ===")
    print(f"Stories considered: {metrics.total_stories}")
    print(f"Stories processed: {metrics.processed_stories}")
    print(f"Stories skipped: {metrics.skipped_stories}")
    print(f"Stories errored: {metrics.errored_stories}")
    print(f"New groups created: {metrics.new_groups_created}")
    print(f"Existing groups updated: {metrics.existing_groups_updated}")
    if metrics.stories_throttled:
        print(f"Stories throttled: {metrics.stories_throttled}")
    print(
        "Candidates evaluated: {0} (avg {1:.2f})".format(
            metrics.candidates_evaluated,
            metrics.average_candidates_per_story,
        )
    )
    print(
        "Timing (ms): context={0}, embedding={1}, similarity={2}, total_run={3}".format(
            metrics.total_context_time_ms,
            metrics.total_embedding_time_ms,
            metrics.total_similarity_time_ms,
            metrics.total_processing_time_ms,
        )
    )
    print(f"Max parallelism observed: {metrics.max_parallelism_observed}")
    print(f"Time budget exhausted: {metrics.time_budget_exhausted}")
    print(f"Run window: {batch_result.started_at} → {batch_result.finished_at}")

    if failures:
        print(f"Failures: {len(failures)}")
        for outcome in failures:
            print(
                f" - {outcome.news_url_id}: {outcome.reason or outcome.error or 'unknown error'}"
            )

    ordered_groups = list(storage.creation_order)
    for group_id in storage.groups.keys():
        if group_id not in ordered_groups:
            ordered_groups.append(group_id)

    print("\n=== Group Summaries ===")
    for group_id in ordered_groups:
        group = storage.groups[group_id]
        members = storage.group_members.get(group_id, [])
        print(
            f"Group {group_id} — members: {len(members)}, status: {group.status.value}"
        )
        for member in members[: args.group_preview_limit]:
            info = article_info.get(member.news_url_id, {})
            title = info.get("title") or "(title unavailable)"
            summary_text = info.get("summary") or ""
            print(
                "   {0} (similarity {1:.3f}) — {2}".format(
                    member.news_url_id,
                    member.similarity_score,
                    title,
                )
            )
            if summary_text:
                print(f"      Summary: {summary_text}")
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
        "--max-parallelism",
        type=int,
        default=4,
        help="Maximum concurrent stories to process when generating context and embeddings",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum number of candidate groups to evaluate per story",
    )
    parser.add_argument(
        "--candidate-floor",
        type=float,
        default=0.35,
        help="Lower bound similarity score for candidate group consideration",
    )
    parser.add_argument(
        "--max-processing-seconds",
        type=float,
        help="Optional time budget in seconds for processing the entire batch",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        help="Optional limit on stories processed after prioritization",
    )
    parser.add_argument(
        "--disable-recent-priority",
        action="store_true",
        help="Disable recency bias when ordering stories for processing",
    )
    parser.add_argument(
        "--disable-relevance-priority",
        action="store_true",
        help="Disable relevance score bias when ordering stories for processing",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Reprocess stories even if embeddings already exist",
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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to story grouping configuration YAML file (overrides individual arguments)",
    )
    args = parser.parse_args(argv)
    if isinstance(args.similarity_metric, str):
        args.similarity_metric = build_similarity_metric(args.similarity_metric)
    
    # Load configuration file if provided
    if args.config:
        try:
            # Add the import only when needed to avoid dependency issues
            sys.path.insert(0, str(ROOT_DIR / "src"))
            from nfl_news_pipeline.story_grouping_config import StoryGroupingConfigManager
            
            logger.info(f"Loading configuration from {args.config}")
            config_manager = StoryGroupingConfigManager(args.config)
            config = config_manager.load_config()
            
            # Override args with config values (args take precedence over config)
            if not hasattr(args, 'provider') or args.provider == "openai":
                args.provider = config.llm.provider
            if args.similarity_threshold == 0.8:  # default value
                args.similarity_threshold = config.similarity.threshold
            # If embedding model refers to an OpenAI embedding, route to openai_model,
            # otherwise assume it's a sentence-transformers model name.
            if args.sentence_model == "all-MiniLM-L6-v2":  # default value
                if str(config.embedding.model_name).startswith("text-embedding-"):
                    args.openai_model = config.embedding.model_name
                else:
                    args.sentence_model = config.embedding.model_name
            if args.embedding_batch_size == 32:  # default value
                args.embedding_batch_size = config.embedding.batch_size
            if args.max_parallelism == 4:  # default value
                args.max_parallelism = config.performance.max_parallelism
            if args.max_candidates == 8:  # default value
                args.max_candidates = config.similarity.max_candidates
            if args.candidate_floor == 0.35:  # default value
                args.candidate_floor = config.similarity.candidate_similarity_floor
            if args.max_group_size == 50:  # default value
                args.max_group_size = config.grouping.max_group_size
            if args.log_level == "INFO":  # default value
                args.log_level = config.monitoring.log_level
                
            logger.info("Configuration loaded successfully from YAML file")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {args.config}: {e}")
            logger.info("Continuing with command line arguments")
    
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_dry_run(args))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
