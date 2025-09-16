"""Incremental story grouping orchestrator.

Implements Tasks 8.1, 8.2, and 8.3 by coordinating context extraction,
embedding generation, candidate similarity search, and group assignment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..embedding import EmbeddingErrorHandler, EmbeddingGenerator
from ..group_manager import GroupManager
from ..models import ContextSummary, GroupCentroid, ProcessedNewsItem, StoryEmbedding
from ..similarity import SimilarityCalculator
from ..story_grouping import URLContextExtractor
from ..monitoring import GroupingMetricsCollector, GroupingAlertsManager, GroupingAnalytics
from ..logging.audit import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class StoryGroupingSettings:
    """Configuration for story grouping orchestration."""

    max_parallelism: int = 4
    max_candidates: int = 8
    candidate_similarity_floor: float = 0.35
    max_total_processing_time: Optional[float] = None  # seconds
    max_stories_per_run: Optional[int] = None
    prioritize_recent: bool = True
    prioritize_high_relevance: bool = True
    reprocess_existing: bool = False

    def validate(self) -> None:
        if self.max_parallelism < 1:
            raise ValueError("max_parallelism must be >= 1")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if not (0.0 <= self.candidate_similarity_floor <= 1.0):
            raise ValueError("candidate_similarity_floor must be in [0.0, 1.0]")
        if self.max_total_processing_time is not None and self.max_total_processing_time <= 0:
            raise ValueError("max_total_processing_time must be positive")
        if self.max_stories_per_run is not None and self.max_stories_per_run <= 0:
            raise ValueError("max_stories_per_run must be positive")


@dataclass
class StoryProcessingOutcome:
    """Result of processing a single story."""

    news_url_id: str
    url: str
    status: str  # 'assigned', 'created', 'skipped', 'error'
    group_id: Optional[str] = None
    similarity_score: Optional[float] = None
    created_new_group: bool = False
    reason: Optional[str] = None
    candidate_groups: List[str] = field(default_factory=list)
    summary_confidence: Optional[float] = None
    embedding_generated: bool = False
    error: Optional[str] = None
    processing_time_ms: int = 0
    context_time_ms: int = 0
    embedding_time_ms: int = 0
    similarity_time_ms: int = 0


@dataclass
class StoryGroupingMetrics:
    """Aggregated metrics for a batch processing run."""

    total_stories: int = 0
    processed_stories: int = 0
    skipped_stories: int = 0
    errored_stories: int = 0
    new_groups_created: int = 0
    existing_groups_updated: int = 0
    stories_throttled: int = 0
    candidates_evaluated: int = 0
    average_candidates_per_story: float = 0.0
    total_context_time_ms: int = 0
    total_embedding_time_ms: int = 0
    total_similarity_time_ms: int = 0
    total_processing_time_ms: int = 0
    max_parallelism_observed: int = 0
    time_budget_exhausted: bool = False


@dataclass
class StoryGroupingBatchResult:
    """Full result of running the orchestrator on a batch of stories."""

    outcomes: List[StoryProcessingOutcome]
    metrics: StoryGroupingMetrics
    started_at: datetime
    finished_at: datetime


class _ConcurrencyTracker:
    """Track in-flight workers to produce accurate concurrency metrics."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active = 0
        self.max_active = 0

    async def worker_started(self) -> None:
        async with self._lock:
            self._active += 1
            if self._active > self.max_active:
                self.max_active = self._active

    async def worker_finished(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)


class StoryGroupingOrchestrator:
    """Coordinate incremental story grouping end-to-end."""

    def __init__(
        self,
        *,
        group_manager: GroupManager,
        context_extractor: URLContextExtractor,
        embedding_generator: EmbeddingGenerator,
        error_handler: Optional[EmbeddingErrorHandler] = None,
        settings: Optional[StoryGroupingSettings] = None,
        audit_logger: Optional[AuditLogger] = None,
        enable_monitoring: bool = True,
    ) -> None:
        self.group_manager = group_manager
        self.context_extractor = context_extractor
        self.embedding_generator = embedding_generator
        self.error_handler = error_handler or EmbeddingErrorHandler()
        self.settings = settings or StoryGroupingSettings()
        self.settings.validate()

        # Reuse the similarity calculator from the group manager so thresholds stay aligned
        self.similarity_calc: SimilarityCalculator = group_manager.similarity_calc
        
        # Initialize monitoring and analytics
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.metrics_collector = GroupingMetricsCollector(audit_logger)
            self.alerts_manager = GroupingAlertsManager(self.metrics_collector)
            self.analytics = GroupingAnalytics(self.metrics_collector)
            
            # Set up alert callbacks
            self.alerts_manager.add_alert_callback(self._handle_alert)
        else:
            self.metrics_collector = None
            self.alerts_manager = None
            self.analytics = None

    async def process_batch(
        self,
        stories: Sequence[ProcessedNewsItem],
        url_id_map: Dict[str, str],
    ) -> StoryGroupingBatchResult:
        """Process a batch of stories and assign them to groups."""
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        outcomes: List[StoryProcessingOutcome] = []
        metrics = StoryGroupingMetrics(total_stories=len(stories))
        throttled = False

        prioritized = self._prioritize_items(stories, url_id_map)
        missing_id_outcomes = [
            StoryProcessingOutcome(
                news_url_id="",
                url=item.url,
                status="skipped",
                reason="missing_news_url_id",
            )
            for item in prioritized
            if item.url not in url_id_map
        ]
        outcomes.extend(missing_id_outcomes)

        # Filter to items we can process (have IDs)
        candidates = [item for item in prioritized if item.url in url_id_map]

        # Apply max stories limit (resource prioritization)
        if self.settings.max_stories_per_run is not None and len(candidates) > self.settings.max_stories_per_run:
            throttled = True
            to_process = candidates[: self.settings.max_stories_per_run]
            for item in candidates[self.settings.max_stories_per_run :]:
                outcomes.append(
                    StoryProcessingOutcome(
                        news_url_id=url_id_map.get(item.url, ""),
                        url=item.url,
                        status="skipped",
                        reason="max_stories_limit",
                    )
                )
        else:
            to_process = candidates

        # Respect time budget by tracking deadline
        deadline: Optional[float] = None
        if self.settings.max_total_processing_time is not None:
            deadline = start_time + self.settings.max_total_processing_time

        semaphore = asyncio.Semaphore(self.settings.max_parallelism)
        tracker = _ConcurrencyTracker()

        async def worker(item: ProcessedNewsItem) -> StoryProcessingOutcome:
            if deadline is not None and time.perf_counter() >= deadline:
                return StoryProcessingOutcome(
                    news_url_id=url_id_map.get(item.url, ""),
                    url=item.url,
                    status="skipped",
                    reason="time_budget_exhausted",
                )

            async with semaphore:
                await tracker.worker_started()
                try:
                    return await self._process_story(
                        item,
                        url_id_map[item.url],
                    )
                finally:
                    await tracker.worker_finished()

        tasks: List[asyncio.Task[StoryProcessingOutcome]] = []
        for idx, item in enumerate(to_process):
            if deadline is not None and time.perf_counter() >= deadline:
                throttled = True
                metrics.time_budget_exhausted = True
                for remaining in to_process[idx:]:
                    outcomes.append(
                        StoryProcessingOutcome(
                            news_url_id=url_id_map.get(remaining.url, ""),
                            url=remaining.url,
                            status="skipped",
                            reason="time_budget_exhausted",
                        )
                    )
                break
            tasks.append(asyncio.create_task(worker(item)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            outcomes.extend(results)

        finished_at = datetime.now(timezone.utc)
        metrics.max_parallelism_observed = tracker.max_active
        metrics.total_processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        self._aggregate_metrics(metrics, outcomes)
        if throttled or metrics.time_budget_exhausted:
            metrics.time_budget_exhausted = metrics.time_budget_exhausted or throttled

        # Update monitoring after batch completion
        if self.enable_monitoring and self.metrics_collector:
            # Update quality metrics
            self.metrics_collector.update_quality_metrics()
            
            # Check for alerts
            if self.alerts_manager:
                new_alerts = self.alerts_manager.check_thresholds()
                self.alerts_manager.auto_resolve_alerts()
                
                if new_alerts:
                    logger.info(f"Generated {len(new_alerts)} new alerts after batch processing")

        return StoryGroupingBatchResult(outcomes=outcomes, metrics=metrics, started_at=started_at, finished_at=finished_at)

    def _prioritize_items(
        self,
        stories: Sequence[ProcessedNewsItem],
        url_id_map: Dict[str, str],
    ) -> List[ProcessedNewsItem]:
        """Sort stories based on configured prioritization strategy."""
        def sort_key(item: ProcessedNewsItem) -> Tuple[float, float, float]:
            relevance = float(item.relevance_score or 0.0) if self.settings.prioritize_high_relevance else 0.0
            timestamp = item.publication_date.timestamp() if (self.settings.prioritize_recent and item.publication_date) else 0.0
            # Items without IDs lose priority to ensure we skip quickly
            has_id = 1.0 if item.url in url_id_map else 0.0
            return (has_id, relevance, timestamp)

        return sorted(stories, key=sort_key, reverse=True)

    async def _process_story(
        self,
        news_item: ProcessedNewsItem,
        news_url_id: str,
    ) -> StoryProcessingOutcome:
        story_start = time.perf_counter()
        context_time = 0.0
        embedding_time = 0.0
        similarity_time = 0.0
        summary: Optional[ContextSummary] = None
        embedding: Optional[StoryEmbedding] = None
        candidate_groups: List[str] = []

        if not self.settings.reprocess_existing:
            existing_embedding = await self.group_manager.storage.get_embedding_by_url_id(news_url_id)
            if existing_embedding:
                return StoryProcessingOutcome(
                    news_url_id=news_url_id,
                    url=news_item.url,
                    status="skipped",
                    reason="embedding_exists",
                )

        try:
            # Extract context summary
            t0 = time.perf_counter()
            summary = await self.context_extractor.extract_context(news_item)
            context_time = time.perf_counter() - t0
            summary.news_url_id = news_url_id
            stored = await self.group_manager.storage.upsert_context_summary(summary)
            if not stored:
                logger.warning("Failed to persist context summary for %s", news_url_id)

            # Generate embedding with retry support
            t0 = time.perf_counter()
            try:
                embedding = await self.embedding_generator.generate_embedding(summary, news_url_id)
            except Exception as gen_error:  # pragma: no cover - error path exercised in tests via handler
                embedding = await self.error_handler.handle_embedding_error(
                    gen_error,
                    summary,
                    news_url_id,
                    self.embedding_generator,
                    attempt=0,
                )
            embedding_time = time.perf_counter() - t0

            if embedding is None:
                return StoryProcessingOutcome(
                    news_url_id=news_url_id,
                    url=news_item.url,
                    status="error",
                    reason="embedding_failed",
                    summary_confidence=summary.confidence_score,
                    processing_time_ms=int((time.perf_counter() - story_start) * 1000),
                    context_time_ms=int(context_time * 1000),
                    embedding_time_ms=int(embedding_time * 1000),
                )

            # Persist embedding early so downstream operations have access
            try:
                stored_embedding = await self.group_manager.storage.store_embedding(embedding)
                if not stored_embedding:
                    logger.warning("Failed to persist embedding for %s", news_url_id)
            except Exception as store_exc:  # pragma: no cover - storage failures handled per-run
                logger.error("Error storing embedding for %s: %s", news_url_id, store_exc)

            # Load the latest centroids for this story to avoid using a stale snapshot
            centroids, centroid_vectors = await self._load_centroids()

            # Select candidate groups using centroid similarity
            t0 = time.perf_counter()
            candidate_pairs = self._select_candidate_groups(embedding, centroids, centroid_vectors)
            candidate_groups = [centroid.group_id for centroid, _ in candidate_pairs]

            # Perform detailed similarity on candidate groups
            best_group_id, best_similarity = await self._evaluate_candidates(embedding, candidate_pairs)
            similarity_time = time.perf_counter() - t0

            # Assign to existing group if similarity passes threshold
            if best_group_id and best_similarity is not None and self.similarity_calc.is_similar(best_similarity):
                assigned = await self.group_manager.add_story_to_group(best_group_id, embedding, best_similarity)
                status = "assigned" if assigned else "error"
                result = StoryProcessingOutcome(
                    news_url_id=news_url_id,
                    url=news_item.url,
                    status=status,
                    group_id=best_group_id,
                    similarity_score=best_similarity,
                    created_new_group=False,
                    candidate_groups=candidate_groups,
                    summary_confidence=summary.confidence_score,
                    embedding_generated=True,
                    processing_time_ms=int((time.perf_counter() - story_start) * 1000),
                    context_time_ms=int(context_time * 1000),
                    embedding_time_ms=int(embedding_time * 1000),
                    similarity_time_ms=int(similarity_time * 1000),
                )
                if status == "error":
                    result.reason = "group_assignment_failed"
                
                # Record monitoring data
                self._record_decision_metrics(result, summary)
                return result

            # Otherwise create a new group
            group_id = await self.group_manager.create_new_group(embedding)
            if group_id:
                result = StoryProcessingOutcome(
                    news_url_id=news_url_id,
                    url=news_item.url,
                    status="created",
                    group_id=group_id,
                    similarity_score=1.0,
                    created_new_group=True,
                    candidate_groups=candidate_groups,
                    summary_confidence=summary.confidence_score,
                    embedding_generated=True,
                    processing_time_ms=int((time.perf_counter() - story_start) * 1000),
                    context_time_ms=int(context_time * 1000),
                    embedding_time_ms=int(embedding_time * 1000),
                    similarity_time_ms=int(similarity_time * 1000),
                )
                
                # Record monitoring data
                self._record_decision_metrics(result, summary)
                return result

            return StoryProcessingOutcome(
                news_url_id=news_url_id,
                url=news_item.url,
                status="error",
                reason="group_creation_failed",
                candidate_groups=candidate_groups,
                summary_confidence=summary.confidence_score,
                embedding_generated=True,
                processing_time_ms=int((time.perf_counter() - story_start) * 1000),
                context_time_ms=int(context_time * 1000),
                embedding_time_ms=int(embedding_time * 1000),
                similarity_time_ms=int(similarity_time * 1000),
            )

        except Exception as exc:  # pragma: no cover - ensures robust error reporting in prod
            logger.exception("Error processing story %s", news_item.url)
            result = StoryProcessingOutcome(
                news_url_id=news_url_id,
                url=news_item.url,
                status="error",
                reason="unexpected_error",
                error=str(exc),
                candidate_groups=candidate_groups,
                summary_confidence=summary.confidence_score if summary else None,
                embedding_generated=embedding is not None,
                processing_time_ms=int((time.perf_counter() - story_start) * 1000),
                context_time_ms=int(context_time * 1000),
                embedding_time_ms=int(embedding_time * 1000),
                similarity_time_ms=int(similarity_time * 1000),
            )
            
            # Record monitoring data for errors too
            self._record_decision_metrics(result, summary)
            return result

    async def _load_centroids(self) -> Tuple[List[GroupCentroid], List[np.ndarray]]:
        """Load centroid records and corresponding numpy vectors once per batch."""
        try:
            centroids = await self.group_manager.storage.get_group_centroids()
            vectors = [np.array(c.centroid_vector, dtype=np.float32) for c in centroids]
            return centroids, vectors
        except Exception as exc:
            logger.error("Failed to load group centroids: %s", exc)
            return [], []

    def _select_candidate_groups(
        self,
        embedding: StoryEmbedding,
        centroids: Sequence[GroupCentroid],
        centroid_vectors: Sequence[np.ndarray],
    ) -> List[Tuple[GroupCentroid, float]]:
        if not centroids:
            return []

        embedding_vector = np.array(embedding.embedding_vector, dtype=np.float32)
        similarities = self.similarity_calc.batch_calculate_similarities(embedding_vector, list(centroid_vectors))

        candidates: List[Tuple[GroupCentroid, float]] = []
        for centroid, score in zip(centroids, similarities):
            if score >= self.settings.candidate_similarity_floor:
                candidates.append((centroid, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        if self.settings.max_candidates is not None:
            candidates = candidates[: self.settings.max_candidates]
        return candidates

    async def _evaluate_candidates(
        self,
        embedding: StoryEmbedding,
        candidate_pairs: Sequence[Tuple[GroupCentroid, float]],
    ) -> Tuple[Optional[str], Optional[float]]:
        if not candidate_pairs:
            return None, None

        embedding_vector = np.array(embedding.embedding_vector, dtype=np.float32)

        async def compute_similarity(centroid: GroupCentroid, centroid_score: float) -> Tuple[str, float]:
            try:
                group_embeddings = await self.group_manager.storage.get_group_embeddings(centroid.group_id)
                if not group_embeddings:
                    return centroid.group_id, centroid_score

                best = 0.0
                for member in group_embeddings:
                    member_vector = np.array(member.embedding_vector, dtype=np.float32)
                    score = self.similarity_calc.calculate_similarity(embedding_vector, member_vector)
                    if score > best:
                        best = score
                return centroid.group_id, max(best, centroid_score)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error("Failed evaluating candidate group %s: %s", centroid.group_id, exc)
                return centroid.group_id, centroid_score

        tasks = [compute_similarity(centroid, score) for centroid, score in candidate_pairs]
        results = await asyncio.gather(*tasks)

        best_group_id: Optional[str] = None
        best_score: float = 0.0
        for group_id, score in results:
            if score > best_score:
                best_group_id = group_id
                best_score = score
        if best_group_id:
            return best_group_id, best_score
        return None, None

    def _aggregate_metrics(self, metrics: StoryGroupingMetrics, outcomes: Iterable[StoryProcessingOutcome]) -> None:
        processed = [o for o in outcomes if o.status in {"assigned", "created"}]
        skipped = [o for o in outcomes if o.status == "skipped"]
        errored = [o for o in outcomes if o.status == "error"]

        metrics.processed_stories = len(processed)
        metrics.skipped_stories = len(skipped)
        metrics.errored_stories = len(errored)
        metrics.new_groups_created = sum(1 for o in processed if o.created_new_group)
        metrics.existing_groups_updated = sum(1 for o in processed if not o.created_new_group)

        metrics.total_context_time_ms = sum(o.context_time_ms for o in outcomes)
        metrics.total_embedding_time_ms = sum(o.embedding_time_ms for o in outcomes)
        metrics.total_similarity_time_ms = sum(o.similarity_time_ms for o in outcomes)

        candidate_counts = [len(o.candidate_groups) for o in processed if o.candidate_groups]
        metrics.candidates_evaluated = sum(candidate_counts)
        if candidate_counts:
            metrics.average_candidates_per_story = sum(candidate_counts) / len(candidate_counts)

        metrics.stories_throttled += sum(1 for o in skipped if o.reason in {"time_budget_exhausted", "max_stories_limit"})

    def _record_decision_metrics(self, outcome: StoryProcessingOutcome, summary: Optional[ContextSummary] = None) -> None:
        """Record metrics and monitoring data for a story processing outcome."""
        if not self.enable_monitoring or not self.metrics_collector:
            return
        
        # Record the grouping decision
        decision_type = "skipped"
        if outcome.status == "assigned":
            decision_type = "existing_group"
        elif outcome.status == "created":
            decision_type = "new_group"
        
        self.metrics_collector.record_grouping_decision(
            story_id=outcome.news_url_id,
            decision_type=decision_type,
            similarity_score=outcome.similarity_score,
            group_id=outcome.group_id,
            candidate_count=len(outcome.candidate_groups),
            processing_time_ms=outcome.processing_time_ms,
            reason=outcome.reason,
        )
        
        # Record performance metrics
        self.metrics_collector.record_performance_metrics(
            context_time_ms=outcome.context_time_ms,
            embedding_time_ms=outcome.embedding_time_ms,
            similarity_time_ms=outcome.similarity_time_ms,
            total_time_ms=outcome.processing_time_ms,
        )
        
        # Record LLM costs if available from summary
        if summary and hasattr(summary, 'llm_model') and hasattr(summary, 'tokens_used'):
            # Estimate cost based on model (would be more accurate with actual API response)
            estimated_cost = self._estimate_llm_cost(summary.llm_model, getattr(summary, 'tokens_used', 0))
            if estimated_cost > 0:
                self.metrics_collector.record_llm_cost(
                    model=summary.llm_model,
                    tokens=getattr(summary, 'tokens_used', 0),
                    cost=estimated_cost,
                    operation="context_extraction",
                )

    def _estimate_llm_cost(self, model: str, tokens: int) -> float:
        """Estimate LLM API cost based on model and token count."""
        # Rough cost estimates per 1K tokens (would be configured externally in practice)
        cost_per_1k_tokens = {
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0005,
            "gemini-2.5-lite": 0.0001,
        }
        
        rate = cost_per_1k_tokens.get(model, 0.0001)  # Default fallback
        return (tokens / 1000) * rate

    def _handle_alert(self, alert) -> None:
        """Handle monitoring alerts."""
        logger.warning(
            f"Story grouping alert: {alert.title} - {alert.description} "
            f"(severity: {alert.severity.value}, value: {alert.current_value:.2f})"
        )

    def get_monitoring_summary(self) -> Optional[Dict[str, Any]]:
        """Get current monitoring and analytics summary."""
        if not self.enable_monitoring or not self.analytics:
            return None
        
        return self.analytics.get_real_time_dashboard_data()

    def check_alerts(self) -> List:
        """Check for any active alerts and return them."""
        if not self.enable_monitoring or not self.alerts_manager:
            return []
        
        # Check thresholds and auto-resolve alerts
        new_alerts = self.alerts_manager.check_thresholds()
        self.alerts_manager.auto_resolve_alerts()
        
        return new_alerts

    def generate_analytics_report(self, time_window_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Generate comprehensive analytics report."""
        if not self.enable_monitoring or not self.analytics:
            return None
        
        return self.analytics.generate_comprehensive_report(time_window_hours)
