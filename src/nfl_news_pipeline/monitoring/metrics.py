"""Enhanced metrics collection for story grouping operations.

Implements Task 9.1: Comprehensive metrics and logging for story grouping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json

from ..logging.audit import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class CostMetrics:
    """Track LLM API costs and usage."""
    
    llm_api_calls: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    avg_cost_per_story: float = 0.0
    daily_spend: float = 0.0
    monthly_spend: float = 0.0

    def record_llm_usage(self, model: str, tokens: int, cost: float) -> None:
        """Record LLM API usage and cost."""
        self.llm_api_calls += 1
        self.total_tokens_used += tokens
        self.total_cost_usd += cost
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost
        logger.info(
            f"LLM usage recorded: model={model}, tokens={tokens}, cost=${cost:.4f}"
        )


@dataclass 
class PerformanceMetrics:
    """Track performance and timing metrics."""
    
    # Processing times
    avg_context_time_ms: float = 0.0
    avg_embedding_time_ms: float = 0.0
    avg_similarity_time_ms: float = 0.0
    avg_total_processing_time_ms: float = 0.0
    
    # Throughput metrics
    stories_per_minute: float = 0.0
    groups_created_per_hour: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    parallel_workers_peak: int = 0
    
    # Operation counts
    similarity_calculations: int = 0
    database_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class QualityMetrics:
    """Track story grouping quality metrics."""
    
    # Group statistics
    avg_group_size: float = 0.0
    median_group_size: float = 0.0
    max_group_size: int = 0
    total_groups: int = 0
    singleton_groups: int = 0
    
    # Similarity scores
    avg_similarity_score: float = 0.0
    median_similarity_score: float = 0.0
    similarity_score_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality indicators
    potential_false_positives: int = 0
    potential_false_negatives: int = 0
    coherence_score: float = 0.0
    coverage_percentage: float = 0.0
    
    # Temporal patterns
    stories_grouped_within_1hr: int = 0
    stories_grouped_within_24hr: int = 0
    avg_time_to_group_minutes: float = 0.0


@dataclass
class GroupingDecision:
    """Record of a grouping decision for analysis."""
    
    story_id: str
    decision_type: str  # "new_group", "existing_group", "skipped"
    similarity_score: Optional[float]
    group_id: Optional[str]
    candidate_count: int
    processing_time_ms: int
    timestamp: datetime
    reason: Optional[str] = None


class GroupingMetricsCollector:
    """Comprehensive metrics collection for story grouping operations."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger
        self.cost_metrics = CostMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        
        # Decision tracking
        self.decisions: List[GroupingDecision] = []
        self.similarity_scores: List[float] = []
        self.group_sizes: Dict[str, int] = defaultdict(int)
        
        # Timing tracking
        self._start_time = time.perf_counter()
        self._stories_processed = 0
        
        # Rolling windows for analysis
        self._recent_decisions = []
        self._max_recent_decisions = 1000

    def record_grouping_decision(
        self,
        story_id: str,
        decision_type: str,
        similarity_score: Optional[float] = None,
        group_id: Optional[str] = None,
        candidate_count: int = 0,
        processing_time_ms: int = 0,
        reason: Optional[str] = None,
    ) -> None:
        """Record a story grouping decision with full context."""
        
        decision = GroupingDecision(
            story_id=story_id,
            decision_type=decision_type,
            similarity_score=similarity_score,
            group_id=group_id,
            candidate_count=candidate_count,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
        )
        
        self.decisions.append(decision)
        self._recent_decisions.append(decision)
        
        # Maintain rolling window
        if len(self._recent_decisions) > self._max_recent_decisions:
            self._recent_decisions.pop(0)
        
        # Update quality metrics
        if similarity_score is not None:
            self.similarity_scores.append(similarity_score)
            
        if group_id:
            self.group_sizes[group_id] += 1
            
        self._stories_processed += 1
        
        # Log decision
        log_data = {
            "story_id": story_id,
            "decision": decision_type,
            "similarity_score": similarity_score,
            "group_id": group_id,
            "candidates": candidate_count,
            "processing_time_ms": processing_time_ms,
            "reason": reason,
        }
        
        logger.info(f"Grouping decision recorded: {json.dumps(log_data)}")
        
        if self.audit_logger:
            self.audit_logger.log_event(
                "story_grouping_decision",
                message=f"Story {story_id} {decision_type}",
                data=log_data,
            )

    def record_llm_cost(
        self,
        model: str,
        tokens: int,
        cost: float,
        operation: str = "context_extraction",
    ) -> None:
        """Record LLM API usage and cost."""
        self.cost_metrics.record_llm_usage(model, tokens, cost)
        
        if self.audit_logger:
            self.audit_logger.log_event(
                "llm_cost_tracking",
                message=f"LLM cost for {operation}",
                data={
                    "model": model,
                    "tokens": tokens,
                    "cost": cost,
                    "operation": operation,
                    "total_cost": self.cost_metrics.total_cost_usd,
                },
            )

    def record_performance_metrics(
        self,
        context_time_ms: int,
        embedding_time_ms: int,
        similarity_time_ms: int,
        total_time_ms: int,
    ) -> None:
        """Record processing performance metrics."""
        
        # Update running averages
        n = self._stories_processed
        if n > 0:
            self.performance_metrics.avg_context_time_ms = (
                (self.performance_metrics.avg_context_time_ms * (n - 1) + context_time_ms) / n
            )
            self.performance_metrics.avg_embedding_time_ms = (
                (self.performance_metrics.avg_embedding_time_ms * (n - 1) + embedding_time_ms) / n
            )
            self.performance_metrics.avg_similarity_time_ms = (
                (self.performance_metrics.avg_similarity_time_ms * (n - 1) + similarity_time_ms) / n
            )
            self.performance_metrics.avg_total_processing_time_ms = (
                (self.performance_metrics.avg_total_processing_time_ms * (n - 1) + total_time_ms) / n
            )

    def update_quality_metrics(self) -> None:
        """Update derived quality metrics from collected data."""
        
        if self.similarity_scores:
            self.quality_metrics.avg_similarity_score = sum(self.similarity_scores) / len(self.similarity_scores)
            sorted_scores = sorted(self.similarity_scores)
            n = len(sorted_scores)
            self.quality_metrics.median_similarity_score = (
                sorted_scores[n // 2] if n % 2 == 1 
                else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            )
            
            # Score distribution
            self.quality_metrics.similarity_score_distribution = {
                "0.0-0.2": sum(1 for s in self.similarity_scores if 0.0 <= s < 0.2),
                "0.2-0.4": sum(1 for s in self.similarity_scores if 0.2 <= s < 0.4),
                "0.4-0.6": sum(1 for s in self.similarity_scores if 0.4 <= s < 0.6),
                "0.6-0.8": sum(1 for s in self.similarity_scores if 0.6 <= s < 0.8),
                "0.8-1.0": sum(1 for s in self.similarity_scores if 0.8 <= s <= 1.0),
            }
        
        # Group size metrics
        if self.group_sizes:
            sizes = list(self.group_sizes.values())
            self.quality_metrics.avg_group_size = sum(sizes) / len(sizes)
            
            sorted_sizes = sorted(sizes)
            n = len(sorted_sizes)
            self.quality_metrics.median_group_size = (
                sorted_sizes[n // 2] if n % 2 == 1
                else (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
            )
            
            self.quality_metrics.max_group_size = max(sizes)
            self.quality_metrics.total_groups = len(sizes)
            self.quality_metrics.singleton_groups = sum(1 for s in sizes if s == 1)
        
        # Calculate throughput
        elapsed_seconds = time.perf_counter() - self._start_time
        if elapsed_seconds > 0:
            self.performance_metrics.stories_per_minute = (self._stories_processed * 60) / elapsed_seconds
            groups_created = len([d for d in self.decisions if d.decision_type == "new_group"])
            self.performance_metrics.groups_created_per_hour = (groups_created * 3600) / elapsed_seconds

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics summary report."""
        
        self.update_quality_metrics()
        
        # Update cost averages
        if self._stories_processed > 0:
            self.cost_metrics.avg_cost_per_story = self.cost_metrics.total_cost_usd / self._stories_processed
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_summary": {
                "stories_processed": self._stories_processed,
                "total_decisions": len(self.decisions),
                "elapsed_time_seconds": time.perf_counter() - self._start_time,
            },
            "cost_metrics": {
                "total_cost_usd": self.cost_metrics.total_cost_usd,
                "avg_cost_per_story": self.cost_metrics.avg_cost_per_story,
                "total_api_calls": self.cost_metrics.llm_api_calls,
                "total_tokens": self.cost_metrics.total_tokens_used,
                "cost_by_model": dict(self.cost_metrics.cost_by_model),
            },
            "performance_metrics": {
                "avg_processing_time_ms": self.performance_metrics.avg_total_processing_time_ms,
                "stories_per_minute": self.performance_metrics.stories_per_minute,
                "groups_created_per_hour": self.performance_metrics.groups_created_per_hour,
                "similarity_calculations": self.performance_metrics.similarity_calculations,
            },
            "quality_metrics": {
                "avg_similarity_score": self.quality_metrics.avg_similarity_score,
                "avg_group_size": self.quality_metrics.avg_group_size,
                "total_groups": self.quality_metrics.total_groups,
                "singleton_groups": self.quality_metrics.singleton_groups,
                "score_distribution": dict(self.quality_metrics.similarity_score_distribution),
            },
            "decision_breakdown": {
                decision_type: len([d for d in self.decisions if d.decision_type == decision_type])
                for decision_type in ["new_group", "existing_group", "skipped"]
            },
        }
        
        logger.info(f"Generated metrics summary: {json.dumps(report, indent=2)}")
        
        if self.audit_logger:
            self.audit_logger.log_event(
                "metrics_summary",
                message="Story grouping metrics summary",
                data=report,
            )
        
        return report

    def get_recent_performance_trends(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze recent performance trends for alerting."""
        
        if len(self._recent_decisions) < window_size:
            window_size = len(self._recent_decisions)
            
        if window_size == 0:
            return {}
        
        recent = self._recent_decisions[-window_size:]
        
        # Calculate trends
        recent_similarity_scores = [d.similarity_score for d in recent if d.similarity_score is not None]
        recent_processing_times = [d.processing_time_ms for d in recent]
        
        avg_similarity = sum(recent_similarity_scores) / len(recent_similarity_scores) if recent_similarity_scores else 0
        avg_processing_time = sum(recent_processing_times) / len(recent_processing_times) if recent_processing_times else 0
        
        decision_counts = defaultdict(int)
        for decision in recent:
            decision_counts[decision.decision_type] += 1
        
        return {
            "window_size": window_size,
            "avg_similarity_score": avg_similarity,
            "avg_processing_time_ms": avg_processing_time,
            "decision_distribution": dict(decision_counts),
            "potential_issues": self._detect_potential_issues(recent),
        }

    def _detect_potential_issues(self, recent_decisions: List[GroupingDecision]) -> List[str]:
        """Detect potential quality or performance issues."""
        issues = []
        
        if not recent_decisions:
            return issues
        
        # Check for high error rate
        error_rate = len([d for d in recent_decisions if d.decision_type == "skipped"]) / len(recent_decisions)
        if error_rate > 0.1:  # More than 10% errors
            issues.append(f"High error rate: {error_rate:.1%}")
        
        # Check for low similarity scores
        similarity_scores = [d.similarity_score for d in recent_decisions if d.similarity_score is not None]
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            if avg_similarity < 0.5:
                issues.append(f"Low average similarity: {avg_similarity:.2f}")
        
        # Check for slow processing
        processing_times = [d.processing_time_ms for d in recent_decisions]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            if avg_time > 5000:  # More than 5 seconds
                issues.append(f"Slow processing: {avg_time:.0f}ms average")
        
        # Check for too many singleton groups
        new_groups = [d for d in recent_decisions if d.decision_type == "new_group"]
        if len(new_groups) / len(recent_decisions) > 0.8:  # More than 80% new groups
            issues.append("High rate of new group creation")
        
        return issues