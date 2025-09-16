"""Simple tests for story grouping monitoring components that don't require full imports."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
from collections import defaultdict
import sys
import os

# Add the source directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import just the monitoring components directly
from nfl_news_pipeline.monitoring.metrics import (
    CostMetrics,
    PerformanceMetrics, 
    QualityMetrics,
    GroupingDecision,
)


class TestCostMetrics:
    """Test cost metrics tracking."""

    def test_record_llm_usage(self):
        """Test recording LLM usage and costs."""
        metrics = CostMetrics()
        
        metrics.record_llm_usage("gpt-4o-mini", 1000, 0.15)
        
        assert metrics.llm_api_calls == 1
        assert metrics.total_tokens_used == 1000
        assert metrics.total_cost_usd == 0.15
        assert metrics.cost_by_model["gpt-4o-mini"] == 0.15

    def test_multiple_llm_calls(self):
        """Test multiple LLM calls accumulate correctly."""
        metrics = CostMetrics()
        
        metrics.record_llm_usage("gpt-4o-mini", 500, 0.075)
        metrics.record_llm_usage("gpt-4o-mini", 300, 0.045)
        metrics.record_llm_usage("gemini-2.5-lite", 200, 0.02)
        
        assert metrics.llm_api_calls == 3
        assert metrics.total_tokens_used == 1000
        assert metrics.total_cost_usd == 0.14
        assert metrics.cost_by_model["gpt-4o-mini"] == 0.12
        assert metrics.cost_by_model["gemini-2.5-lite"] == 0.02


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    def test_initial_values(self):
        """Test initial performance metrics values."""
        metrics = PerformanceMetrics()
        
        assert metrics.avg_context_time_ms == 0.0
        assert metrics.avg_embedding_time_ms == 0.0
        assert metrics.avg_similarity_time_ms == 0.0
        assert metrics.stories_per_minute == 0.0
        assert metrics.similarity_calculations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0


class TestQualityMetrics:
    """Test quality metrics tracking."""

    def test_initial_values(self):
        """Test initial quality metrics values."""
        metrics = QualityMetrics()
        
        assert metrics.avg_group_size == 0.0
        assert metrics.median_group_size == 0.0
        assert metrics.max_group_size == 0
        assert metrics.total_groups == 0
        assert metrics.singleton_groups == 0
        assert metrics.avg_similarity_score == 0.0
        assert metrics.potential_false_positives == 0
        assert metrics.potential_false_negatives == 0


class TestGroupingDecision:
    """Test grouping decision data structure."""

    def test_create_decision(self):
        """Test creating a grouping decision."""
        timestamp = datetime.now(timezone.utc)
        
        decision = GroupingDecision(
            story_id="story_123",
            decision_type="existing_group",
            similarity_score=0.85,
            group_id="group_456",
            candidate_count=3,
            processing_time_ms=500,
            timestamp=timestamp,
            reason="high_similarity",
        )
        
        assert decision.story_id == "story_123"
        assert decision.decision_type == "existing_group"
        assert decision.similarity_score == 0.85
        assert decision.group_id == "group_456"
        assert decision.candidate_count == 3
        assert decision.processing_time_ms == 500
        assert decision.timestamp == timestamp
        assert decision.reason == "high_similarity"

    def test_decision_without_optional_fields(self):
        """Test creating a decision with minimal fields."""
        timestamp = datetime.now(timezone.utc)
        
        decision = GroupingDecision(
            story_id="story_789",
            decision_type="skipped",
            similarity_score=None,
            group_id=None,
            candidate_count=0,
            processing_time_ms=100,
            timestamp=timestamp,
        )
        
        assert decision.story_id == "story_789"
        assert decision.decision_type == "skipped"
        assert decision.similarity_score is None
        assert decision.group_id is None
        assert decision.candidate_count == 0
        assert decision.reason is None


class MockMetricsCollector:
    """Mock metrics collector for testing that doesn't require full imports."""
    
    def __init__(self):
        self.decisions = []
        self.similarity_scores = []
        self.group_sizes = defaultdict(int)
        self._stories_processed = 0
        self._start_time = datetime.now().timestamp()
        self.cost_metrics = CostMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()

    def record_grouping_decision(self, story_id, decision_type, similarity_score=None, 
                                group_id=None, candidate_count=0, processing_time_ms=0, reason=None):
        """Record a grouping decision."""
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
        
        if similarity_score is not None:
            self.similarity_scores.append(similarity_score)
            
        if group_id:
            self.group_sizes[group_id] += 1
            
        self._stories_processed += 1

    def update_quality_metrics(self):
        """Update derived quality metrics."""
        if self.similarity_scores:
            self.quality_metrics.avg_similarity_score = sum(self.similarity_scores) / len(self.similarity_scores)
            sorted_scores = sorted(self.similarity_scores)
            n = len(sorted_scores)
            self.quality_metrics.median_similarity_score = (
                sorted_scores[n // 2] if n % 2 == 1 
                else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            )
        
        if self.group_sizes:
            sizes = list(self.group_sizes.values())
            self.quality_metrics.avg_group_size = sum(sizes) / len(sizes)
            self.quality_metrics.total_groups = len(sizes)
            self.quality_metrics.singleton_groups = sum(1 for s in sizes if s == 1)
            self.quality_metrics.max_group_size = max(sizes)


class TestMockMetricsCollector:
    """Test the mock metrics collector."""

    def test_record_decisions(self):
        """Test recording multiple decisions."""
        collector = MockMetricsCollector()
        
        # Record decisions
        collector.record_grouping_decision("story_1", "existing_group", 0.8, "group_1", 3, 500)
        collector.record_grouping_decision("story_2", "new_group", 1.0, "group_2", 0, 300)
        collector.record_grouping_decision("story_3", "skipped", None, None, 0, 100, "no_candidates")
        
        assert len(collector.decisions) == 3
        assert collector._stories_processed == 3
        assert len(collector.similarity_scores) == 2  # Only for non-skipped
        assert collector.group_sizes["group_1"] == 1
        assert collector.group_sizes["group_2"] == 1

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        collector = MockMetricsCollector()
        
        # Add test data
        collector.record_grouping_decision("story_1", "existing_group", 0.8, "group_1", 2)
        collector.record_grouping_decision("story_2", "existing_group", 0.9, "group_1", 2)
        collector.record_grouping_decision("story_3", "existing_group", 0.7, "group_1", 2)
        collector.record_grouping_decision("story_4", "new_group", 1.0, "group_2", 0)
        
        collector.update_quality_metrics()
        
        assert collector.quality_metrics.avg_similarity_score == 0.85  # (0.8+0.9+0.7+1.0)/4
        assert collector.quality_metrics.total_groups == 2
        assert collector.quality_metrics.singleton_groups == 1  # group_2 has 1 story
        assert collector.quality_metrics.avg_group_size == 2.0  # (3+1)/2
        assert collector.quality_metrics.max_group_size == 3

    def test_cost_tracking(self):
        """Test cost tracking integration."""
        collector = MockMetricsCollector()
        
        collector.cost_metrics.record_llm_usage("gpt-4o-mini", 1000, 0.15)
        collector.cost_metrics.record_llm_usage("gemini-2.5-lite", 500, 0.05)
        
        assert collector.cost_metrics.total_cost_usd == 0.20
        assert collector.cost_metrics.llm_api_calls == 2
        assert collector.cost_metrics.total_tokens_used == 1500


def test_monitoring_workflow():
    """Test a complete monitoring workflow."""
    collector = MockMetricsCollector()
    
    # Simulate processing a batch of stories
    stories_data = [
        ("story_1", "existing_group", 0.85, "group_1", 3, 450),
        ("story_2", "existing_group", 0.92, "group_1", 3, 380),
        ("story_3", "new_group", 1.0, "group_2", 0, 520),
        ("story_4", "existing_group", 0.78, "group_2", 2, 410),
        ("story_5", "skipped", None, None, 0, 200, "embedding_failed"),
    ]
    
    # Record all decisions
    for story_data in stories_data:
        collector.record_grouping_decision(*story_data)
    
    # Record some LLM costs
    collector.cost_metrics.record_llm_usage("gpt-4o-mini", 800, 0.12)
    collector.cost_metrics.record_llm_usage("gpt-4o-mini", 600, 0.09)
    
    # Update metrics
    collector.update_quality_metrics()
    
    # Verify results
    assert len(collector.decisions) == 5
    assert collector._stories_processed == 5
    assert collector.quality_metrics.total_groups == 2
    assert collector.quality_metrics.avg_similarity_score == pytest.approx(0.8875)  # (0.85+0.92+1.0+0.78)/4
    assert collector.cost_metrics.total_cost_usd == 0.21
    
    # Check decision breakdown
    decision_types = [d.decision_type for d in collector.decisions]
    assert decision_types.count("existing_group") == 3
    assert decision_types.count("new_group") == 1
    assert decision_types.count("skipped") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])