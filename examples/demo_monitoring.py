#!/usr/bin/env python3
"""Standalone demonstration of story grouping monitoring capabilities."""

import json
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CostMetrics:
    """Track LLM API costs and usage."""
    
    llm_api_calls: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_operation: Dict[str, float] = field(default_factory=dict)
    tokens_by_operation: Dict[str, int] = field(default_factory=dict)
    api_calls_by_operation: Dict[str, int] = field(default_factory=dict)
    avg_cost_per_story: float = 0.0
    daily_spend: float = 0.0
    monthly_spend: float = 0.0

    def record_llm_usage(self, model: str, tokens: int, cost: float, operation: Optional[str] = None) -> None:
        """Record LLM API usage and cost."""
        self.llm_api_calls += 1
        self.total_tokens_used += tokens
        self.total_cost_usd = round(self.total_cost_usd + float(cost), 12)
        self.cost_by_model[model] = round(self.cost_by_model.get(model, 0.0) + float(cost), 12)
        if operation:
            self.cost_by_operation[operation] = round(self.cost_by_operation.get(operation, 0.0) + float(cost), 12)
            self.tokens_by_operation[operation] = self.tokens_by_operation.get(operation, 0) + int(tokens)
            self.api_calls_by_operation[operation] = self.api_calls_by_operation.get(operation, 0) + 1
        logger.info(f"LLM usage recorded: model={model}, tokens={tokens}, cost=${cost:.4f}")


@dataclass 
class PerformanceMetrics:
    """Track performance and timing metrics."""
    
    avg_context_time_ms: float = 0.0
    avg_embedding_time_ms: float = 0.0
    avg_similarity_time_ms: float = 0.0
    avg_total_processing_time_ms: float = 0.0
    stories_per_minute: float = 0.0
    groups_created_per_hour: float = 0.0
    similarity_calculations: int = 0


@dataclass
class QualityMetrics:
    """Track story grouping quality metrics."""
    
    avg_group_size: float = 0.0
    median_group_size: float = 0.0
    max_group_size: int = 0
    total_groups: int = 0
    singleton_groups: int = 0
    avg_similarity_score: float = 0.0
    median_similarity_score: float = 0.0


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
    
    def __init__(self):
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
        
        # Update quality metrics
        if similarity_score is not None:
            self.similarity_scores.append(similarity_score)
            
        if group_id:
            self.group_sizes[group_id] += 1
            
        self._stories_processed += 1
        
        logger.info(f"Grouping decision recorded: {story_id} -> {decision_type} (score: {similarity_score})")

    def record_llm_cost(self, model: str, tokens: int, cost: float) -> None:
        """Record LLM API usage and cost."""
        self.cost_metrics.record_llm_usage(model, tokens, cost)

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

    def generate_summary_report(self) -> Dict[str, any]:
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
                "cost_by_operation": dict(self.cost_metrics.cost_by_operation),
                "tokens_by_operation": dict(self.cost_metrics.tokens_by_operation),
                "api_calls_by_operation": dict(self.cost_metrics.api_calls_by_operation),
            },
            "performance_metrics": {
                "stories_per_minute": self.performance_metrics.stories_per_minute,
                "groups_created_per_hour": self.performance_metrics.groups_created_per_hour,
            },
            "quality_metrics": {
                "avg_similarity_score": self.quality_metrics.avg_similarity_score,
                "avg_group_size": self.quality_metrics.avg_group_size,
                "total_groups": self.quality_metrics.total_groups,
                "singleton_groups": self.quality_metrics.singleton_groups,
            },
            "decision_breakdown": {
                decision_type: len([d for d in self.decisions if d.decision_type == decision_type])
                for decision_type in ["new_group", "existing_group", "skipped"]
            },
        }
        
        logger.info("Generated comprehensive metrics summary")
        return report


def demonstrate_monitoring():
    """Demonstrate the monitoring system capabilities."""
    
    print("🔍 Story Grouping Monitoring System Demonstration")
    print("=" * 60)
    
    # Initialize monitoring
    collector = GroupingMetricsCollector()
    
    # Simulate story processing with various outcomes
    print("\n📊 Simulating story processing...")
    
    # Simulate some successful groupings
    test_stories = [
        ("story_001", "existing_group", 0.87, "group_sports_1", 4, 450, None),
        ("story_002", "existing_group", 0.91, "group_sports_1", 4, 380, None),
        ("story_003", "new_group", 1.0, "group_trade_1", 0, 520, None),
        ("story_004", "existing_group", 0.82, "group_trade_1", 2, 410, None),
        ("story_005", "existing_group", 0.89, "group_trade_1", 2, 390, None),
        ("story_006", "new_group", 1.0, "group_injury_1", 0, 480, None),
        ("story_007", "skipped", None, None, 0, 200, "embedding_failed"),
        ("story_008", "existing_group", 0.85, "group_sports_1", 4, 420, None),
        ("story_009", "skipped", None, None, 1, 150, "low_similarity"),
        ("story_010", "existing_group", 0.93, "group_injury_1", 3, 440, None),
    ]
    
    for story_data in test_stories:
        collector.record_grouping_decision(*story_data)
        time.sleep(0.01)  # Small delay to show processing time
    
    # Simulate LLM costs (token-based) for gpt-5-nano
    print("\n💰 Recording LLM API costs (token-based)...")
    def estimate_gpt5_nano_cost(input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        return (
            (input_tokens / 1000.0) * 0.00005
            + (cached_input_tokens / 1000.0) * 0.000005
            + (output_tokens / 1000.0) * 0.0004
        )

    # context_extraction call 1
    in1, out1, cached1 = 600, 200, 0
    cost1 = estimate_gpt5_nano_cost(in1, out1, cached1)
    collector.cost_metrics.record_llm_usage("gpt-5-nano", in1 + out1 + cached1, cost1, operation="context_extraction")

    # context_extraction call 2 (cache hit on most of the prompt)
    in2, out2, cached2 = 100, 150, 400
    cost2 = estimate_gpt5_nano_cost(in2, out2, cached2)
    collector.cost_metrics.record_llm_usage("gpt-5-nano", in2 + out2 + cached2, cost2, operation="context_extraction")

    # embedding/reasoning or another operation example
    in3, out3, cached3 = 300, 100, 0
    cost3 = estimate_gpt5_nano_cost(in3, out3, cached3)
    collector.cost_metrics.record_llm_usage("gpt-5-nano", in3 + out3 + cached3, cost3, operation="similarity_assist")
    
    # Generate comprehensive report
    print("\n📈 Generating analytics report...")
    report = collector.generate_summary_report()
    
    # Display key metrics
    print(f"\n✅ Processing Summary:")
    print(f"   Stories processed: {report['processing_summary']['stories_processed']}")
    print(f"   Total decisions: {report['processing_summary']['total_decisions']}")
    print(f"   Processing time: {report['processing_summary']['elapsed_time_seconds']:.2f} seconds")
    
    print(f"\n💰 Cost Metrics:")
    print(f"   Total cost: ${report['cost_metrics']['total_cost_usd']:.3f}")
    print(f"   Average cost per story: ${report['cost_metrics']['avg_cost_per_story']:.4f}")
    print(f"   Total API calls: {report['cost_metrics']['total_api_calls']}")
    print(f"   Total tokens: {report['cost_metrics']['total_tokens']}")
    print(f"   Cost by operation: {report['cost_metrics'].get('cost_by_operation', {})}")
    print(f"   Tokens by operation: {report['cost_metrics'].get('tokens_by_operation', {})}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Stories per minute: {report['performance_metrics']['stories_per_minute']:.1f}")
    print(f"   Groups created per hour: {report['performance_metrics']['groups_created_per_hour']:.1f}")
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   Average similarity score: {report['quality_metrics']['avg_similarity_score']:.3f}")
    print(f"   Average group size: {report['quality_metrics']['avg_group_size']:.1f}")
    print(f"   Total groups: {report['quality_metrics']['total_groups']}")
    print(f"   Singleton groups: {report['quality_metrics']['singleton_groups']}")
    
    print(f"\n📋 Decision Breakdown:")
    for decision_type, count in report['decision_breakdown'].items():
        print(f"   {decision_type}: {count}")
    
    # Demonstrate alerting logic
    print(f"\n🚨 Alert Detection:")
    error_rate = report['decision_breakdown']['skipped'] / report['processing_summary']['stories_processed']
    if error_rate > 0.1:
        print(f"   ⚠️  HIGH ERROR RATE: {error_rate:.1%} (threshold: 10%)")
    else:
        print(f"   ✅ Error rate normal: {error_rate:.1%}")
    
    if report['quality_metrics']['avg_similarity_score'] < 0.5:
        print(f"   ⚠️  LOW SIMILARITY: {report['quality_metrics']['avg_similarity_score']:.3f}")
    else:
        print(f"   ✅ Similarity scores healthy: {report['quality_metrics']['avg_similarity_score']:.3f}")
    
    if report['cost_metrics']['total_cost_usd'] > 1.0:
        print(f"   ⚠️  HIGH COSTS: ${report['cost_metrics']['total_cost_usd']:.3f}")
    else:
        print(f"   ✅ Costs within budget: ${report['cost_metrics']['total_cost_usd']:.3f}")
    
    # Show full report as JSON
    print(f"\n📄 Full Report (JSON):")
    print(json.dumps(report, indent=2))
    
    print(f"\n🎉 Monitoring demonstration completed successfully!")
    print(f"✨ The system tracked {len(collector.decisions)} decisions across {len(collector.group_sizes)} groups")
    
    return collector, report


if __name__ == "__main__":
    collector, report = demonstrate_monitoring()