"""Analytics and reporting system for story grouping insights.

Implements Task 9.3: Analytics reporting for trending stories, group evolution, and coverage analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import statistics

from ..models import ProcessedNewsItem
from .metrics import GroupingMetricsCollector, GroupingDecision

logger = logging.getLogger(__name__)


@dataclass
class TrendingStory:
    """Represents a trending story with metadata."""
    
    story_id: str
    url: str
    title: str
    group_id: str
    group_size: int
    latest_similarity_score: float
    first_seen: datetime
    last_updated: datetime
    velocity_score: float  # How quickly the story is growing
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


@dataclass
class GroupEvolution:
    """Track how a story group evolves over time."""
    
    group_id: str
    creation_time: datetime
    current_size: int
    growth_rate: float  # Stories per hour
    peak_size: int
    dominant_entities: List[str]
    dominant_topics: List[str]
    lifecycle_stage: str  # "emerging", "growing", "mature", "declining"
    coherence_score: float
    timeline: List[Tuple[datetime, int]] = field(default_factory=list)  # (timestamp, size)


@dataclass
class CoverageAnalysis:
    """Analysis of story coverage patterns."""
    
    total_stories: int
    grouped_stories: int
    singleton_groups: int
    coverage_percentage: float
    duplicate_detection_rate: float
    avg_stories_per_group: float
    temporal_distribution: Dict[str, int]  # Hour -> story count
    entity_coverage: Dict[str, int]  # Entity -> story count
    topic_coverage: Dict[str, int]  # Topic -> story count


class TrendingAnalysis:
    """Analyze trending stories and identify hot topics."""
    
    def __init__(self, metrics_collector: GroupingMetricsCollector):
        self.metrics_collector = metrics_collector

    def identify_trending_stories(
        self,
        time_window_hours: int = 24,
        min_group_size: int = 3,
        min_velocity: float = 1.0,
    ) -> List[TrendingStory]:
        """Identify stories that are trending based on group growth."""
        trending_stories = []
        
        # Get recent decisions within time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_decisions = [
            d for d in self.metrics_collector.decisions
            if d.timestamp >= cutoff_time and d.group_id
        ]
        
        # Group decisions by group_id
        groups_timeline = defaultdict(list)
        for decision in recent_decisions:
            groups_timeline[decision.group_id].append(decision)
        
        # Analyze each group for trending characteristics
        for group_id, decisions in groups_timeline.items():
            if len(decisions) < min_group_size:
                continue
            
            # Sort by timestamp
            decisions.sort(key=lambda d: d.timestamp)
            
            # Calculate velocity (stories per hour)
            time_span = (decisions[-1].timestamp - decisions[0].timestamp).total_seconds() / 3600
            velocity = len(decisions) / max(time_span, 0.1)  # Avoid division by zero
            
            if velocity < min_velocity:
                continue
            
            # Create trending story entry
            latest_decision = decisions[-1]
            trending_story = TrendingStory(
                story_id=latest_decision.story_id,
                url="",  # Would need to fetch from storage
                title="",  # Would need to fetch from storage
                group_id=group_id,
                group_size=len(decisions),
                latest_similarity_score=latest_decision.similarity_score or 0.0,
                first_seen=decisions[0].timestamp,
                last_updated=decisions[-1].timestamp,
                velocity_score=velocity,
            )
            
            trending_stories.append(trending_story)
        
        # Sort by velocity score (highest first)
        trending_stories.sort(key=lambda s: s.velocity_score, reverse=True)
        
        logger.info(f"Identified {len(trending_stories)} trending stories")
        return trending_stories

    def analyze_topic_trends(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze trending topics and entities."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_decisions = [
            d for d in self.metrics_collector.decisions
            if d.timestamp >= cutoff_time
        ]
        
        # This would be enhanced with actual entity/topic data from stories
        # For now, we analyze based on available decision data
        
        group_popularity = Counter([d.group_id for d in recent_decisions if d.group_id])
        
        return {
            "time_window_hours": time_window_hours,
            "total_decisions": len(recent_decisions),
            "active_groups": len(group_popularity),
            "most_active_groups": dict(group_popularity.most_common(10)),
            "new_groups_created": len([d for d in recent_decisions if d.decision_type == "new_group"]),
            "grouping_velocity": len(recent_decisions) / time_window_hours,
        }


class GroupEvolutionAnalysis:
    """Analyze how story groups evolve over time."""
    
    def __init__(self, metrics_collector: GroupingMetricsCollector):
        self.metrics_collector = metrics_collector

    def track_group_evolution(self, group_id: str) -> Optional[GroupEvolution]:
        """Track the evolution of a specific group."""
        group_decisions = [
            d for d in self.metrics_collector.decisions
            if d.group_id == group_id
        ]
        
        if not group_decisions:
            return None
        
        # Sort by timestamp
        group_decisions.sort(key=lambda d: d.timestamp)
        
        # Build timeline
        timeline = []
        for i, decision in enumerate(group_decisions):
            timeline.append((decision.timestamp, i + 1))  # Size grows with each addition
        
        # Calculate growth rate
        if len(timeline) > 1:
            time_span = (timeline[-1][0] - timeline[0][0]).total_seconds() / 3600
            growth_rate = len(timeline) / max(time_span, 0.1)
        else:
            growth_rate = 0.0
        
        # Determine lifecycle stage
        current_size = len(group_decisions)
        if current_size <= 2:
            lifecycle_stage = "emerging"
        elif growth_rate > 2.0:
            lifecycle_stage = "growing"
        elif growth_rate > 0.5:
            lifecycle_stage = "mature"
        else:
            lifecycle_stage = "declining"
        
        # Calculate coherence based on similarity scores
        similarity_scores = [d.similarity_score for d in group_decisions if d.similarity_score is not None]
        coherence_score = statistics.mean(similarity_scores) if similarity_scores else 0.0
        
        return GroupEvolution(
            group_id=group_id,
            creation_time=timeline[0][0],
            current_size=current_size,
            growth_rate=growth_rate,
            peak_size=current_size,  # Would track historically in a real implementation
            dominant_entities=[],  # Would extract from actual story content
            dominant_topics=[],  # Would extract from actual story content
            lifecycle_stage=lifecycle_stage,
            coherence_score=coherence_score,
            timeline=timeline,
        )

    def analyze_all_group_evolutions(self) -> List[GroupEvolution]:
        """Analyze evolution patterns for all groups."""
        group_ids = set(d.group_id for d in self.metrics_collector.decisions if d.group_id)
        
        evolutions = []
        for group_id in group_ids:
            evolution = self.track_group_evolution(group_id)
            if evolution:
                evolutions.append(evolution)
        
        return evolutions

    def get_lifecycle_distribution(self) -> Dict[str, int]:
        """Get distribution of groups by lifecycle stage."""
        evolutions = self.analyze_all_group_evolutions()
        
        distribution = Counter([e.lifecycle_stage for e in evolutions])
        return dict(distribution)


class CoverageAnalyzer:
    """Analyze story coverage and duplicate detection effectiveness."""
    
    def __init__(self, metrics_collector: GroupingMetricsCollector):
        self.metrics_collector = metrics_collector

    def analyze_coverage(self, time_window_hours: Optional[int] = None) -> CoverageAnalysis:
        """Analyze story coverage patterns."""
        
        decisions = self.metrics_collector.decisions
        if time_window_hours:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            decisions = [d for d in decisions if d.timestamp >= cutoff_time]
        
        total_stories = len(decisions)
        if total_stories == 0:
            return CoverageAnalysis(
                total_stories=0,
                grouped_stories=0,
                singleton_groups=0,
                coverage_percentage=0.0,
                duplicate_detection_rate=0.0,
                avg_stories_per_group=0.0,
                temporal_distribution={},
                entity_coverage={},
                topic_coverage={},
            )
        
        # Count grouped vs singleton stories
        grouped_decisions = [d for d in decisions if d.decision_type in ["existing_group", "new_group"]]
        grouped_stories = len(grouped_decisions)
        
        # Count singleton groups (groups with only one story)
        group_sizes = Counter([d.group_id for d in grouped_decisions if d.group_id])
        singleton_groups = sum(1 for size in group_sizes.values() if size == 1)
        
        # Calculate metrics
        coverage_percentage = (grouped_stories / total_stories) * 100
        duplicate_detection_rate = ((grouped_stories - len(group_sizes)) / total_stories) * 100 if total_stories > 0 else 0
        avg_stories_per_group = grouped_stories / len(group_sizes) if group_sizes else 0
        
        # Temporal distribution (by hour)
        temporal_distribution = defaultdict(int)
        for decision in decisions:
            hour_key = decision.timestamp.strftime("%H")
            temporal_distribution[hour_key] += 1
        
        return CoverageAnalysis(
            total_stories=total_stories,
            grouped_stories=grouped_stories,
            singleton_groups=singleton_groups,
            coverage_percentage=coverage_percentage,
            duplicate_detection_rate=duplicate_detection_rate,
            avg_stories_per_group=avg_stories_per_group,
            temporal_distribution=dict(temporal_distribution),
            entity_coverage={},  # Would populate with actual entity data
            topic_coverage={},  # Would populate with actual topic data
        )

    def detect_potential_duplicates(self, similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Detect potential duplicate stories that should have been grouped."""
        
        potential_duplicates = []
        
        # Get all decisions with high similarity scores
        high_similarity_decisions = [
            d for d in self.metrics_collector.decisions
            if d.similarity_score and d.similarity_score >= similarity_threshold
        ]
        
        # Group by similarity score ranges
        for decision in high_similarity_decisions:
            if decision.decision_type == "new_group":
                # This might be a missed duplicate
                potential_duplicates.append({
                    "story_id": decision.story_id,
                    "similarity_score": decision.similarity_score,
                    "reason": "High similarity but created new group",
                    "timestamp": decision.timestamp.isoformat(),
                })
        
        logger.info(f"Detected {len(potential_duplicates)} potential duplicate misclassifications")
        return potential_duplicates


class GroupingAnalytics:
    """Main analytics coordinator for story grouping insights."""
    
    def __init__(self, metrics_collector: GroupingMetricsCollector):
        self.metrics_collector = metrics_collector
        self.trending_analysis = TrendingAnalysis(metrics_collector)
        self.evolution_analysis = GroupEvolutionAnalysis(metrics_collector)
        self.coverage_analyzer = CoverageAnalyzer(metrics_collector)

    def generate_comprehensive_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        
        logger.info(f"Generating comprehensive analytics report for {time_window_hours}h window")
        
        # Get trending analysis
        trending_stories = self.trending_analysis.identify_trending_stories(time_window_hours)
        topic_trends = self.trending_analysis.analyze_topic_trends(time_window_hours)
        
        # Get group evolution analysis
        lifecycle_distribution = self.evolution_analysis.get_lifecycle_distribution()
        
        # Get coverage analysis
        coverage_analysis = self.coverage_analyzer.analyze_coverage(time_window_hours)
        potential_duplicates = self.coverage_analyzer.detect_potential_duplicates()
        
        # Get basic metrics summary
        metrics_summary = self.metrics_collector.generate_summary_report()
        
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "time_window_hours": time_window_hours,
            "summary": {
                "total_stories_processed": metrics_summary["processing_summary"]["stories_processed"],
                "total_groups_created": len(set(d.group_id for d in self.metrics_collector.decisions if d.group_id)),
                "trending_stories_count": len(trending_stories),
                "coverage_percentage": coverage_analysis.coverage_percentage,
                "duplicate_detection_rate": coverage_analysis.duplicate_detection_rate,
            },
            "trending_analysis": {
                "trending_stories": [
                    {
                        "group_id": story.group_id,
                        "group_size": story.group_size,
                        "velocity_score": story.velocity_score,
                        "first_seen": story.first_seen.isoformat(),
                        "last_updated": story.last_updated.isoformat(),
                    }
                    for story in trending_stories[:10]  # Top 10
                ],
                "topic_trends": topic_trends,
            },
            "group_evolution": {
                "lifecycle_distribution": lifecycle_distribution,
                "total_active_groups": sum(lifecycle_distribution.values()),
            },
            "coverage_analysis": {
                "total_stories": coverage_analysis.total_stories,
                "grouped_stories": coverage_analysis.grouped_stories,
                "singleton_groups": coverage_analysis.singleton_groups,
                "coverage_percentage": coverage_analysis.coverage_percentage,
                "duplicate_detection_rate": coverage_analysis.duplicate_detection_rate,
                "avg_stories_per_group": coverage_analysis.avg_stories_per_group,
                "temporal_distribution": coverage_analysis.temporal_distribution,
            },
            "quality_insights": {
                "potential_duplicates_count": len(potential_duplicates),
                "avg_similarity_score": metrics_summary["quality_metrics"]["avg_similarity_score"],
                "similarity_distribution": metrics_summary["quality_metrics"]["score_distribution"],
            },
            "performance_insights": {
                "processing_rate": metrics_summary["performance_metrics"]["stories_per_minute"],
                "avg_processing_time": metrics_summary["performance_metrics"]["avg_processing_time_ms"],
                "cost_efficiency": {
                    "total_cost": metrics_summary["cost_metrics"]["total_cost_usd"],
                    "cost_per_story": metrics_summary["cost_metrics"]["avg_cost_per_story"],
                },
            },
        }
        
        logger.info(f"Generated comprehensive report with {len(report)} sections")
        return report

    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for monitoring dashboard."""
        
        recent_trends = self.metrics_collector.get_recent_performance_trends(window_size=50)
        
        # Get last 1 hour of activity
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_decisions = [
            d for d in self.metrics_collector.decisions
            if d.timestamp >= one_hour_ago
        ]
        
        return {
            "current_time": datetime.now(timezone.utc).isoformat(),
            "last_hour_activity": {
                "stories_processed": len(recent_decisions),
                "new_groups_created": len([d for d in recent_decisions if d.decision_type == "new_group"]),
                "existing_groups_updated": len([d for d in recent_decisions if d.decision_type == "existing_group"]),
                "stories_skipped": len([d for d in recent_decisions if d.decision_type == "skipped"]),
            },
            "current_performance": {
                "avg_processing_time_ms": recent_trends.get("avg_processing_time_ms", 0),
                "avg_similarity_score": recent_trends.get("avg_similarity_score", 0),
                "stories_per_minute": self.metrics_collector.performance_metrics.stories_per_minute,
            },
            "quality_indicators": {
                "total_groups": self.metrics_collector.quality_metrics.total_groups,
                "avg_group_size": self.metrics_collector.quality_metrics.avg_group_size,
                "singleton_groups": self.metrics_collector.quality_metrics.singleton_groups,
            },
            "cost_tracking": {
                "total_cost_usd": self.metrics_collector.cost_metrics.total_cost_usd,
                "api_calls_made": self.metrics_collector.cost_metrics.llm_api_calls,
            },
            "potential_issues": recent_trends.get("potential_issues", []),
        }

    def export_analytics_data(self, format: str = "json") -> str:
        """Export analytics data in specified format."""
        
        if format.lower() != "json":
            raise ValueError("Only JSON format is currently supported")
        
        comprehensive_report = self.generate_comprehensive_report()
        
        import json
        return json.dumps(comprehensive_report, indent=2, default=str)