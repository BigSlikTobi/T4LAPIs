"""Tests for story grouping monitoring and analytics features."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from src.nfl_news_pipeline.monitoring.metrics import (
    GroupingMetricsCollector,
    CostMetrics,
    PerformanceMetrics,
    QualityMetrics,
)
from src.nfl_news_pipeline.monitoring.alerts import GroupingAlertsManager, AlertSeverity
from src.nfl_news_pipeline.monitoring.analytics import GroupingAnalytics, TrendingAnalysis


class TestGroupingMetricsCollector:
    """Test the metrics collection functionality."""

    def test_record_grouping_decision(self):
        """Test recording grouping decisions."""
        collector = GroupingMetricsCollector()
        
        # Record a decision to assign to existing group
        collector.record_grouping_decision(
            story_id="story_1",
            decision_type="existing_group",
            similarity_score=0.85,
            group_id="group_1",
            candidate_count=3,
            processing_time_ms=500,
        )
        
        assert len(collector.decisions) == 1
        assert collector.decisions[0].decision_type == "existing_group"
        assert collector.decisions[0].similarity_score == 0.85
        assert len(collector.similarity_scores) == 1
        assert collector.group_sizes["group_1"] == 1

    def test_record_llm_cost(self):
        """Test recording LLM costs."""
        collector = GroupingMetricsCollector()
        
        collector.record_llm_cost(
            model="gpt-4o-mini",
            tokens=1000,
            cost=0.15,
        )
        
        assert collector.cost_metrics.llm_api_calls == 1
        assert collector.cost_metrics.total_tokens_used == 1000
        assert collector.cost_metrics.total_cost_usd == 0.15
        assert collector.cost_metrics.cost_by_model["gpt-4o-mini"] == 0.15

    def test_update_quality_metrics(self):
        """Test quality metrics calculation."""
        collector = GroupingMetricsCollector()
        
        # Add some test data
        collector.similarity_scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        collector.group_sizes = {"group_1": 3, "group_2": 1, "group_3": 5}
        
        collector.update_quality_metrics()
        
        assert collector.quality_metrics.avg_similarity_score == 0.85
        assert collector.quality_metrics.total_groups == 3
        assert collector.quality_metrics.singleton_groups == 1
        assert collector.quality_metrics.max_group_size == 5

    def test_generate_summary_report(self):
        """Test generating comprehensive summary report."""
        collector = GroupingMetricsCollector()
        
        # Add some test decisions
        for i in range(5):
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type="existing_group" if i < 3 else "new_group",
                similarity_score=0.8 + (i * 0.05),
                group_id=f"group_{i % 2}",
                candidate_count=2,
                processing_time_ms=100 + (i * 50),
            )
        
        report = collector.generate_summary_report()
        
        assert "processing_summary" in report
        assert "cost_metrics" in report
        assert "performance_metrics" in report
        assert "quality_metrics" in report
        assert report["processing_summary"]["stories_processed"] == 5

    def test_detect_potential_issues(self):
        """Test detection of potential issues."""
        collector = GroupingMetricsCollector()
        
        # Create decisions with issues
        decisions = []
        for i in range(10):
            decision = Mock()
            decision.decision_type = "skipped" if i < 3 else "new_group"  # High error rate
            decision.similarity_score = 0.3 if i < 5 else 0.9  # Low similarity
            decision.processing_time_ms = 6000  # Slow processing
            decisions.append(decision)
        
        issues = collector._detect_potential_issues(decisions)
        
        assert len(issues) > 0
        assert any("High error rate" in issue for issue in issues)
        assert any("Low average similarity" in issue for issue in issues)
        assert any("Slow processing" in issue for issue in issues)


class TestGroupingAlertsManager:
    """Test the alerting functionality."""

    def test_threshold_breach_detection(self):
        """Test detection of threshold breaches."""
        collector = GroupingMetricsCollector()
        alerts_manager = GroupingAlertsManager(collector)
        
        # Simulate high error rate
        for i in range(10):
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type="skipped",  # All skipped = high error rate
                processing_time_ms=12000,  # Slow processing
            )
        
        alerts = alerts_manager.check_thresholds()
        
        # Should have alerts for error rate and slow processing
        assert len(alerts) > 0
        alert_metrics = [alert.metric_name for alert in alerts]
        assert "error_rate" in alert_metrics
        assert "avg_processing_time_ms" in alert_metrics

    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        collector = GroupingMetricsCollector()
        alerts_manager = GroupingAlertsManager(collector)
        
        # Trigger an alert
        for i in range(5):
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type="skipped",
                processing_time_ms=15000,
            )
        
        alerts1 = alerts_manager.check_thresholds()
        alerts2 = alerts_manager.check_thresholds()  # Should be in cooldown
        
        assert len(alerts1) > 0
        assert len(alerts2) == 0  # In cooldown

    def test_auto_resolve_alerts(self):
        """Test automatic alert resolution."""
        collector = GroupingMetricsCollector()
        alerts_manager = GroupingAlertsManager(collector)
        
        # First create conditions that trigger alerts
        for i in range(5):
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type="skipped",
                processing_time_ms=15000,
            )
        
        alerts = alerts_manager.check_thresholds()
        assert len(alerts_manager.active_alerts) > 0
        
        # Then improve conditions
        for i in range(10):
            collector.record_grouping_decision(
                story_id=f"story_good_{i}",
                decision_type="existing_group",
                similarity_score=0.9,
                processing_time_ms=100,
            )
        
        resolved = alerts_manager.auto_resolve_alerts()
        assert len(resolved) > 0

    def test_anomaly_detection(self):
        """Test statistical anomaly detection."""
        collector = GroupingMetricsCollector()
        alerts_manager = GroupingAlertsManager(collector)
        
        # Create normal decisions
        for i in range(50):
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type="existing_group",
                similarity_score=0.8,
                processing_time_ms=1000,
            )
        
        # Add some anomalous decisions
        for i in range(10):
            collector.record_grouping_decision(
                story_id=f"anomaly_{i}",
                decision_type="new_group",  # All new groups
                similarity_score=0.2,  # Low similarity
                processing_time_ms=10000,  # Slow processing
            )
        
        anomalies = alerts_manager.detect_anomalies()
        assert len(anomalies) > 0


class TestGroupingAnalytics:
    """Test analytics and reporting functionality."""

    def test_trending_analysis(self):
        """Test trending story identification."""
        collector = GroupingMetricsCollector()
        trending_analysis = TrendingAnalysis(collector)
        
        # Create a trending group
        base_time = datetime.now(timezone.utc)
        for i in range(10):
            decision = Mock()
            decision.story_id = f"story_{i}"
            decision.group_id = "trending_group"
            decision.decision_type = "existing_group"
            decision.similarity_score = 0.9
            decision.timestamp = base_time + timedelta(minutes=i * 10)
            collector.decisions.append(decision)
        
        trending_stories = trending_analysis.identify_trending_stories(
            time_window_hours=24,
            min_group_size=5,
            min_velocity=0.5,
        )
        
        assert len(trending_stories) > 0
        assert trending_stories[0].group_id == "trending_group"
        assert trending_stories[0].velocity_score > 0

    def test_coverage_analysis(self):
        """Test coverage analysis functionality."""
        collector = GroupingMetricsCollector()
        analytics = GroupingAnalytics(collector)
        
        # Add various decision types
        for i in range(20):
            decision_type = "existing_group" if i < 10 else "new_group" if i < 15 else "skipped"
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type=decision_type,
                similarity_score=0.8 if decision_type != "skipped" else None,
                group_id=f"group_{i % 3}" if decision_type != "skipped" else None,
            )
        
        coverage = analytics.coverage_analyzer.analyze_coverage()
        
        assert coverage.total_stories == 20
        assert coverage.grouped_stories == 15  # 10 existing + 5 new
        assert coverage.coverage_percentage == 75.0  # 15/20 * 100

    def test_comprehensive_report_generation(self):
        """Test comprehensive analytics report generation."""
        collector = GroupingMetricsCollector()
        analytics = GroupingAnalytics(collector)
        
        # Add some test data
        for i in range(30):
            collector.record_grouping_decision(
                story_id=f"story_{i}",
                decision_type="existing_group" if i % 2 == 0 else "new_group",
                similarity_score=0.8 + (i % 5) * 0.05,
                group_id=f"group_{i % 5}",
                processing_time_ms=1000 + (i * 100),
            )
        
        report = analytics.generate_comprehensive_report(time_window_hours=24)
        
        assert "report_timestamp" in report
        assert "summary" in report
        assert "trending_analysis" in report
        assert "group_evolution" in report
        assert "coverage_analysis" in report
        assert "quality_insights" in report
        assert "performance_insights" in report

    def test_real_time_dashboard_data(self):
        """Test real-time dashboard data generation."""
        collector = GroupingMetricsCollector()
        analytics = GroupingAnalytics(collector)
        
        # Add recent activity
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        for i in range(5):
            decision = Mock()
            decision.story_id = f"recent_story_{i}"
            decision.decision_type = "existing_group"
            decision.timestamp = recent_time + timedelta(minutes=i * 5)
            collector.decisions.append(decision)
        
        dashboard_data = analytics.get_real_time_dashboard_data()
        
        assert "current_time" in dashboard_data
        assert "last_hour_activity" in dashboard_data
        assert "current_performance" in dashboard_data
        assert "quality_indicators" in dashboard_data
        assert "cost_tracking" in dashboard_data


@pytest.mark.asyncio
async def test_integration_with_orchestrator():
    """Test integration of monitoring with the story grouping orchestrator."""
    # This would be a more complex integration test
    # For now, just test that the monitoring components can be instantiated
    
    from src.nfl_news_pipeline.monitoring import GroupingMetricsCollector, GroupingAlertsManager, GroupingAnalytics
    
    collector = GroupingMetricsCollector()
    alerts_manager = GroupingAlertsManager(collector)
    analytics = GroupingAnalytics(collector)
    
    assert collector is not None
    assert alerts_manager is not None
    assert analytics is not None
    
    # Test that they work together
    collector.record_grouping_decision(
        story_id="test_story",
        decision_type="new_group",
        similarity_score=0.9,
        group_id="test_group",
    )
    
    alerts = alerts_manager.check_thresholds()
    dashboard = analytics.get_real_time_dashboard_data()
    
    assert dashboard is not None
    assert "last_hour_activity" in dashboard