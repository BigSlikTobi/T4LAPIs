# Story Grouping Monitoring and Analytics Guide

This guide demonstrates how to use the new monitoring and analytics capabilities implemented for the story similarity grouping feature.

## Overview

The monitoring system provides comprehensive observability for:
- **Metrics Collection**: Processing times, similarity scores, costs, and quality metrics
- **Alerting**: Real-time detection of performance and quality issues  
- **Analytics**: Trending analysis, group evolution, and coverage reporting

## Quick Start

### Basic Monitoring Setup

```python
from nfl_news_pipeline.monitoring import (
    GroupingMetricsCollector,
    GroupingAlertsManager, 
    GroupingAnalytics
)

# Initialize monitoring components
collector = GroupingMetricsCollector()
alerts_manager = GroupingAlertsManager(collector)
analytics = GroupingAnalytics(collector)

# Record a grouping decision
collector.record_grouping_decision(
    story_id="story_123",
    decision_type="existing_group",
    similarity_score=0.85,
    group_id="group_456",
    candidate_count=3,
    processing_time_ms=500
)

# Generate summary report
report = collector.generate_summary_report()
print(f"Processed {report['processing_summary']['stories_processed']} stories")
```

### Integration with Story Grouping Orchestrator

```python
from nfl_news_pipeline.orchestrator.story_grouping import StoryGroupingOrchestrator

# Create orchestrator with monitoring enabled
orchestrator = StoryGroupingOrchestrator(
    group_manager=group_manager,
    context_extractor=context_extractor,
    embedding_generator=embedding_generator,
    enable_monitoring=True,  # Enable comprehensive monitoring
    audit_logger=audit_logger
)

# Process stories - monitoring happens automatically
result = await orchestrator.process_batch(stories, url_id_map)

# Get monitoring insights
summary = orchestrator.get_monitoring_summary()
alerts = orchestrator.check_alerts()
analytics_report = orchestrator.generate_analytics_report(time_window_hours=24)
```

## Monitoring Features

### 1. Metrics Collection (Task 9.1)

#### Cost Tracking
```python
# Record LLM API costs
collector.record_llm_cost(
    model="gpt-5-nano",
    tokens=1000,
    cost=0.15,
    operation="context_extraction"
)

# Access cost metrics
print(f"Total cost: ${collector.cost_metrics.total_cost_usd:.3f}")
print(f"API calls: {collector.cost_metrics.llm_api_calls}")
```

#### Performance Metrics
```python
# Record processing performance
collector.record_performance_metrics(
    context_time_ms=200,
    embedding_time_ms=300,
    similarity_time_ms=100,
    total_time_ms=600
)

# Access performance data
print(f"Stories/min: {collector.performance_metrics.stories_per_minute:.1f}")
```

#### Quality Metrics
```python
# Quality metrics are automatically calculated
collector.update_quality_metrics()

print(f"Avg similarity: {collector.quality_metrics.avg_similarity_score:.3f}")
print(f"Total groups: {collector.quality_metrics.total_groups}")
print(f"Avg group size: {collector.quality_metrics.avg_group_size:.1f}")
```

### 2. Alerting and Anomaly Detection (Task 9.2)

#### Configure Alert Thresholds
```python
from nfl_news_pipeline.monitoring.alerts import AlertThreshold, AlertSeverity

# Add custom threshold
custom_threshold = AlertThreshold(
    metric_name="error_rate",
    threshold_value=0.05,  # 5%
    comparison="gt",
    severity=AlertSeverity.WARNING,
    description="Error rate exceeded 5%",
    cooldown_minutes=10
)

alerts_manager.thresholds.append(custom_threshold)
```

#### Check for Alerts
```python
# Check thresholds and get new alerts
new_alerts = alerts_manager.check_thresholds()

for alert in new_alerts:
    print(f"🚨 {alert.title}: {alert.description}")
    print(f"   Current: {alert.current_value:.2f}, Threshold: {alert.threshold_value:.2f}")

# Auto-resolve alerts when conditions improve
resolved = alerts_manager.auto_resolve_alerts()
print(f"Resolved {len(resolved)} alerts")
```

#### Anomaly Detection
```python
# Detect statistical anomalies
anomalies = alerts_manager.detect_anomalies(window_size=100)

for anomaly in anomalies:
    print(f"⚠️ {anomaly['type']}: {anomaly['description']}")
```

### 3. Analytics and Reporting (Task 9.3)

#### Trending Analysis
```python
# Identify trending stories
trending = analytics.trending_analysis.identify_trending_stories(
    time_window_hours=24,
    min_group_size=3,
    min_velocity=1.0
)

for story in trending[:5]:  # Top 5
    print(f"📈 Group {story.group_id}: {story.group_size} stories, "
          f"velocity {story.velocity_score:.1f}/hour")
```

#### Coverage Analysis
```python
# Analyze story coverage effectiveness
coverage = analytics.coverage_analyzer.analyze_coverage(time_window_hours=24)

print(f"Coverage: {coverage.coverage_percentage:.1f}%")
print(f"Duplicate detection rate: {coverage.duplicate_detection_rate:.1f}%")
print(f"Average stories per group: {coverage.avg_stories_per_group:.1f}")
```

#### Comprehensive Analytics Report
```python
# Generate full analytics report
report = analytics.generate_comprehensive_report(time_window_hours=24)

# Access key sections
summary = report['summary']
trending_data = report['trending_analysis'] 
quality_insights = report['quality_insights']
performance_insights = report['performance_insights']

# Export as JSON
json_report = analytics.export_analytics_data(format="json")
```

#### Real-time Dashboard Data
```python
# Get real-time dashboard data
dashboard = analytics.get_real_time_dashboard_data()

print("Last Hour Activity:")
print(f"  Stories processed: {dashboard['last_hour_activity']['stories_processed']}")
print(f"  New groups: {dashboard['last_hour_activity']['new_groups_created']}")
print(f"  Existing groups updated: {dashboard['last_hour_activity']['existing_groups_updated']}")

print("Current Performance:")
print(f"  Processing time: {dashboard['current_performance']['avg_processing_time_ms']:.0f}ms")
print(f"  Similarity score: {dashboard['current_performance']['avg_similarity_score']:.3f}")
```

## Alert Callback Integration

```python
def handle_alert(alert):
    """Custom alert handler - could send to Slack, email, etc."""
    if alert.severity == AlertSeverity.CRITICAL:
        # Send urgent notification
        send_urgent_notification(f"CRITICAL: {alert.title}")
    elif alert.severity == AlertSeverity.ERROR:
        # Log to monitoring system
        log_error_alert(alert)
    
    # Log all alerts
    logger.warning(f"Alert: {alert.title} - {alert.description}")

# Register callback
alerts_manager.add_alert_callback(handle_alert)
```

## Configuration Example

```python
# Story grouping with comprehensive monitoring
orchestrator = StoryGroupingOrchestrator(
    group_manager=group_manager,
    context_extractor=context_extractor,
    embedding_generator=embedding_generator,
    enable_monitoring=True,
    audit_logger=audit_logger,
    settings=StoryGroupingSettings(
        max_parallelism=4,
        max_candidates=8,
        candidate_similarity_floor=0.35
    )
)

# Configure alert thresholds
alerts_manager = orchestrator.alerts_manager
alerts_manager.thresholds.extend([
    AlertThreshold("avg_processing_time_ms", 5000, "gt", AlertSeverity.WARNING, 
                   "Processing time exceeded 5 seconds"),
    AlertThreshold("hourly_cost_usd", 5.0, "gt", AlertSeverity.ERROR,
                   "Hourly costs exceeded $5"),
    AlertThreshold("avg_similarity_score", 0.4, "lt", AlertSeverity.WARNING,
                   "Average similarity scores very low"),
])

# Process batch with monitoring
result = await orchestrator.process_batch(stories, url_id_map)

# Generate monitoring report
monitoring_summary = orchestrator.get_monitoring_summary()
active_alerts = orchestrator.check_alerts()
analytics_report = orchestrator.generate_analytics_report()

print(f"Processed {len(result.outcomes)} stories")
print(f"Active alerts: {len(active_alerts)}")
print(f"Cost this batch: ${analytics_report['performance_insights']['cost_efficiency']['total_cost']:.3f}")
```

## Key Benefits

1. **Proactive Monitoring**: Detect issues before they impact users
2. **Cost Control**: Track and alert on LLM API spending
3. **Quality Assurance**: Monitor grouping effectiveness and similarity scores
4. **Performance Optimization**: Identify bottlenecks and optimization opportunities
5. **Trending Detection**: Automatically identify viral or breaking stories
6. **Data-Driven Decisions**: Comprehensive analytics for threshold tuning

## Demo Script

Run the included demonstration:

```bash
python examples/demo_monitoring.py
```

This shows a complete workflow with simulated story processing, cost tracking, quality metrics, and alert detection.

## Integration Points

The monitoring system integrates with:
- **Existing Audit Logging**: Uses the established audit logging patterns
- **Storage Layer**: Tracks database operations and performance
- **LLM Services**: Monitors API costs and usage patterns
- **Group Management**: Tracks group lifecycle and evolution
- **Similarity Calculations**: Monitors scoring patterns and thresholds

The system is designed to be minimally invasive while providing comprehensive observability for production deployments.
