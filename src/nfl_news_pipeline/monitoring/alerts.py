"""Alerting system for story grouping quality and performance monitoring.

Implements Task 9.2: Quality monitoring and alerting for unusual patterns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import statistics

from .metrics import GroupingMetricsCollector, GroupingDecision

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertThreshold:
    """Configuration for alert thresholds."""
    
    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "eq"
    severity: AlertSeverity
    description: str
    cooldown_minutes: int = 15  # Minimum time between alerts
    enabled: bool = True


class GroupingAlertsManager:
    """Monitor story grouping metrics and generate alerts for anomalies."""
    
    def __init__(self, metrics_collector: GroupingMetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Default alert thresholds
        self.thresholds = self._get_default_thresholds()
        
        # Last alert times for cooldown
        self._last_alert_times: Dict[str, datetime] = {}

    def _get_default_thresholds(self) -> List[AlertThreshold]:
        """Define default alert thresholds for story grouping metrics."""
        return [
            # Performance alerts
            AlertThreshold(
                metric_name="avg_processing_time_ms",
                threshold_value=10000,  # 10 seconds
                comparison="gt",
                severity=AlertSeverity.WARNING,
                description="Average story processing time exceeds 10 seconds",
                cooldown_minutes=10,
            ),
            AlertThreshold(
                metric_name="error_rate",
                threshold_value=0.15,  # 15%
                comparison="gt",
                severity=AlertSeverity.ERROR,
                description="Story processing error rate exceeds 15%",
                cooldown_minutes=5,
            ),
            
            # Quality alerts
            AlertThreshold(
                metric_name="avg_similarity_score",
                threshold_value=0.3,
                comparison="lt",
                severity=AlertSeverity.WARNING,
                description="Average similarity score is unusually low",
                cooldown_minutes=30,
            ),
            AlertThreshold(
                metric_name="singleton_group_rate",
                threshold_value=0.9,  # 90%
                comparison="gt",
                severity=AlertSeverity.WARNING,
                description="High rate of singleton groups may indicate poor clustering",
                cooldown_minutes=20,
            ),
            AlertThreshold(
                metric_name="avg_group_size",
                threshold_value=20,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                description="Groups are becoming too large",
                cooldown_minutes=30,
            ),
            
            # Cost alerts
            AlertThreshold(
                metric_name="hourly_cost_usd",
                threshold_value=10.0,
                comparison="gt",
                severity=AlertSeverity.ERROR,
                description="LLM costs exceed $10/hour",
                cooldown_minutes=60,
            ),
            AlertThreshold(
                metric_name="daily_cost_usd",
                threshold_value=100.0,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                description="Daily LLM costs exceed $100",
                cooldown_minutes=120,
            ),
            
            # Throughput alerts
            AlertThreshold(
                metric_name="stories_per_minute",
                threshold_value=0.5,
                comparison="lt",
                severity=AlertSeverity.WARNING,
                description="Story processing throughput is very low",
                cooldown_minutes=15,
            ),
        ]

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)

    def check_thresholds(self) -> List[Alert]:
        """Check all configured thresholds and generate alerts."""
        new_alerts = []
        
        # Get current metrics
        current_metrics = self._calculate_current_metrics()
        
        for threshold in self.thresholds:
            if not threshold.enabled:
                continue
                
            if threshold.metric_name not in current_metrics:
                continue
                
            current_value = current_metrics[threshold.metric_name]
            
            # Check if threshold is breached
            if self._is_threshold_breached(current_value, threshold):
                alert = self._create_alert(threshold, current_value)
                
                # Check cooldown
                if self._is_in_cooldown(threshold.metric_name, threshold.cooldown_minutes):
                    continue
                
                new_alerts.append(alert)
                self._activate_alert(alert)
                
        return new_alerts

    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current metric values for threshold checking."""
        metrics = {}
        
        # Get recent performance trends
        recent_trends = self.metrics_collector.get_recent_performance_trends(window_size=50)
        
        if recent_trends:
            metrics["avg_processing_time_ms"] = recent_trends.get("avg_processing_time_ms", 0)
            metrics["avg_similarity_score"] = recent_trends.get("avg_similarity_score", 0)
            
            # Calculate error rate
            decision_dist = recent_trends.get("decision_distribution", {})
            total_decisions = sum(decision_dist.values())
            if total_decisions > 0:
                metrics["error_rate"] = decision_dist.get("skipped", 0) / total_decisions
                metrics["singleton_group_rate"] = decision_dist.get("new_group", 0) / total_decisions
        
        # Get quality metrics
        quality = self.metrics_collector.quality_metrics
        metrics["avg_group_size"] = quality.avg_group_size
        metrics["total_groups"] = float(quality.total_groups)
        
        # Get performance metrics
        performance = self.metrics_collector.performance_metrics
        metrics["stories_per_minute"] = performance.stories_per_minute
        
        # Get cost metrics
        cost = self.metrics_collector.cost_metrics
        metrics["total_cost_usd"] = cost.total_cost_usd
        
        # Calculate hourly/daily cost estimates
        elapsed_hours = (time.perf_counter() - self.metrics_collector._start_time) / 3600
        if elapsed_hours > 0:
            metrics["hourly_cost_usd"] = cost.total_cost_usd / elapsed_hours
            metrics["daily_cost_usd"] = metrics["hourly_cost_usd"] * 24
        
        return metrics

    def _is_threshold_breached(self, current_value: float, threshold: AlertThreshold) -> bool:
        """Check if a threshold is breached."""
        if threshold.comparison == "gt":
            return current_value > threshold.threshold_value
        elif threshold.comparison == "lt":
            return current_value < threshold.threshold_value
        elif threshold.comparison == "eq":
            return abs(current_value - threshold.threshold_value) < 0.001
        return False

    def _is_in_cooldown(self, metric_name: str, cooldown_minutes: int) -> bool:
        """Check if an alert is in cooldown period."""
        if metric_name not in self._last_alert_times:
            return False
            
        last_alert = self._last_alert_times[metric_name]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        return datetime.now(timezone.utc) - last_alert < cooldown_period

    def _create_alert(self, threshold: AlertThreshold, current_value: float) -> Alert:
        """Create an alert from a threshold breach."""
        alert_id = f"{threshold.metric_name}_{int(time.time())}"
        
        return Alert(
            alert_id=alert_id,
            severity=threshold.severity,
            title=f"Story Grouping Alert: {threshold.metric_name}",
            description=threshold.description,
            metric_name=threshold.metric_name,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            timestamp=datetime.now(timezone.utc),
            tags={
                "component": "story_grouping",
                "metric": threshold.metric_name,
                "severity": threshold.severity.value,
            },
        )

    def _activate_alert(self, alert: Alert) -> None:
        """Activate an alert and notify callbacks."""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self._last_alert_times[alert.metric_name] = alert.timestamp
        
        # Log the alert
        logger.warning(
            f"Story grouping alert triggered: {alert.title} - "
            f"{alert.description} (current: {alert.current_value:.2f}, "
            f"threshold: {alert.threshold_value:.2f})"
        )
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Manually resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolution_timestamp = datetime.now(timezone.utc)
        
        if resolution_note:
            alert.tags["resolution_note"] = resolution_note
        
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert.title} - {resolution_note}")
        return True

    def auto_resolve_alerts(self) -> List[str]:
        """Automatically resolve alerts when conditions improve."""
        resolved_alerts = []
        current_metrics = self._calculate_current_metrics()
        
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.metric_name in current_metrics:
                current_value = current_metrics[alert.metric_name]
                
                # Find the corresponding threshold
                threshold = next(
                    (t for t in self.thresholds if t.metric_name == alert.metric_name),
                    None
                )
                
                if threshold and not self._is_threshold_breached(current_value, threshold):
                    self.resolve_alert(alert_id, "Metric returned to normal range")
                    resolved_alerts.append(alert_id)
        
        return resolved_alerts

    def detect_anomalies(self, window_size: int = 100) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in recent grouping patterns."""
        anomalies = []
        
        if len(self.metrics_collector._recent_decisions) < window_size:
            return anomalies
        
        recent_decisions = self.metrics_collector._recent_decisions[-window_size:]
        
        # Analyze similarity score anomalies
        similarity_scores = [d.similarity_score for d in recent_decisions if d.similarity_score is not None]
        if len(similarity_scores) >= 10:
            anomalies.extend(self._detect_similarity_anomalies(similarity_scores))
        
        # Analyze processing time anomalies
        processing_times = [d.processing_time_ms for d in recent_decisions]
        if processing_times:
            anomalies.extend(self._detect_processing_time_anomalies(processing_times))
        
        # Analyze grouping pattern anomalies
        anomalies.extend(self._detect_grouping_pattern_anomalies(recent_decisions))
        
        return anomalies

    def _detect_similarity_anomalies(self, scores: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in similarity scores using statistical methods."""
        anomalies = []
        
        if len(scores) < 10:
            return anomalies
        
        mean_score = statistics.mean(scores)
        stdev_score = statistics.stdev(scores)
        
        # Detect unusually low average similarity
        if mean_score < 0.4:
            anomalies.append({
                "type": "low_similarity_scores",
                "description": f"Average similarity score is unusually low: {mean_score:.3f}",
                "severity": "warning",
                "metric": "avg_similarity_score",
                "value": mean_score,
            })
        
        # Detect high variance in similarity scores
        if stdev_score > 0.3:
            anomalies.append({
                "type": "high_similarity_variance",
                "description": f"High variance in similarity scores: {stdev_score:.3f}",
                "severity": "info",
                "metric": "similarity_variance",
                "value": stdev_score,
            })
        
        return anomalies

    def _detect_processing_time_anomalies(self, times: List[int]) -> List[Dict[str, Any]]:
        """Detect anomalies in processing times."""
        anomalies = []
        
        if len(times) < 5:
            return anomalies
        
        mean_time = statistics.mean(times)
        
        # Detect unusually slow processing
        if mean_time > 8000:  # More than 8 seconds
            anomalies.append({
                "type": "slow_processing",
                "description": f"Processing time is unusually slow: {mean_time:.0f}ms",
                "severity": "warning",
                "metric": "avg_processing_time",
                "value": mean_time,
            })
        
        # Detect processing time spikes
        if len(times) >= 10:
            recent_avg = statistics.mean(times[-10:])
            older_avg = statistics.mean(times[:-10])
            
            if recent_avg > older_avg * 2:  # Recent average is 2x older average
                anomalies.append({
                    "type": "processing_time_spike",
                    "description": f"Recent processing times increased significantly: {recent_avg:.0f}ms vs {older_avg:.0f}ms",
                    "severity": "warning",
                    "metric": "processing_time_trend",
                    "value": recent_avg / older_avg,
                })
        
        return anomalies

    def _detect_grouping_pattern_anomalies(self, decisions: List[GroupingDecision]) -> List[Dict[str, Any]]:
        """Detect anomalies in grouping decision patterns."""
        anomalies = []
        
        # Count decision types
        decision_counts = {"new_group": 0, "existing_group": 0, "skipped": 0}
        for decision in decisions:
            decision_counts[decision.decision_type] += 1
        
        total_decisions = len(decisions)
        
        # Detect excessive new group creation
        new_group_rate = decision_counts["new_group"] / total_decisions
        if new_group_rate > 0.85:
            anomalies.append({
                "type": "excessive_new_groups",
                "description": f"High rate of new group creation: {new_group_rate:.1%}",
                "severity": "warning",
                "metric": "new_group_rate",
                "value": new_group_rate,
            })
        
        # Detect high skip rate
        skip_rate = decision_counts["skipped"] / total_decisions
        if skip_rate > 0.2:
            anomalies.append({
                "type": "high_skip_rate",
                "description": f"High rate of skipped stories: {skip_rate:.1%}",
                "severity": "error",
                "metric": "skip_rate",
                "value": skip_rate,
            })
        
        return anomalies

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of current alerts and alert history."""
        active_by_severity = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
        
        recent_history = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]
        
        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": active_by_severity,
            "total_alerts_24h": len(recent_history),
            "alert_rate_per_hour": len(recent_history) / 24,
            "most_frequent_alerts": self._get_most_frequent_alert_types(recent_history),
            "current_anomalies": len(self.detect_anomalies()),
        }

    def _get_most_frequent_alert_types(self, alerts: List[Alert]) -> Dict[str, int]:
        """Get the most frequently triggered alert types."""
        alert_counts = {}
        for alert in alerts:
            metric = alert.metric_name
            alert_counts[metric] = alert_counts.get(metric, 0) + 1
        
        # Return top 5 most frequent
        return dict(sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)[:5])