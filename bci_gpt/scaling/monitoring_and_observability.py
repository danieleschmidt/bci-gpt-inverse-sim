"""Comprehensive monitoring and observability system for BCI-GPT production scaling."""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricDataPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    duration_seconds: int = 300
    active: bool = False
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class HealthCheck:
    """Service health check configuration."""
    name: str
    endpoint: str
    interval_seconds: int = 30
    timeout_seconds: int = 10
    expected_status: int = 200
    expected_response: Optional[str] = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

class MetricsCollector:
    """Collects and stores metrics for monitoring."""
    
    def __init__(self, retention_days: int = 7):
        """Initialize metrics collector.
        
        Args:
            retention_days: How long to retain metrics data
        """
        self.retention_days = retention_days
        self.retention_seconds = retention_days * 24 * 3600
        
        # Metrics storage: {metric_name: deque of MetricDataPoint}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_types: Dict[str, MetricType] = {}
        
        # Aggregated metrics cache
        self._aggregation_cache: Dict[str, Dict[str, float]] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # seconds
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric.
        
        Args:
            name: Metric name
            value: Value to add to counter
            labels: Optional metric labels
        """
        self._record_metric(name, value, MetricType.COUNTER, labels)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric.
        
        Args:
            name: Metric name
            value: Current gauge value
            labels: Optional metric labels
        """
        self._record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Optional metric labels
        """
        self._record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Record a metric data point.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels
        """
        timestamp = time.time()
        data_point = MetricDataPoint(timestamp=timestamp, value=value, labels=labels or {})
        
        self.metrics[name].append(data_point)
        self.metric_types[name] = metric_type
        
        # Clear cache when new data arrives
        self._cache_timestamp = 0
    
    def get_metric_values(self, name: str, start_time: float = None, end_time: float = None) -> List[MetricDataPoint]:
        """Get metric values within time range.
        
        Args:
            name: Metric name
            start_time: Start timestamp (default: 1 hour ago)
            end_time: End timestamp (default: now)
            
        Returns:
            List of metric data points
        """
        if name not in self.metrics:
            return []
        
        now = time.time()
        start_time = start_time or (now - 3600)  # Default: 1 hour ago
        end_time = end_time or now
        
        return [
            point for point in self.metrics[name]
            if start_time <= point.timestamp <= end_time
        ]
    
    def get_metric_aggregation(self, name: str, aggregation: str = "avg", window_seconds: int = 300) -> Optional[float]:
        """Get aggregated metric value.
        
        Args:
            name: Metric name
            aggregation: Aggregation type ("avg", "sum", "min", "max", "count")
            window_seconds: Time window for aggregation
            
        Returns:
            Aggregated value or None if no data
        """
        cache_key = f"{name}_{aggregation}_{window_seconds}"
        now = time.time()
        
        # Use cache if available and fresh
        if (now - self._cache_timestamp < self._cache_ttl and 
            cache_key in self._aggregation_cache):
            return self._aggregation_cache[cache_key]
        
        # Calculate aggregation
        start_time = now - window_seconds
        data_points = self.get_metric_values(name, start_time, now)
        
        if not data_points:
            return None
        
        values = [point.value for point in data_points]
        
        if aggregation == "avg":
            result = statistics.mean(values)
        elif aggregation == "sum":
            result = sum(values)
        elif aggregation == "min":
            result = min(values)
        elif aggregation == "max":
            result = max(values)
        elif aggregation == "count":
            result = len(values)
        else:
            logger.warning(f"Unknown aggregation type: {aggregation}")
            return None
        
        # Update cache
        if now - self._cache_timestamp >= self._cache_ttl:
            self._aggregation_cache.clear()
            self._cache_timestamp = now
        
        self._aggregation_cache[cache_key] = result
        return result
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - self.retention_seconds
        
        for name, metric_deque in self.metrics.items():
            # Remove old data points
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()
        
        logger.info(f"Cleaned up metrics older than {self.retention_days} days")

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self._evaluation_thread: Optional[threading.Thread] = None
        self._stop_evaluation = threading.Event()
        
    def register_alert(self, alert: Alert) -> bool:
        """Register a new alert.
        
        Args:
            alert: Alert configuration
            
        Returns:
            True if registration successful
        """
        try:
            self.alerts[alert.name] = alert
            logger.info(f"Registered alert: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register alert {alert.name}: {e}")
            return False
    
    def start_alert_evaluation(self, interval_seconds: int = 30):
        """Start background alert evaluation.
        
        Args:
            interval_seconds: Evaluation interval
        """
        if self._evaluation_thread and self._evaluation_thread.is_alive():
            logger.warning("Alert evaluation already running")
            return
        
        self._stop_evaluation.clear()
        self._evaluation_thread = threading.Thread(
            target=self._evaluate_alerts_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._evaluation_thread.start()
        logger.info("Started alert evaluation")
    
    def stop_alert_evaluation(self):
        """Stop background alert evaluation."""
        if self._evaluation_thread:
            self._stop_evaluation.set()
            self._evaluation_thread.join(timeout=5)
            logger.info("Stopped alert evaluation")
    
    def _evaluate_alerts_loop(self, interval_seconds: int):
        """Background loop for evaluating alerts."""
        while not self._stop_evaluation.wait(interval_seconds):
            try:
                self.evaluate_alerts()
            except Exception as e:
                logger.error(f"Alert evaluation failed: {e}")
    
    def evaluate_alerts(self):
        """Evaluate all registered alerts."""
        current_time = time.time()
        
        for alert_name, alert in self.alerts.items():
            try:
                # Simple condition evaluation - in production would support complex expressions
                if ">" in alert.condition:
                    metric_name, threshold_str = alert.condition.split(">")
                    metric_name = metric_name.strip()
                    threshold = float(threshold_str.strip())
                    
                    current_value = self.metrics_collector.get_metric_aggregation(
                        metric_name, "avg", alert.duration_seconds
                    )
                    
                    if current_value is not None and current_value > threshold:
                        self._trigger_alert(alert, current_time, current_value)
                    else:
                        self._resolve_alert(alert, current_time)
                
                elif "<" in alert.condition:
                    metric_name, threshold_str = alert.condition.split("<")
                    metric_name = metric_name.strip()
                    threshold = float(threshold_str.strip())
                    
                    current_value = self.metrics_collector.get_metric_aggregation(
                        metric_name, "avg", alert.duration_seconds
                    )
                    
                    if current_value is not None and current_value < threshold:
                        self._trigger_alert(alert, current_time, current_value)
                    else:
                        self._resolve_alert(alert, current_time)
                        
            except Exception as e:
                logger.error(f"Failed to evaluate alert {alert_name}: {e}")
    
    def _trigger_alert(self, alert: Alert, timestamp: float, value: float):
        """Trigger an alert.
        
        Args:
            alert: Alert to trigger
            timestamp: Trigger timestamp
            value: Current metric value
        """
        if not alert.active:
            alert.active = True
            alert.triggered_at = timestamp
            alert.resolved_at = None
            
            alert_event = {
                "alert_name": alert.name,
                "severity": alert.severity.value,
                "description": alert.description,
                "condition": alert.condition,
                "current_value": value,
                "threshold": alert.threshold,
                "triggered_at": timestamp,
                "action": "triggered",
                "labels": alert.labels
            }
            
            self.alert_history.append(alert_event)
            
            logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description} "
                          f"(value: {value}, threshold: {alert.threshold})")
            
            # In production, would send notifications (email, Slack, PagerDuty, etc.)
            self._send_notification(alert_event)
    
    def _resolve_alert(self, alert: Alert, timestamp: float):
        """Resolve an alert.
        
        Args:
            alert: Alert to resolve
            timestamp: Resolution timestamp
        """
        if alert.active:
            alert.active = False
            alert.resolved_at = timestamp
            
            alert_event = {
                "alert_name": alert.name,
                "severity": alert.severity.value,
                "description": alert.description,
                "resolved_at": timestamp,
                "duration_seconds": timestamp - (alert.triggered_at or timestamp),
                "action": "resolved",
                "labels": alert.labels
            }
            
            self.alert_history.append(alert_event)
            
            logger.info(f"ALERT RESOLVED: {alert.name}")
            self._send_notification(alert_event)
    
    def _send_notification(self, alert_event: Dict[str, Any]):
        """Send alert notification.
        
        Args:
            alert_event: Alert event details
        """
        # In production, would integrate with notification systems
        # For now, just log the notification
        action = alert_event["action"].upper()
        severity = alert_event["severity"].upper()
        logger.info(f"NOTIFICATION: {action} {severity} alert: {alert_event['alert_name']}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts.
        
        Returns:
            List of active alert details
        """
        return [
            {
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "triggered_at": alert.triggered_at,
                "duration_seconds": time.time() - (alert.triggered_at or time.time()),
                "labels": alert.labels
            }
            for alert in self.alerts.values()
            if alert.active
        ]

class HealthCheckMonitor:
    """Monitors service health through HTTP health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize health check monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    def register_health_check(self, health_check: HealthCheck) -> bool:
        """Register a health check.
        
        Args:
            health_check: Health check configuration
            
        Returns:
            True if registration successful
        """
        try:
            self.health_checks[health_check.name] = health_check
            logger.info(f"Registered health check: {health_check.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register health check {health_check.name}: {e}")
            return False
    
    def start_monitoring(self):
        """Start health check monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started health check monitoring")
    
    def stop_monitoring(self):
        """Stop health check monitoring."""
        if self._monitor_thread:
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=5)
            logger.info("Stopped health check monitoring")
    
    def _monitoring_loop(self):
        """Background loop for health check monitoring."""
        while not self._stop_monitoring.is_set():
            for name, check in self.health_checks.items():
                try:
                    self._perform_health_check(name, check)
                except Exception as e:
                    logger.error(f"Health check {name} failed: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def _perform_health_check(self, name: str, check: HealthCheck):
        """Perform a single health check.
        
        Args:
            name: Health check name
            check: Health check configuration
        """
        start_time = time.time()
        
        try:
            # In production, would make actual HTTP request
            # Here we simulate the health check
            response_time = 0.05  # Simulate 50ms response time
            status_code = 200     # Simulate healthy response
            
            # Simulate occasional failures
            import random
            if random.random() < 0.05:  # 5% failure rate
                status_code = 503
                response_time = 5.0
            
            time.sleep(response_time)
            
            # Record metrics
            self.metrics_collector.record_histogram(
                f"health_check_response_time",
                response_time * 1000,  # Convert to ms
                labels={"service": name}
            )
            
            self.metrics_collector.record_gauge(
                f"health_check_status",
                1 if status_code == check.expected_status else 0,
                labels={"service": name}
            )
            
            # Store result
            is_healthy = status_code == check.expected_status
            self.check_results[name] = {
                "healthy": is_healthy,
                "status_code": status_code,
                "response_time_ms": response_time * 1000,
                "last_check": time.time(),
                "endpoint": check.endpoint
            }
            
            if not is_healthy:
                logger.warning(f"Health check failed: {name} (status: {status_code})")
            
        except Exception as e:
            self.check_results[name] = {
                "healthy": False,
                "error": str(e),
                "last_check": time.time(),
                "endpoint": check.endpoint
            }
            
            self.metrics_collector.record_gauge(
                f"health_check_status",
                0,
                labels={"service": name}
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Health status summary
        """
        total_checks = len(self.health_checks)
        healthy_checks = sum(1 for result in self.check_results.values() if result.get("healthy", False))
        
        return {
            "overall_healthy": healthy_checks == total_checks and total_checks > 0,
            "healthy_services": healthy_checks,
            "total_services": total_checks,
            "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            "service_details": self.check_results
        }

class MonitoringDashboard:
    """Comprehensive monitoring dashboard for BCI-GPT."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 alert_manager: AlertManager,
                 health_monitor: HealthCheckMonitor):
        """Initialize monitoring dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            alert_manager: Alert manager instance
            health_monitor: Health check monitor instance
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
        
        # Set up default BCI-GPT metrics and alerts
        self._setup_default_monitoring()
    
    def _setup_default_monitoring(self):
        """Set up default monitoring for BCI-GPT services."""
        # Default health checks
        health_checks = [
            HealthCheck(
                name="bci_inference_service",
                endpoint="http://bci-inference:8080/health",
                labels={"service": "inference", "critical": "true"}
            ),
            HealthCheck(
                name="eeg_processing_service",
                endpoint="http://eeg-processor:8080/health",
                labels={"service": "processing", "critical": "true"}
            ),
            HealthCheck(
                name="model_training_service",
                endpoint="http://model-trainer:8080/health",
                labels={"service": "training", "critical": "false"}
            )
        ]
        
        for check in health_checks:
            self.health_monitor.register_health_check(check)
        
        # Default alerts
        alerts = [
            Alert(
                name="high_inference_latency",
                description="BCI inference latency is too high",
                severity=AlertSeverity.HIGH,
                condition="inference_latency_ms > 100",
                threshold=100,
                duration_seconds=300
            ),
            Alert(
                name="low_model_accuracy",
                description="Model accuracy has dropped significantly",
                severity=AlertSeverity.CRITICAL,
                condition="model_accuracy < 0.85",
                threshold=0.85,
                duration_seconds=600
            ),
            Alert(
                name="high_memory_usage",
                description="Service memory usage is high",
                severity=AlertSeverity.MEDIUM,
                condition="memory_usage_percent > 85",
                threshold=85,
                duration_seconds=300
            ),
            Alert(
                name="service_down",
                description="Critical service is down",
                severity=AlertSeverity.CRITICAL,
                condition="health_check_status < 1",
                threshold=1,
                duration_seconds=60
            )
        ]
        
        for alert in alerts:
            self.alert_manager.register_alert(alert)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data.
        
        Returns:
            Dashboard data including metrics, alerts, and health status
        """
        # System metrics
        system_metrics = {}
        metric_names = [
            "inference_latency_ms",
            "model_accuracy",
            "memory_usage_percent",
            "cpu_usage_percent",
            "request_rate",
            "error_rate"
        ]
        
        for metric_name in metric_names:
            current_value = self.metrics_collector.get_metric_aggregation(metric_name, "avg", 300)
            system_metrics[metric_name] = {
                "current": current_value,
                "1h_avg": self.metrics_collector.get_metric_aggregation(metric_name, "avg", 3600),
                "24h_avg": self.metrics_collector.get_metric_aggregation(metric_name, "avg", 86400)
            }
        
        # Alert summary
        active_alerts = self.alert_manager.get_active_alerts()
        alert_summary = {
            "total_active": len(active_alerts),
            "critical": len([a for a in active_alerts if a["severity"] == "critical"]),
            "high": len([a for a in active_alerts if a["severity"] == "high"]),
            "medium": len([a for a in active_alerts if a["severity"] == "medium"]),
            "low": len([a for a in active_alerts if a["severity"] == "low"])
        }
        
        # Health status
        health_status = self.health_monitor.get_health_status()
        
        return {
            "timestamp": time.time(),
            "system_metrics": system_metrics,
            "active_alerts": active_alerts,
            "alert_summary": alert_summary,
            "health_status": health_status,
            "system_overview": {
                "overall_health": health_status["overall_healthy"] and alert_summary["critical"] == 0,
                "services_healthy": f"{health_status['healthy_services']}/{health_status['total_services']}",
                "critical_alerts": alert_summary["critical"]
            }
        }
    
    def generate_monitoring_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Monitoring report
        """
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Collect metrics for the time period
        report_metrics = {}
        for metric_name in ["inference_latency_ms", "model_accuracy", "memory_usage_percent", "cpu_usage_percent"]:
            data_points = self.metrics_collector.get_metric_values(metric_name, start_time, end_time)
            
            if data_points:
                values = [point.value for point in data_points]
                report_metrics[metric_name] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "data_points": len(values)
                }
            else:
                report_metrics[metric_name] = {"no_data": True}
        
        # Alert summary for period
        period_alerts = [
            alert for alert in self.alert_manager.alert_history
            if alert.get("triggered_at", 0) >= start_time
        ]
        
        alert_stats = {
            "total_alerts": len(period_alerts),
            "by_severity": {},
            "most_frequent": {}
        }
        
        # Count alerts by severity
        for alert in period_alerts:
            severity = alert["severity"]
            alert_stats["by_severity"][severity] = alert_stats["by_severity"].get(severity, 0) + 1
        
        # Count most frequent alerts
        for alert in period_alerts:
            name = alert["alert_name"]
            alert_stats["most_frequent"][name] = alert_stats["most_frequent"].get(name, 0) + 1
        
        return {
            "report_period": {
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": hours
            },
            "metrics_summary": report_metrics,
            "alert_summary": alert_stats,
            "current_health": self.health_monitor.get_health_status(),
            "recommendations": self._generate_recommendations(report_metrics, alert_stats),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any], alerts: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations based on data.
        
        Args:
            metrics: Metrics summary
            alerts: Alert summary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check inference latency
        if "inference_latency_ms" in metrics and "avg" in metrics["inference_latency_ms"]:
            avg_latency = metrics["inference_latency_ms"]["avg"]
            if avg_latency > 100:
                recommendations.append(f"High average inference latency ({avg_latency:.1f}ms). Consider model optimization or scaling.")
        
        # Check memory usage
        if "memory_usage_percent" in metrics and "avg" in metrics["memory_usage_percent"]:
            avg_memory = metrics["memory_usage_percent"]["avg"]
            if avg_memory > 80:
                recommendations.append(f"High memory usage ({avg_memory:.1f}%). Consider increasing memory limits or optimization.")
        
        # Check alert frequency
        if alerts["total_alerts"] > 50:
            recommendations.append(f"High alert volume ({alerts['total_alerts']} alerts). Review alert thresholds and system stability.")
        
        # Check critical alerts
        critical_alerts = alerts["by_severity"].get("critical", 0)
        if critical_alerts > 0:
            recommendations.append(f"Critical alerts detected ({critical_alerts}). Immediate attention required.")
        
        if not recommendations:
            recommendations.append("System appears healthy. Continue monitoring.")
        
        return recommendations

    def start_monitoring(self):
        """Start all monitoring components."""
        self.health_monitor.start_monitoring()
        self.alert_manager.start_alert_evaluation()
        logger.info("Started comprehensive monitoring")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.health_monitor.stop_monitoring()
        self.alert_manager.stop_alert_evaluation()
        logger.info("Stopped comprehensive monitoring")