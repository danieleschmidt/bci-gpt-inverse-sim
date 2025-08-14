"""
Advanced monitoring and observability for BCI-GPT system.
Provides real-time metrics, performance tracking, and alerting.
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
import json


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: Callable[[float], bool]
    threshold: float
    severity: str = "warning"
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    message_template: str = "Alert {name}: value {value} exceeded threshold {threshold}"


class MetricsCollector:
    """High-performance metrics collection and aggregation."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_points_per_metric: Maximum data points to keep per metric
        """
        self.max_points = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_points))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            full_name = self._build_metric_name(name, tags)
            self.counters[full_name] += value
            self._add_point(full_name, self.counters[full_name], tags)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            full_name = self._build_metric_name(name, tags)
            self.gauges[full_name] = value
            self._add_point(full_name, value, tags)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            full_name = self._build_metric_name(name, tags)
            self.histograms[full_name].append(value)
            
            # Keep only recent values for histograms
            if len(self.histograms[full_name]) > 1000:
                self.histograms[full_name] = self.histograms[full_name][-1000:]
            
            self._add_point(full_name, value, tags)
    
    def timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric in milliseconds."""
        self.histogram(f"{name}_duration_ms", duration_ms, tags)
    
    def _build_metric_name(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Build full metric name with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_str}"
    
    def _add_point(self, name: str, value: float, tags: Optional[Dict[str, str]]):
        """Add a data point to the time series."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(point)
    
    def get_metric_summary(self, name: str, minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric over the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Find all metrics matching the name (with different tags)
        matching_metrics = [k for k in self.metrics.keys() if k.startswith(name)]
        
        if not matching_metrics:
            return {'error': f'No metrics found for {name}'}
        
        # Aggregate data from all matching metrics
        recent_values = []
        for metric_name in matching_metrics:
            points = self.metrics[metric_name]
            for point in points:
                if point.timestamp > cutoff_time:
                    recent_values.append(point.value)
        
        if not recent_values:
            return {'error': f'No recent data for {name}'}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'avg': sum(recent_values) / len(recent_values),
            'latest': recent_values[-1] if recent_values else None,
            'period_minutes': minutes
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get current values for all metrics."""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_counts': {k: len(v) for k, v in self.histograms.items()},
                'total_metrics': len(self.metrics),
                'timestamp': datetime.now().isoformat()
            }


class PerformanceProfiler:
    """Detailed performance profiling for BCI-GPT components."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance profiler.
        
        Args:
            metrics_collector: Metrics collector to send data to
        """
        self.metrics = metrics_collector
        self.active_spans: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def span(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        span_id = f"{operation_name}_{threading.get_ident()}_{time.time()}"
        start_time = time.time()
        
        with self._lock:
            self.active_spans[span_id] = datetime.now()
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                if span_id in self.active_spans:
                    del self.active_spans[span_id]
            
            # Record timing metrics
            self.metrics.timing(operation_name, duration_ms, tags)
            self.metrics.counter(f"{operation_name}_calls", 1, tags)
    
    def profile_function(self, func_name: str, tags: Optional[Dict[str, str]] = None):
        """Decorator to profile function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.span(func_name, tags):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_active_spans(self) -> Dict[str, str]:
        """Get currently active performance spans."""
        with self._lock:
            return {
                span_id: start_time.isoformat()
                for span_id, start_time in self.active_spans.items()
            }


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Source of metrics data
        """
        self.metrics = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.triggered_alerts: List[Dict] = []
        
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_alert(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        condition: str = "greater_than",
        severity: str = "warning",
        cooldown_minutes: int = 5
    ):
        """Add a new alert configuration.
        
        Args:
            name: Unique alert name
            metric_name: Name of metric to monitor
            threshold: Alert threshold value
            condition: Condition type (greater_than, less_than, equals)
            severity: Alert severity (info, warning, error, critical)
            cooldown_minutes: Minimum time between alerts
        """
        condition_func = self._build_condition_func(condition, threshold)
        
        alert = Alert(
            name=name,
            condition=condition_func,
            threshold=threshold,
            severity=severity,
            cooldown_minutes=cooldown_minutes,
            message_template=f"Alert {name}: {metric_name} value {{value}} {condition} {threshold}"
        )
        
        self.alerts[f"{name}:{metric_name}"] = alert
        self.logger.info(f"Added alert: {name} for metric {metric_name}")
    
    def _build_condition_func(self, condition: str, threshold: float) -> Callable[[float], bool]:
        """Build condition function from string description."""
        if condition == "greater_than":
            return lambda x: x > threshold
        elif condition == "less_than":
            return lambda x: x < threshold
        elif condition == "equals":
            return lambda x: abs(x - threshold) < 0.001
        elif condition == "not_equals":
            return lambda x: abs(x - threshold) >= 0.001
        else:
            raise ValueError(f"Unknown condition: {condition}")
    
    def start_monitoring(self, check_interval_seconds: int = 60):
        """Start background alert monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self._check_thread.start()
        self.logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background alert monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join()
        self.logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_all_alerts()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _check_all_alerts(self):
        """Check all registered alerts."""
        current_time = datetime.now()
        
        for alert_key, alert in self.alerts.items():
            try:
                metric_name = alert_key.split(":", 1)[1]
                metric_summary = self.metrics.get_metric_summary(metric_name, minutes=5)
                
                if 'error' in metric_summary:
                    continue
                
                current_value = metric_summary.get('latest')
                if current_value is None:
                    continue
                
                # Check if alert condition is met
                if alert.condition(current_value):
                    # Check cooldown
                    if (alert.last_triggered is None or 
                        (current_time - alert.last_triggered).total_seconds() >= alert.cooldown_minutes * 60):
                        
                        # Trigger alert
                        self._trigger_alert(alert, current_value, metric_summary)
                        alert.last_triggered = current_time
                        
            except Exception as e:
                self.logger.error(f"Error checking alert {alert_key}: {e}")
    
    def _trigger_alert(self, alert: Alert, value: float, metric_summary: Dict):
        """Trigger an alert."""
        alert_data = {
            'name': alert.name,
            'severity': alert.severity,
            'threshold': alert.threshold,
            'current_value': value,
            'metric_summary': metric_summary,
            'timestamp': datetime.now().isoformat(),
            'message': alert.message_template.format(
                name=alert.name,
                value=value,
                threshold=alert.threshold
            )
        }
        
        self.triggered_alerts.append(alert_data)
        
        # Keep only recent alerts
        if len(self.triggered_alerts) > 1000:
            self.triggered_alerts = self.triggered_alerts[-1000:]
        
        # Log alert
        log_level = getattr(logging, alert.severity.upper(), logging.WARNING)
        self.logger.log(log_level, alert_data['message'])
        
        # Could add additional notification methods here (email, Slack, etc.)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.triggered_alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]


class SystemMonitor:
    """Comprehensive system monitoring facade."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.metrics = MetricsCollector()
        self.profiler = PerformanceProfiler(self.metrics)
        self.alerts = AlertManager(self.metrics)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup basic system alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default system alerts."""
        # Memory usage alert
        self.alerts.add_alert(
            name="high_memory_usage",
            metric_name="system_memory_percent",
            threshold=85.0,
            condition="greater_than",
            severity="warning",
            cooldown_minutes=10
        )
        
        # CPU usage alert
        self.alerts.add_alert(
            name="high_cpu_usage", 
            metric_name="system_cpu_percent",
            threshold=90.0,
            condition="greater_than",
            severity="warning",
            cooldown_minutes=5
        )
        
        # Error rate alert
        self.alerts.add_alert(
            name="high_error_rate",
            metric_name="error_rate_per_minute",
            threshold=10.0,
            condition="greater_than",
            severity="error",
            cooldown_minutes=5
        )
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.alerts.start_monitoring()
        
        # Start periodic system metrics collection
        self._start_system_metrics()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.alerts.stop_monitoring()
        self.logger.info("System monitoring stopped")
    
    def _start_system_metrics(self):
        """Start collecting system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU and memory metrics
                    try:
                        import psutil
                        
                        cpu_percent = psutil.cpu_percent(interval=1)
                        memory = psutil.virtual_memory()
                        disk = psutil.disk_usage('/')
                        
                        self.metrics.gauge("system_cpu_percent", cpu_percent)
                        self.metrics.gauge("system_memory_percent", memory.percent)
                        self.metrics.gauge("system_disk_percent", disk.percent)
                        
                    except ImportError:
                        pass  # psutil not available
                    
                    time.sleep(60)  # Collect every minute
                    
                except Exception as e:
                    self.logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)
        
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'metrics_summary': self.metrics.get_all_metrics(),
            'active_spans': self.profiler.get_active_spans(),
            'recent_alerts': self.alerts.get_recent_alerts(hours=1),
            'system_status': self._get_system_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except ImportError:
            return {'error': 'System metrics not available (psutil not installed)'}


# Global monitor instance
_system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return _system_monitor


def monitor_performance(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance."""
    return _system_monitor.profiler.profile_function(operation_name, tags)


# Context managers for easy monitoring
@contextmanager
def monitor_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for monitoring operations."""
    with _system_monitor.profiler.span(operation_name, tags):
        yield


def record_metric(metric_type: str, name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a metric value."""
    if metric_type == "counter":
        _system_monitor.metrics.counter(name, int(value), tags)
    elif metric_type == "gauge":
        _system_monitor.metrics.gauge(name, value, tags)
    elif metric_type == "histogram":
        _system_monitor.metrics.histogram(name, value, tags)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")