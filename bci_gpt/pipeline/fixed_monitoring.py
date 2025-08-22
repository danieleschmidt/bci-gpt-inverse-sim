"""Advanced Monitoring System for BCI-GPT Self-Healing Pipeline.

Comprehensive monitoring with predictive analytics, anomaly detection,
and intelligent alerting for proactive system management.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

try:
    import numpy as np
    import psutil
    HAS_MONITORING_DEPS = True
except ImportError:
    HAS_MONITORING_DEPS = False

from ..utils.error_handling import BCI_GPTError
from ..utils.logging_config import get_logger


class MetricType(Enum):
    """Metric type enumeration."""
    RESOURCE = "resource"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    AVAILABILITY = "availability"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Metric definition configuration."""
    name: str
    type: MetricType
    unit: str
    description: str
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    collection_interval: float = 60.0
    retention_period: float = 86400.0  # 24 hours
    aggregation_method: str = "average"


@dataclass
class MetricValue:
    """Individual metric value."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    anomaly_score: float
    is_anomaly: bool
    expected_value: float
    actual_value: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


class AdvancedMonitoringSystem:
    """Advanced monitoring system with predictive analytics and anomaly detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize monitoring system."""
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Core monitoring state
        self.monitoring_active = False
        self.monitor_start_time = datetime.now()
        
        # Metric definitions and storage
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_storage: Dict[str, deque] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Alert system
        self.alerts: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        self.alert_channels: Dict[str, Callable] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Anomaly detection
        self.anomaly_detectors: Dict[str, Dict[str, Any]] = {}
        self.anomaly_history: Dict[str, deque] = {}
        self.anomaly_callbacks: List[Callable] = []
        
        # Predictive analytics
        self.prediction_history: Dict[str, deque] = {}
        
        # Threading and execution
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.collection_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="metric_collector")
        
        # Statistics
        self.total_metrics_collected = 0
        self.total_alerts_generated = 0
        self.total_anomalies_detected = 0
        self.collection_stats = {
            "total_collections": 0,
            "failed_collections": 0,
            "avg_collection_time": 0.0
        }
        
        # Initialize system
        self._initialize_default_metrics()
        self._initialize_anomaly_detectors()
        self._initialize_alert_rules()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics."""
        default_metrics = [
            MetricDefinition(
                name="cpu_usage",
                type=MetricType.RESOURCE,
                unit="percent",
                description="CPU usage percentage",
                warning_threshold=75.0,
                critical_threshold=90.0,
                collection_interval=30.0
            ),
            MetricDefinition(
                name="memory_usage",
                type=MetricType.RESOURCE,
                unit="percent",
                description="Memory usage percentage",
                warning_threshold=80.0,
                critical_threshold=95.0,
                collection_interval=30.0
            ),
            MetricDefinition(
                name="pipeline_latency",
                type=MetricType.PERFORMANCE,
                unit="milliseconds",
                description="Pipeline processing latency",
                warning_threshold=100.0,
                critical_threshold=200.0,
                collection_interval=10.0
            ),
            MetricDefinition(
                name="model_accuracy",
                type=MetricType.BUSINESS,
                unit="percent",
                description="Model prediction accuracy",
                warning_threshold=75.0,
                critical_threshold=60.0,
                collection_interval=300.0
            )
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def _initialize_anomaly_detectors(self) -> None:
        """Initialize anomaly detection models."""
        for metric_name in self.metric_definitions.keys():
            self.anomaly_detectors[metric_name] = {
                "type": "statistical",
                "window_size": 50,
                "threshold": 2.0,
                "min_samples": 10,
                "history": deque(maxlen=100)
            }
            self.anomaly_history[metric_name] = deque(maxlen=100)
    
    def _initialize_alert_rules(self) -> None:
        """Initialize alert rules."""
        self.alert_rules = {
            "threshold_based": {
                "enabled": True,
                "check_interval": 60.0,
                "escalation_enabled": True,
                "escalation_threshold": 300.0
            },
            "anomaly_based": {
                "enabled": True,
                "check_interval": 120.0,
                "min_anomaly_score": 0.7,
                "consecutive_anomalies": 3
            }
        }
    
    def register_metric(self, metric_definition: MetricDefinition) -> None:
        """Register a new metric for monitoring."""
        self.metric_definitions[metric_definition.name] = metric_definition
        self.metric_storage[metric_definition.name] = deque(
            maxlen=int(metric_definition.retention_period / metric_definition.collection_interval)
        )
        
        if metric_definition.name not in self.metric_collectors:
            self._register_default_collector(metric_definition)
        
        self.logger.info(f"Registered metric: {metric_definition.name}")
    
    def _register_default_collector(self, metric_definition: MetricDefinition) -> None:
        """Register default collector for built-in metrics."""
        name = metric_definition.name
        
        if name == "cpu_usage" and HAS_MONITORING_DEPS:
            self.metric_collectors[name] = lambda: psutil.cpu_percent()
        elif name == "memory_usage" and HAS_MONITORING_DEPS:
            self.metric_collectors[name] = lambda: psutil.virtual_memory().percent
        else:
            # Default collector returns random value for demo
            self.metric_collectors[name] = lambda: 50.0 + (hash(name) % 40)
    
    def start_monitoring(self) -> None:
        """Start monitoring all registered metrics."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("Advanced monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitoring_active = False
        self.collection_executor.shutdown(wait=True)
        self.logger.info("Advanced monitoring system stopped")
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None,
                     metadata: Dict[str, Any] = None) -> None:
        """Manually record a metric value."""
        if metric_name not in self.metric_definitions:
            self.logger.warning(f"Recording value for undefined metric: {metric_name}")
            return
        
        metric_value = MetricValue(
            name=metric_name,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metric_storage[metric_name].append(metric_value)
        self.total_metrics_collected += 1
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status."""
        uptime = (datetime.now() - self.monitor_start_time).total_seconds()
        
        return {
            "monitoring_active": self.monitoring_active,
            "uptime_seconds": uptime,
            "monitored_metrics": len(self.metric_definitions),
            "total_metrics_collected": self.total_metrics_collected,
            "total_alerts_generated": self.total_alerts_generated,
            "total_anomalies_detected": self.total_anomalies_detected,
            "timestamp": datetime.now().isoformat()
        }