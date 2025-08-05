"""
Advanced monitoring and logging system for BCI-GPT.

This module provides comprehensive monitoring capabilities including:
- Performance metrics tracking
- Error tracking and alerting  
- Resource monitoring
- Model performance monitoring
- Clinical safety monitoring
"""

import logging
import time
import psutil
import threading
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import json
import numpy as np

# Optional dependencies
try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    warnings.warn("TensorBoard not available for advanced logging")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("Weights & Biases not available for experiment tracking")

try:
    import opentelemetry
    from opentelemetry import trace, metrics
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    warnings.warn("OpenTelemetry not available for distributed tracing")


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: float
    inference_time_ms: float
    accuracy: Optional[float] = None
    confidence: Optional[float] = None
    word_error_rate: Optional[float] = None
    information_transfer_rate: Optional[float] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None


@dataclass
class StreamingMetrics:
    """Streaming performance metrics."""
    timestamp: float
    samples_per_second: float
    buffer_fill_ratio: float
    dropped_samples: int
    latency_ms: float
    jitter_ms: float
    signal_quality_score: Optional[float] = None


@dataclass
class ErrorEvent:
    """Error event tracking."""
    timestamp: float
    level: str  # 'warning', 'error', 'critical'
    component: str
    message: str
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 max_history_size: int = 10000):
        """Initialize metrics collector.
        
        Args:
            collection_interval: How often to collect metrics (seconds)
            max_history_size: Maximum number of metrics to keep in memory
        """
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        
        # Metrics storage
        self.system_metrics = deque(maxlen=max_history_size)
        self.model_metrics = deque(maxlen=max_history_size)
        self.streaming_metrics = deque(maxlen=max_history_size)
        self.error_events = deque(maxlen=max_history_size)
        
        # Collection state
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[ErrorEvent], None]] = []
        self.metric_callbacks: List[Callable[[Any], None]] = []
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            self.logger.warning("Metrics collection already started")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, 
            daemon=True
        )
        self.collection_thread.start()
        self.logger.info("Started metrics collection")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        self.logger.info("Stopped metrics collection")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Trigger callbacks
                for callback in self.metric_callbacks:
                    try:
                        callback(system_metrics)
                    except Exception as e:
                        self.logger.error(f"Metric callback error: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network
        network = psutil.net_io_counters()
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk_percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv
        )
        
        # GPU metrics if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            metrics.gpu_utilization = gpu_util.gpu
            metrics.gpu_memory_used_gb = gpu_memory.used / (1024**3)
            metrics.gpu_memory_total_gb = gpu_memory.total / (1024**3)
            
        except ImportError:
            pass  # GPU monitoring not available
        except Exception as e:
            self.logger.debug(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    def record_model_metrics(self, 
                           inference_time_ms: float,
                           accuracy: Optional[float] = None,
                           confidence: Optional[float] = None,
                           **kwargs) -> None:
        """Record model performance metrics."""
        metrics = ModelMetrics(
            timestamp=time.time(),
            inference_time_ms=inference_time_ms,
            accuracy=accuracy,
            confidence=confidence,
            **kwargs
        )
        self.model_metrics.append(metrics)
        
        # Trigger callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                self.logger.error(f"Model metrics callback error: {e}")
    
    def record_streaming_metrics(self,
                               samples_per_second: float,
                               buffer_fill_ratio: float,
                               dropped_samples: int = 0,
                               latency_ms: float = 0.0,
                               **kwargs) -> None:
        """Record streaming performance metrics."""
        metrics = StreamingMetrics(
            timestamp=time.time(),
            samples_per_second=samples_per_second,
            buffer_fill_ratio=buffer_fill_ratio,
            dropped_samples=dropped_samples,
            latency_ms=latency_ms,
            jitter_ms=kwargs.get('jitter_ms', 0.0),
            signal_quality_score=kwargs.get('signal_quality_score')
        )
        self.streaming_metrics.append(metrics)
        
        # Check for streaming issues
        if buffer_fill_ratio > 0.9:
            self.record_error(
                'warning', 'streaming', 
                f'High buffer fill ratio: {buffer_fill_ratio:.2f}',
                context={'buffer_fill_ratio': buffer_fill_ratio}
            )
        
        if dropped_samples > 0:
            self.record_error(
                'warning', 'streaming',
                f'Dropped {dropped_samples} samples',
                context={'dropped_samples': dropped_samples}
            )
    
    def record_error(self,
                    level: str,
                    component: str,
                    message: str,
                    exception: Optional[Exception] = None,
                    context: Optional[Dict[str, Any]] = None) -> None:
        """Record error event."""
        import traceback
        
        error_event = ErrorEvent(
            timestamp=time.time(),
            level=level,
            component=component,
            message=message,
            exception_type=type(exception).__name__ if exception else None,
            stack_trace=traceback.format_exc() if exception else None,
            context=context or {}
        )
        
        self.error_events.append(error_event)
        
        # Log the error
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"[{component}] {message}")
        
        # Trigger alert callbacks for errors and critical events
        if level in ['error', 'critical']:
            for callback in self.alert_callbacks:
                try:
                    callback(error_event)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        self.counters[name] += value
    
    def record_timing(self, name: str, duration_ms: float) -> None:
        """Record timing measurement."""
        self.timers[name].append(duration_ms)
        
        # Keep only recent timings
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-500:]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.system_metrics:
            return {"status": "no_data"}
        
        recent_metrics = self.system_metrics[-10:]  # Last 10 samples
        
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        
        # Determine health status
        status = "healthy"
        issues = []
        
        if avg_cpu > 90:
            status = "critical"
            issues.append("High CPU usage")
        elif avg_cpu > 70:
            status = "warning"
            issues.append("Elevated CPU usage")
        
        if avg_memory > 90:
            status = "critical"
            issues.append("High memory usage")
        elif avg_memory > 80:
            status = "warning"
            issues.append("Elevated memory usage")
        
        # Check for recent errors
        recent_errors = [e for e in self.error_events 
                        if time.time() - e.timestamp < 300]  # Last 5 minutes
        
        critical_errors = [e for e in recent_errors if e.level == 'critical']
        if critical_errors:
            status = "critical"
            issues.append(f"{len(critical_errors)} critical errors")
        
        error_count = len([e for e in recent_errors if e.level == 'error'])
        if error_count > 5:
            if status != "critical":
                status = "warning"
            issues.append(f"{error_count} recent errors")
        
        return {
            "status": status,
            "issues": issues,
            "cpu_percent": avg_cpu,
            "memory_percent": avg_memory,
            "error_count_5min": len(recent_errors),
            "uptime_seconds": time.time() - (recent_metrics[0].timestamp if recent_metrics else time.time())
        }
    
    def export_metrics(self, filepath: Union[str, Path], 
                      format: str = "json") -> None:
        """Export collected metrics to file."""
        filepath = Path(filepath)
        
        data = {
            "export_timestamp": time.time(),
            "system_metrics": [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_gb": m.memory_used_gb,
                    "gpu_utilization": m.gpu_utilization,
                    "gpu_memory_used_gb": m.gpu_memory_used_gb
                }
                for m in self.system_metrics
            ],
            "model_metrics": [
                {
                    "timestamp": m.timestamp,
                    "inference_time_ms": m.inference_time_ms,
                    "accuracy": m.accuracy,
                    "confidence": m.confidence,
                    "word_error_rate": m.word_error_rate
                }
                for m in self.model_metrics
            ],
            "streaming_metrics": [
                {
                    "timestamp": m.timestamp,
                    "samples_per_second": m.samples_per_second,
                    "buffer_fill_ratio": m.buffer_fill_ratio,
                    "dropped_samples": m.dropped_samples,
                    "latency_ms": m.latency_ms
                }
                for m in self.streaming_metrics
            ],
            "error_events": [
                {
                    "timestamp": e.timestamp,
                    "level": e.level,
                    "component": e.component,
                    "message": e.message,
                    "exception_type": e.exception_type
                }
                for e in self.error_events
            ],
            "counters": dict(self.counters),
            "timing_stats": {
                name: {
                    "count": len(timings),
                    "mean_ms": np.mean(timings),
                    "std_ms": np.std(timings),
                    "min_ms": np.min(timings),
                    "max_ms": np.max(timings),
                    "p95_ms": np.percentile(timings, 95)
                }
                for name, timings in self.timers.items()
                if timings
            }
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported metrics to {filepath}")


class PerformanceProfiler:
    """Context manager for performance profiling."""
    
    def __init__(self, 
                 name: str, 
                 metrics_collector: Optional[MetricsCollector] = None):
        self.name = name
        self.metrics_collector = metrics_collector
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return
        
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Record timing
        if self.metrics_collector:
            self.metrics_collector.record_timing(self.name, duration_ms)
        
        # Log if duration is unusually long
        if duration_ms > 1000:  # More than 1 second
            self.logger.warning(f"{self.name} took {duration_ms:.1f}ms")
        elif duration_ms > 100:  # More than 100ms
            self.logger.info(f"{self.name} took {duration_ms:.1f}ms")
        
        # Record any exceptions
        if exc_type is not None and self.metrics_collector:
            self.metrics_collector.record_error(
                'error', self.name,
                f"Exception in {self.name}: {exc_val}",
                exception=exc_val
            )


class ClinicalSafetyMonitor:
    """Clinical safety monitoring for BCI applications."""
    
    def __init__(self, 
                 max_session_duration: int = 3600,  # 1 hour
                 fatigue_threshold: float = 0.8,
                 seizure_detection_enabled: bool = True):
        """Initialize clinical safety monitor.
        
        Args:
            max_session_duration: Maximum session duration in seconds
            fatigue_threshold: Threshold for fatigue detection
            seizure_detection_enabled: Enable seizure detection
        """
        self.max_session_duration = max_session_duration
        self.fatigue_threshold = fatigue_threshold
        self.seizure_detection_enabled = seizure_detection_enabled
        
        # Session tracking
        self.session_start_time: Optional[float] = None
        self.session_user_id: Optional[str] = None
        self.break_recommendations = 0
        self.forced_breaks = 0
        
        # Safety state
        self.is_monitoring = False
        self.safety_violations = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, user_id: str) -> None:
        """Start monitoring a BCI session."""
        self.session_start_time = time.time()
        self.session_user_id = user_id
        self.is_monitoring = True
        self.break_recommendations = 0
        self.forced_breaks = 0
        
        self.logger.info(f"Started safety monitoring for user {user_id}")
    
    def end_session(self) -> Dict[str, Any]:
        """End session and return safety report."""
        if not self.is_monitoring:
            return {"error": "No active session"}
        
        session_duration = time.time() - (self.session_start_time or time.time())
        
        report = {
            "user_id": self.session_user_id,
            "session_duration_minutes": session_duration / 60,
            "break_recommendations": self.break_recommendations,
            "forced_breaks": self.forced_breaks,
            "safety_violations": len(self.safety_violations),
            "status": "completed_safely"
        }
        
        # Check for safety concerns
        if session_duration > self.max_session_duration:
            report["status"] = "exceeded_max_duration"
        
        if self.forced_breaks > 0:
            report["status"] = "safety_breaks_required"
        
        self.is_monitoring = False
        self.logger.info(f"Ended safety monitoring: {report}")
        
        return report
    
    def is_safe(self) -> bool:
        """Check if current session is safe to continue."""
        if not self.is_monitoring:
            return True
        
        # Check session duration
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            if duration > self.max_session_duration:
                self._record_violation("max_duration_exceeded")
                return False
        
        return True
    
    def detect_fatigue(self,
                      eeg_data: Optional[np.ndarray] = None,
                      performance_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Detect user fatigue from EEG or performance data."""
        if not self.is_monitoring:
            return False
        
        fatigue_indicators = []
        
        # EEG-based fatigue detection
        if eeg_data is not None:
            try:
                # Simplified fatigue detection using alpha/beta ratio
                # In real implementation, would use more sophisticated analysis
                alpha_power = np.mean(np.abs(eeg_data) ** 2)  # Simplified
                
                # Higher alpha activity may indicate drowsiness
                if alpha_power > self.fatigue_threshold:
                    fatigue_indicators.append("elevated_alpha")
                    
            except Exception as e:
                self.logger.error(f"EEG fatigue detection error: {e}")
        
        # Performance-based fatigue detection
        if performance_metrics:
            accuracy = performance_metrics.get('accuracy', 1.0)
            reaction_time = performance_metrics.get('reaction_time_ms', 0)
            
            if accuracy < 0.7:  # Decreased accuracy
                fatigue_indicators.append("low_accuracy")
            
            if reaction_time > 2000:  # Slow reactions
                fatigue_indicators.append("slow_reaction")
        
        # Determine fatigue level
        is_fatigued = len(fatigue_indicators) >= 2
        
        if is_fatigued:
            self._record_violation("fatigue_detected", 
                                 context={"indicators": fatigue_indicators})
            self.logger.warning(f"Fatigue detected: {fatigue_indicators}")
        
        return is_fatigued
    
    def recommend_break(self, duration_seconds: int = 300) -> None:
        """Recommend a break to the user."""
        self.break_recommendations += 1
        self.logger.info(f"Recommended {duration_seconds}s break (#{self.break_recommendations})")
    
    def enforce_break(self, duration_seconds: int = 300) -> None:
        """Enforce a mandatory break."""
        self.forced_breaks += 1
        self._record_violation("forced_break", 
                             context={"duration": duration_seconds})
        self.logger.warning(f"Enforced {duration_seconds}s break (#{self.forced_breaks})")
    
    def _record_violation(self, violation_type: str, 
                         context: Optional[Dict[str, Any]] = None) -> None:
        """Record a safety violation."""
        violation = {
            "timestamp": time.time(),
            "type": violation_type,
            "user_id": self.session_user_id,
            "context": context or {}
        }
        
        self.safety_violations.append(violation)
        self.logger.error(f"Safety violation: {violation}")


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
        _global_metrics_collector.start_collection()
    
    return _global_metrics_collector


def profile_performance(name: str) -> PerformanceProfiler:
    """Create performance profiler context manager."""
    return PerformanceProfiler(name, get_metrics_collector())


# Decorators for easy integration
def monitor_function(name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            with profile_performance(func_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def safe_function(max_retries: int = 3, 
                 backoff_factor: float = 1.0):
    """Decorator to add safety and retry logic to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics_collector = get_metrics_collector()
            
            for attempt in range(max_retries + 1):
                try:
                    with profile_performance(f"{func.__name__}_attempt_{attempt + 1}"):
                        result = func(*args, **kwargs)
                        
                    # Record success
                    metrics_collector.increment_counter(f"{func.__name__}_success")
                    return result
                    
                except Exception as e:
                    # Record failure
                    metrics_collector.increment_counter(f"{func.__name__}_failure")
                    metrics_collector.record_error(
                        'error', func.__name__,
                        f"Function failed on attempt {attempt + 1}: {e}",
                        exception=e
                    )
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        metrics_collector.record_error(
                            'critical', func.__name__,
                            f"Function failed after {max_retries + 1} attempts",
                            exception=e
                        )
                        raise
                    
                    # Wait before retry
                    time.sleep(backoff_factor * (2 ** attempt))
            
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Example of using the monitoring system
    logging.basicConfig(level=logging.INFO)
    
    # Get metrics collector
    collector = get_metrics_collector()
    
    # Add an alert callback
    def alert_callback(error_event: ErrorEvent):
        print(f"ALERT: {error_event.level.upper()} in {error_event.component}: {error_event.message}")
    
    collector.alert_callbacks.append(alert_callback)
    
    # Simulate some operations
    with profile_performance("test_operation"):
        time.sleep(0.1)  # Simulate work
        
        # Record some metrics
        collector.record_model_metrics(
            inference_time_ms=50.0,
            accuracy=0.85,
            confidence=0.9
        )
        
        collector.record_streaming_metrics(
            samples_per_second=1000.0,
            buffer_fill_ratio=0.3,
            latency_ms=10.0
        )
        
        # Simulate an error
        collector.record_error(
            'warning', 'test_component',
            'This is a test warning',
            context={'test': True}
        )
    
    # Check system health
    health = collector.get_system_health()
    print(f"System health: {health}")
    
    # Test clinical safety monitor
    safety_monitor = ClinicalSafetyMonitor()
    safety_monitor.start_session("test_user")
    
    # Simulate fatigue detection
    test_eeg = np.random.normal(0, 1, (8, 1000))
    is_fatigued = safety_monitor.detect_fatigue(test_eeg)
    print(f"Fatigue detected: {is_fatigued}")
    
    report = safety_monitor.end_session()
    print(f"Session report: {report}")
    
    # Stop collection
    collector.stop_collection()
    
    print("Monitoring system test completed!")