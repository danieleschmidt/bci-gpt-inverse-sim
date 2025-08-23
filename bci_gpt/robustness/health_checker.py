"""
Comprehensive Health Monitoring for BCI-GPT System

Provides real-time health checks, system monitoring,
and alerting for production BCI deployments.
"""

import time
import threading
import asyncio
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component types for health monitoring"""
    EEG_PROCESSOR = "eeg_processor"
    MODEL_INFERENCE = "model_inference"
    DATA_PIPELINE = "data_pipeline"
    STORAGE = "storage"
    NETWORK = "network"
    COMPUTE_RESOURCE = "compute_resource"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms
        }


class HealthCheck(ABC):
    """Abstract base class for health checks"""
    
    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform health check"""
        pass
    
    @abstractmethod
    def get_timeout(self) -> float:
        """Get check timeout in seconds"""
        pass


class SystemResourceCheck(HealthCheck):
    """Check system resource usage"""
    
    def __init__(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0
    ):
        super().__init__("system_resources", ComponentType.COMPUTE_RESOURCE)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def check(self) -> HealthCheckResult:
        """Check system resources"""
        start_time = time.time()
        
        try:
            # Mock system resource check (would use psutil in production)
            cpu_usage = 45.0  # Mock CPU usage
            memory_usage = 62.0  # Mock memory usage
            disk_usage = 55.0  # Mock disk usage
            
            details = {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory_usage,
                'disk_usage_percent': disk_usage,
                'thresholds': {
                    'cpu': self.cpu_threshold,
                    'memory': self.memory_threshold,
                    'disk': self.disk_threshold
                }
            }
            
            # Determine status
            if (cpu_usage > self.cpu_threshold or 
                memory_usage > self.memory_threshold or
                disk_usage > self.disk_threshold):
                status = HealthStatus.UNHEALTHY
                message = "Resource usage exceeds thresholds"
            elif (cpu_usage > self.cpu_threshold * 0.8 or
                  memory_usage > self.memory_threshold * 0.8 or
                  disk_usage > self.disk_threshold * 0.8):
                status = HealthStatus.DEGRADED
                message = "Resource usage approaching thresholds"
            else:
                status = HealthStatus.HEALTHY
                message = "Resource usage within normal limits"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {e}",
                details={'error': str(e)},
                duration_ms=duration_ms
            )
    
    def get_timeout(self) -> float:
        return 5.0


class EEGProcessorCheck(HealthCheck):
    """Check EEG processor health"""
    
    def __init__(self, processor_instance=None):
        super().__init__("eeg_processor", ComponentType.EEG_PROCESSOR)
        self.processor = processor_instance
    
    def check(self) -> HealthCheckResult:
        """Check EEG processor"""
        start_time = time.time()
        
        try:
            # Mock EEG processor check
            if self.processor is None:
                # Simulate processor availability
                processor_available = True
                buffer_size = 1000
                processing_rate = 950.0  # samples/sec
                latency_ms = 45.0
            else:
                # Would check actual processor here
                processor_available = True
                buffer_size = getattr(self.processor, 'buffer_size', 1000)
                processing_rate = getattr(self.processor, 'processing_rate', 950.0)
                latency_ms = getattr(self.processor, 'avg_latency_ms', 45.0)
            
            details = {
                'processor_available': processor_available,
                'buffer_size': buffer_size,
                'processing_rate_hz': processing_rate,
                'average_latency_ms': latency_ms
            }
            
            if not processor_available:
                status = HealthStatus.UNHEALTHY
                message = "EEG processor not available"
            elif latency_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"EEG processing latency high: {latency_ms:.1f}ms"
            elif processing_rate < 900:
                status = HealthStatus.DEGRADED
                message = f"EEG processing rate low: {processing_rate:.1f}Hz"
            else:
                status = HealthStatus.HEALTHY
                message = "EEG processor operating normally"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"EEG processor check failed: {e}",
                details={'error': str(e)},
                duration_ms=duration_ms
            )
    
    def get_timeout(self) -> float:
        return 3.0


class ModelInferenceCheck(HealthCheck):
    """Check model inference health"""
    
    def __init__(self, model_instance=None):
        super().__init__("model_inference", ComponentType.MODEL_INFERENCE)
        self.model = model_instance
    
    def check(self) -> HealthCheckResult:
        """Check model inference"""
        start_time = time.time()
        
        try:
            # Mock model inference check
            if self.model is None:
                # Simulate model health
                model_loaded = True
                inference_time_ms = 25.0
                gpu_available = True
                memory_usage_mb = 2048
            else:
                # Would check actual model here
                model_loaded = True
                inference_time_ms = 25.0
                gpu_available = True
                memory_usage_mb = 2048
            
            details = {
                'model_loaded': model_loaded,
                'inference_time_ms': inference_time_ms,
                'gpu_available': gpu_available,
                'memory_usage_mb': memory_usage_mb
            }
            
            if not model_loaded:
                status = HealthStatus.UNHEALTHY
                message = "Model not loaded"
            elif inference_time_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"Model inference slow: {inference_time_ms:.1f}ms"
            elif not gpu_available:
                status = HealthStatus.DEGRADED
                message = "GPU not available, using CPU"
            else:
                status = HealthStatus.HEALTHY
                message = "Model inference operating normally"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Model inference check failed: {e}",
                details={'error': str(e)},
                duration_ms=duration_ms
            )
    
    def get_timeout(self) -> float:
        return 5.0


class DataPipelineCheck(HealthCheck):
    """Check data pipeline health"""
    
    def __init__(self):
        super().__init__("data_pipeline", ComponentType.DATA_PIPELINE)
    
    def check(self) -> HealthCheckResult:
        """Check data pipeline"""
        start_time = time.time()
        
        try:
            # Mock data pipeline check
            queue_size = 45
            processing_rate = 980.0
            error_rate = 0.02
            last_processed = time.time() - 1.5
            
            details = {
                'queue_size': queue_size,
                'processing_rate_items_per_sec': processing_rate,
                'error_rate': error_rate,
                'seconds_since_last_processed': time.time() - last_processed
            }
            
            if error_rate > 0.1:
                status = HealthStatus.UNHEALTHY
                message = f"High error rate: {error_rate:.2%}"
            elif queue_size > 100:
                status = HealthStatus.DEGRADED
                message = f"Queue backing up: {queue_size} items"
            elif time.time() - last_processed > 30:
                status = HealthStatus.DEGRADED
                message = "No recent processing activity"
            else:
                status = HealthStatus.HEALTHY
                message = "Data pipeline operating normally"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Data pipeline check failed: {e}",
                details={'error': str(e)},
                duration_ms=duration_ms
            )
    
    def get_timeout(self) -> float:
        return 3.0


class HealthChecker:
    """
    Comprehensive health monitoring system for BCI-GPT
    
    Orchestrates multiple health checks and provides
    system-wide health status and alerting.
    """
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.alert_handlers: List[Callable[[List[HealthCheckResult]], None]] = []
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def add_check(self, check: HealthCheck) -> None:
        """Add a health check"""
        with self._lock:
            self.checks[check.name] = check
            logger.info(f"Added health check: {check.name}")
    
    def remove_check(self, name: str) -> bool:
        """Remove a health check"""
        with self._lock:
            removed = self.checks.pop(name, None)
            if removed:
                self.results.pop(name, None)
                logger.info(f"Removed health check: {name}")
            return removed is not None
    
    def add_alert_handler(self, handler: Callable[[List[HealthCheckResult]], None]) -> None:
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def run_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks once"""
        results = {}
        
        with self._lock:
            checks_to_run = list(self.checks.values())
        
        for check in checks_to_run:
            try:
                logger.debug(f"Running health check: {check.name}")
                result = check.check()
                results[check.name] = result
                
                logger.debug(
                    f"Health check {check.name}: {result.status.value} "
                    f"({result.duration_ms:.1f}ms)"
                )
                
            except Exception as e:
                logger.error(f"Health check {check.name} failed: {e}")
                results[check.name] = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check execution failed: {e}",
                    details={'error': str(e)}
                )
        
        with self._lock:
            self.results.update(results)
        
        # Check for alerts
        unhealthy_results = [
            result for result in results.values()
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        ]
        
        if unhealthy_results:
            self._trigger_alerts(unhealthy_results)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        with self._lock:
            if not self.results:
                return {
                    'overall_status': HealthStatus.UNKNOWN.value,
                    'message': 'No health check results available',
                    'checks': [],
                    'summary': {
                        'total_checks': 0,
                        'healthy': 0,
                        'degraded': 0,
                        'unhealthy': 0,
                        'unknown': 0
                    }
                }
            
            results = list(self.results.values())
        
        # Count statuses
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{status_counts[HealthStatus.UNHEALTHY]} critical issues"
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{status_counts[HealthStatus.DEGRADED]} performance issues"
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.UNKNOWN
            message = f"{status_counts[HealthStatus.UNKNOWN]} checks failed"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        return {
            'overall_status': overall_status.value,
            'message': message,
            'timestamp': time.time(),
            'checks': [result.to_dict() for result in results],
            'summary': {
                'total_checks': len(results),
                'healthy': status_counts[HealthStatus.HEALTHY],
                'degraded': status_counts[HealthStatus.DEGRADED], 
                'unhealthy': status_counts[HealthStatus.UNHEALTHY],
                'unknown': status_counts[HealthStatus.UNKNOWN]
            }
        }
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous monitoring"""
        if self._running:
            logger.warning("Health monitoring already running")
            return
        
        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._check_thread.start()
        logger.info(f"Started health monitoring (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if not self._running:
            return
        
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self, interval: float) -> None:
        """Continuous monitoring loop"""
        while self._running:
            try:
                self.run_checks()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep in small increments to allow quick shutdown
            elapsed = 0
            while elapsed < interval and self._running:
                time.sleep(min(1.0, interval - elapsed))
                elapsed += 1.0
    
    def _trigger_alerts(self, results: List[HealthCheckResult]) -> None:
        """Trigger alerts for unhealthy results"""
        for handler in self.alert_handlers:
            try:
                handler(results)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


def create_default_health_checker() -> HealthChecker:
    """Create health checker with default checks"""
    checker = HealthChecker()
    
    # Add default checks
    checker.add_check(SystemResourceCheck())
    checker.add_check(EEGProcessorCheck())
    checker.add_check(ModelInferenceCheck())
    checker.add_check(DataPipelineCheck())
    
    return checker


def log_alert_handler(results: List[HealthCheckResult]) -> None:
    """Default alert handler that logs alerts"""
    for result in results:
        if result.status == HealthStatus.UNHEALTHY:
            logger.critical(f"CRITICAL: {result.name} - {result.message}")
        elif result.status == HealthStatus.DEGRADED:
            logger.warning(f"WARNING: {result.name} - {result.message}")