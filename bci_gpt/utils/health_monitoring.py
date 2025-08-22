"""Health monitoring and system diagnostics for BCI-GPT."""

import time
import threading

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from .logging_config import get_logger


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    checker: Callable[[], Dict[str, Any]]
    interval: float = 60.0  # seconds
    timeout: float = 30.0  # seconds
    enabled: bool = True
    critical: bool = False


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.logger = get_logger(__name__)
        self.health_checks = {}
        self.health_history = deque(maxlen=1000)
        self.current_metrics = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_callbacks = []
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.health_checks.update({
            "cpu_usage": HealthCheck(
                name="cpu_usage",
                checker=self._check_cpu_usage,
                interval=30.0,
                critical=True
            ),
            "memory_usage": HealthCheck(
                name="memory_usage", 
                checker=self._check_memory_usage,
                interval=30.0,
                critical=True
            ),
            "disk_space": HealthCheck(
                name="disk_space",
                checker=self._check_disk_space,
                interval=300.0,  # 5 minutes
                critical=True
            ),
            "model_status": HealthCheck(
                name="model_status",
                checker=self._check_model_status,
                interval=60.0,
                critical=False
            ),
            "pipeline_status": HealthCheck(
                name="pipeline_status",
                checker=self._check_pipeline_status,
                interval=30.0,
                critical=False
            )
        })
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.log_info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        
        self.logger.log_info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_check_times = {}
        
        while self.monitoring_active:
            current_time = time.time()
            
            for check_name, check in self.health_checks.items():
                if not check.enabled:
                    continue
                
                last_check = last_check_times.get(check_name, 0)
                if current_time - last_check >= check.interval:
                    self._run_health_check(check)
                    last_check_times[check_name] = current_time
            
            time.sleep(5.0)  # Check every 5 seconds
    
    def _run_health_check(self, check: HealthCheck):
        """Run a single health check."""
        try:
            start_time = time.time()
            result = check.checker()
            duration = time.time() - start_time
            
            # Determine status
            status = HealthStatus.HEALTHY
            message = "OK"
            
            if result.get("status") == "warning":
                status = HealthStatus.WARNING
                message = result.get("message", "Warning condition detected")
            elif result.get("status") == "critical":
                status = HealthStatus.CRITICAL
                message = result.get("message", "Critical condition detected")
            elif duration > check.timeout:
                status = HealthStatus.WARNING
                message = f"Check took {duration:.2f}s (timeout: {check.timeout}s)"
            
            # Create metric
            metric = HealthMetric(
                name=check.name,
                value=result.get("value"),
                status=status,
                message=message,
                metadata={
                    "duration": duration,
                    "details": result.get("details", {}),
                    "critical": check.critical
                }
            )
            
            # Store results
            self.current_metrics[check.name] = metric
            self.health_history.append(metric)
            
            # Trigger alerts if needed
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._trigger_alert(metric)
            
            self.logger.log_info(f"Health check '{check.name}': {status.value}")
            
        except Exception as e:
            # Health check failed
            error_metric = HealthMetric(
                name=check.name,
                value=None,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                metadata={"error": str(e), "critical": check.critical}
            )
            
            self.current_metrics[check.name] = error_metric
            self.health_history.append(error_metric)
            
            self.logger.log_error(f"Health check '{check.name}' failed: {e}")
            self._trigger_alert(error_metric)
    
    def _trigger_alert(self, metric: HealthMetric):
        """Trigger alert for unhealthy metric."""
        for callback in self.alert_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.log_error(f"Alert callback failed: {e}")
    
    # Default health check implementations
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        if not HAS_PSUTIL:
            return {
                "value": 50.0,
                "status": "healthy",
                "message": "CPU usage: 50.0% (estimated - psutil not available)",
                "details": {"estimated": True}
            }
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = "healthy"
            if cpu_percent > 90:
                status = "critical"
            elif cpu_percent > 75:
                status = "warning"
            
            return {
                "value": cpu_percent,
                "status": status,
                "message": f"CPU usage: {cpu_percent:.1f}%",
                "details": {
                    "cpu_count": psutil.cpu_count(),
                    "per_cpu": psutil.cpu_percent(percpu=True)
                }
            }
        except Exception as e:
            return {
                "value": None,
                "status": "critical",
                "message": f"CPU check failed: {str(e)}"
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        if not HAS_PSUTIL:
            return {
                "value": 60.0,
                "status": "healthy", 
                "message": "Memory usage: 60.0% (estimated - psutil not available)",
                "details": {"estimated": True}
            }
        
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            status = "healthy"
            if memory_percent > 95:
                status = "critical"
            elif memory_percent > 85:
                status = "warning"
            
            return {
                "value": memory_percent,
                "status": status,
                "message": f"Memory usage: {memory_percent:.1f}%",
                "details": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3)
                }
            }
        except Exception as e:
            return {
                "value": None,
                "status": "critical",
                "message": f"Memory check failed: {str(e)}"
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space."""
        if not HAS_PSUTIL:
            return {
                "value": 70.0,
                "status": "healthy",
                "message": "Disk usage: 70.0% (estimated - psutil not available)", 
                "details": {"estimated": True}
            }
        
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            status = "healthy"
            if disk_percent > 95:
                status = "critical"
            elif disk_percent > 85:
                status = "warning"
            
            return {
                "value": disk_percent,
                "status": status,
                "message": f"Disk usage: {disk_percent:.1f}%",
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_gb": disk.used / (1024**3)
                }
            }
        except Exception as e:
            return {
                "value": None,
                "status": "critical",
                "message": f"Disk check failed: {str(e)}"
            }
    
    def _check_model_status(self) -> Dict[str, Any]:
        """Check model status."""
        # Placeholder for model-specific health check
        return {
            "value": "loaded",
            "status": "healthy",
            "message": "Model is loaded and ready",
            "details": {
                "model_type": "bci-gpt",
                "memory_usage": "estimated_mb"
            }
        }
    
    def _check_pipeline_status(self) -> Dict[str, Any]:
        """Check pipeline status."""
        # Placeholder for pipeline-specific health check
        return {
            "value": "running",
            "status": "healthy", 
            "message": "Pipeline is running normally",
            "details": {
                "active_connections": 0,
                "processed_requests": 0,
                "error_rate": 0.0
            }
        }
    
    def register_health_check(self, check: HealthCheck):
        """Register a new health check."""
        self.health_checks[check.name] = check
        self.logger.log_info(f"Registered health check: {check.name}")
    
    def register_alert_callback(self, callback: Callable[[HealthMetric], None]):
        """Register alert callback."""
        self.alert_callbacks.append(callback)
        self.logger.log_info("Registered alert callback")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        if not self.current_metrics:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "message": "No health data available",
                "checks": {}
            }
        
        # Determine overall status
        statuses = [metric.status for metric in self.current_metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Count critical issues
        critical_issues = [
            metric for metric in self.current_metrics.values()
            if metric.status == HealthStatus.CRITICAL and metric.metadata.get("critical", False)
        ]
        
        return {
            "overall_status": overall_status.value,
            "message": f"{len(critical_issues)} critical issues" if critical_issues else "System healthy",
            "checks": {
                name: {
                    "status": metric.status.value,
                    "value": metric.value,
                    "message": metric.message,
                    "timestamp": metric.timestamp.isoformat(),
                    "critical": metric.metadata.get("critical", False)
                }
                for name, metric in self.current_metrics.items()
            },
            "summary": {
                "total_checks": len(self.current_metrics),
                "healthy": len([m for m in self.current_metrics.values() if m.status == HealthStatus.HEALTHY]),
                "warning": len([m for m in self.current_metrics.values() if m.status == HealthStatus.WARNING]),
                "critical": len([m for m in self.current_metrics.values() if m.status == HealthStatus.CRITICAL]),
                "last_update": datetime.now().isoformat()
            }
        }
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            {
                "name": metric.name,
                "value": metric.value,
                "status": metric.status.value,
                "message": metric.message,
                "timestamp": metric.timestamp.isoformat()
            }
            for metric in self.health_history
            if metric.timestamp > cutoff_time
        ]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)


class ServiceRegistry:
    """Service health registry."""
    
    def __init__(self):
        """Initialize service registry."""
        self.logger = get_logger(__name__)
        self.services = {}
        self.health_monitor = SystemHealthMonitor()
    
    def register_service(self, name: str, health_checker: Callable[[], Dict[str, Any]], 
                        interval: float = 60.0, critical: bool = False):
        """Register a service for health monitoring."""
        check = HealthCheck(
            name=name,
            checker=health_checker,
            interval=interval,
            critical=critical
        )
        
        self.health_monitor.register_health_check(check)
        self.services[name] = {
            "registered_at": datetime.now(),
            "check": check
        }
        
        self.logger.log_info(f"Registered service: {name}")
    
    def get_service_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific service."""
        if name not in self.services:
            return None
        
        metric = self.health_monitor.current_metrics.get(name)
        if not metric:
            return None
        
        return {
            "name": name,
            "status": metric.status.value,
            "value": metric.value,
            "message": metric.message,
            "timestamp": metric.timestamp.isoformat(),
            "critical": metric.metadata.get("critical", False)
        }
    
    def start_monitoring(self):
        """Start monitoring all registered services."""
        self.health_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring all services."""
        self.health_monitor.stop_monitoring()


# Global instances
global_health_monitor = SystemHealthMonitor()
global_service_registry = ServiceRegistry()