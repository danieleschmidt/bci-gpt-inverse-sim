"""
Graceful Degradation Manager for BCI-GPT System

Provides intelligent service degradation when resources
are constrained or components fail.
"""

import logging
import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ServiceLevel(Enum):
    """Service quality levels"""
    FULL = "full"           # All features enabled
    HIGH = "high"           # Minor optimizations  
    MEDIUM = "medium"       # Moderate degradation
    LOW = "low"             # Minimal functionality
    CRITICAL = "critical"   # Emergency mode only


class DegradationTrigger(Enum):
    """Triggers for degradation"""
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    HIGH_LATENCY = "high_latency"
    ERROR_RATE = "error_rate"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMPONENT_FAILURE = "component_failure"


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    priority: int  # Lower number = higher priority
    min_service_level: ServiceLevel = ServiceLevel.LOW
    degradation_handlers: Dict[ServiceLevel, Callable[[], None]] = field(default_factory=dict)
    restoration_handlers: Dict[ServiceLevel, Callable[[], None]] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Current system metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    timestamp: float = field(default_factory=time.time)


class GracefulDegradationManager:
    """
    Manages graceful degradation of BCI-GPT services
    
    Monitors system health and automatically reduces service
    quality to maintain core functionality under stress.
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.current_service_level = ServiceLevel.FULL
        self.degradation_thresholds = {
            DegradationTrigger.HIGH_CPU: 85.0,
            DegradationTrigger.HIGH_MEMORY: 80.0,
            DegradationTrigger.HIGH_LATENCY: 500.0,  # ms
            DegradationTrigger.ERROR_RATE: 0.05,     # 5%
        }
        self.restoration_thresholds = {
            DegradationTrigger.HIGH_CPU: 70.0,
            DegradationTrigger.HIGH_MEMORY: 65.0,
            DegradationTrigger.HIGH_LATENCY: 200.0,
            DegradationTrigger.ERROR_RATE: 0.02,
        }
        
        self.active_triggers: Set[DegradationTrigger] = set()
        self.metrics_history: List[SystemMetrics] = []
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("Graceful degradation manager initialized")
    
    def register_service(self, config: ServiceConfig) -> None:
        """Register a service for degradation management"""
        with self._lock:
            self.services[config.name] = config
        logger.info(f"Registered service: {config.name} (priority: {config.priority})")
    
    def unregister_service(self, name: str) -> bool:
        """Unregister a service"""
        with self._lock:
            removed = self.services.pop(name, None)
        if removed:
            logger.info(f"Unregistered service: {name}")
        return removed is not None
    
    def update_metrics(self, metrics: SystemMetrics) -> None:
        """Update system metrics"""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Keep only last 100 metrics
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
        
        # Check for degradation triggers
        self._check_degradation_triggers(metrics)
    
    def _check_degradation_triggers(self, metrics: SystemMetrics) -> None:
        """Check if degradation should be triggered"""
        new_triggers = set()
        
        # Check CPU usage
        if metrics.cpu_usage > self.degradation_thresholds[DegradationTrigger.HIGH_CPU]:
            new_triggers.add(DegradationTrigger.HIGH_CPU)
        
        # Check memory usage
        if metrics.memory_usage > self.degradation_thresholds[DegradationTrigger.HIGH_MEMORY]:
            new_triggers.add(DegradationTrigger.HIGH_MEMORY)
        
        # Check latency
        if metrics.average_latency_ms > self.degradation_thresholds[DegradationTrigger.HIGH_LATENCY]:
            new_triggers.add(DegradationTrigger.HIGH_LATENCY)
        
        # Check error rate
        if metrics.error_rate > self.degradation_thresholds[DegradationTrigger.ERROR_RATE]:
            new_triggers.add(DegradationTrigger.ERROR_RATE)
        
        # Handle changes in triggers
        added_triggers = new_triggers - self.active_triggers
        removed_triggers = self.active_triggers - new_triggers
        
        if added_triggers:
            logger.warning(f"New degradation triggers: {[t.value for t in added_triggers]}")
            self.active_triggers.update(added_triggers)
            self._consider_degradation()
        
        if removed_triggers:
            logger.info(f"Degradation triggers resolved: {[t.value for t in removed_triggers]}")
            self.active_triggers -= removed_triggers
            self._consider_restoration()
    
    def _consider_degradation(self) -> None:
        """Consider degrading service level"""
        if not self.active_triggers:
            return
        
        # Determine required service level based on triggers
        required_level = self._calculate_required_service_level()
        
        if self._service_level_ordinal(required_level) < self._service_level_ordinal(self.current_service_level):
            self._degrade_to_level(required_level)
    
    def _consider_restoration(self) -> None:
        """Consider restoring service level"""
        # Check if we can restore based on current metrics
        if not self._should_restore():
            return
        
        # Find the highest service level we can support
        target_level = self._calculate_safe_service_level()
        
        if self._service_level_ordinal(target_level) > self._service_level_ordinal(self.current_service_level):
            self._restore_to_level(target_level)
    
    def _calculate_required_service_level(self) -> ServiceLevel:
        """Calculate required service level based on active triggers"""
        if not self.active_triggers:
            return ServiceLevel.FULL
        
        # More triggers or severe triggers require lower service levels
        trigger_severity = len(self.active_triggers)
        
        if DegradationTrigger.RESOURCE_EXHAUSTION in self.active_triggers:
            return ServiceLevel.CRITICAL
        elif DegradationTrigger.COMPONENT_FAILURE in self.active_triggers:
            return ServiceLevel.LOW
        elif trigger_severity >= 3:
            return ServiceLevel.LOW
        elif trigger_severity >= 2:
            return ServiceLevel.MEDIUM
        else:
            return ServiceLevel.HIGH
    
    def _calculate_safe_service_level(self) -> ServiceLevel:
        """Calculate safe service level for restoration"""
        if not self.metrics_history:
            return self.current_service_level
        
        # Use recent metrics to determine safe level
        recent_metrics = self.metrics_history[-5:]  # Last 5 metrics
        
        # Check if metrics are consistently good for restoration
        for metrics in recent_metrics:
            if (metrics.cpu_usage > self.restoration_thresholds[DegradationTrigger.HIGH_CPU] or
                metrics.memory_usage > self.restoration_thresholds[DegradationTrigger.HIGH_MEMORY] or
                metrics.average_latency_ms > self.restoration_thresholds[DegradationTrigger.HIGH_LATENCY] or
                metrics.error_rate > self.restoration_thresholds[DegradationTrigger.ERROR_RATE]):
                return self.current_service_level
        
        # If we get here, metrics are good for restoration
        return ServiceLevel.FULL
    
    def _should_restore(self) -> bool:
        """Check if we should attempt restoration"""
        if self.current_service_level == ServiceLevel.FULL:
            return False
        
        if not self.metrics_history:
            return False
        
        # Need stable metrics for restoration
        if len(self.metrics_history) < 3:
            return False
        
        return True
    
    def _degrade_to_level(self, target_level: ServiceLevel) -> None:
        """Degrade services to target level"""
        logger.warning(f"Degrading services from {self.current_service_level.value} to {target_level.value}")
        
        # Sort services by priority (lower number = higher priority)
        sorted_services = sorted(
            self.services.values(),
            key=lambda s: s.priority
        )
        
        for service in sorted_services:
            # Don't degrade below minimum service level
            if self._service_level_ordinal(target_level) < self._service_level_ordinal(service.min_service_level):
                effective_level = service.min_service_level
            else:
                effective_level = target_level
            
            # Apply degradation if handler exists
            if effective_level in service.degradation_handlers:
                try:
                    logger.info(f"Degrading service {service.name} to {effective_level.value}")
                    service.degradation_handlers[effective_level]()
                except Exception as e:
                    logger.error(f"Failed to degrade service {service.name}: {e}")
        
        self.current_service_level = target_level
    
    def _restore_to_level(self, target_level: ServiceLevel) -> None:
        """Restore services to target level"""
        logger.info(f"Restoring services from {self.current_service_level.value} to {target_level.value}")
        
        # Sort services by priority (higher priority first for restoration)
        sorted_services = sorted(
            self.services.values(),
            key=lambda s: s.priority
        )
        
        for service in sorted_services:
            # Apply restoration if handler exists
            if target_level in service.restoration_handlers:
                try:
                    logger.info(f"Restoring service {service.name} to {target_level.value}")
                    service.restoration_handlers[target_level]()
                except Exception as e:
                    logger.error(f"Failed to restore service {service.name}: {e}")
        
        self.current_service_level = target_level
    
    def _service_level_ordinal(self, level: ServiceLevel) -> int:
        """Get ordinal value for service level comparison"""
        order = {
            ServiceLevel.CRITICAL: 0,
            ServiceLevel.LOW: 1,
            ServiceLevel.MEDIUM: 2,
            ServiceLevel.HIGH: 3,
            ServiceLevel.FULL: 4
        }
        return order[level]
    
    def force_service_level(self, level: ServiceLevel) -> None:
        """Force service level (for testing or emergency)"""
        logger.warning(f"Forcing service level to {level.value}")
        
        if self._service_level_ordinal(level) < self._service_level_ordinal(self.current_service_level):
            self._degrade_to_level(level)
        elif self._service_level_ordinal(level) > self._service_level_ordinal(self.current_service_level):
            self._restore_to_level(level)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        with self._lock:
            recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'current_service_level': self.current_service_level.value,
            'active_triggers': [t.value for t in self.active_triggers],
            'registered_services': len(self.services),
            'recent_metrics': recent_metrics.timestamp if recent_metrics else None,
            'degradation_thresholds': {
                k.value: v for k, v in self.degradation_thresholds.items()
            },
            'restoration_thresholds': {
                k.value: v for k, v in self.restoration_thresholds.items()
            }
        }
    
    def start_monitoring(self, interval: float = 10.0) -> None:
        """Start continuous monitoring"""
        if self._monitoring:
            logger.warning("Degradation monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started degradation monitoring (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped degradation monitoring")
    
    def _monitoring_loop(self, interval: float) -> None:
        """Monitoring loop"""
        while self._monitoring:
            try:
                # Generate mock metrics (in production, this would collect real metrics)
                mock_metrics = SystemMetrics(
                    cpu_usage=45.0 + (time.time() % 10),  # Varying CPU
                    memory_usage=60.0,
                    average_latency_ms=150.0,
                    error_rate=0.01
                )
                self.update_metrics(mock_metrics)
                
            except Exception as e:
                logger.error(f"Error in degradation monitoring loop: {e}")
            
            # Sleep in small increments for responsive shutdown
            elapsed = 0
            while elapsed < interval and self._monitoring:
                time.sleep(min(1.0, interval - elapsed))
                elapsed += 1.0


# Helper function to create default BCI service configurations
def create_bci_service_configs() -> List[ServiceConfig]:
    """Create default BCI service configurations"""
    
    def degrade_eeg_processing_high():
        logger.info("EEG processing: Reduced filter complexity")
    
    def degrade_eeg_processing_medium():
        logger.info("EEG processing: Simplified preprocessing")
    
    def degrade_eeg_processing_low():
        logger.info("EEG processing: Basic filtering only")
    
    def degrade_model_inference_high():
        logger.info("Model inference: Reduced batch size")
    
    def degrade_model_inference_medium():
        logger.info("Model inference: Faster model variant")
    
    def degrade_model_inference_low():
        logger.info("Model inference: Simplified model")
    
    return [
        ServiceConfig(
            name="eeg_processing",
            priority=1,  # Highest priority
            min_service_level=ServiceLevel.LOW,
            degradation_handlers={
                ServiceLevel.HIGH: degrade_eeg_processing_high,
                ServiceLevel.MEDIUM: degrade_eeg_processing_medium,
                ServiceLevel.LOW: degrade_eeg_processing_low
            }
        ),
        ServiceConfig(
            name="model_inference", 
            priority=2,
            min_service_level=ServiceLevel.MEDIUM,
            degradation_handlers={
                ServiceLevel.HIGH: degrade_model_inference_high,
                ServiceLevel.MEDIUM: degrade_model_inference_medium,
                ServiceLevel.LOW: degrade_model_inference_low
            }
        )
    ]