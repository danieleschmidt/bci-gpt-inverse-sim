"""Real-time Processing Guard for BCI-GPT Self-Healing System.

Ensures real-time processing constraints are met, manages latency,
optimizes performance, and maintains quality of service.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np
import psutil

from ..utils.monitoring import HealthStatus
from ..utils.error_handling import BCI_GPTError
from ..utils.performance_optimizer import PerformanceOptimizer


class ProcessingPriority(Enum):
    """Processing priority levels."""
    CRITICAL = "critical"
    HIGH = "high" 
    NORMAL = "normal"
    LOW = "low"


class QualityLevel(Enum):
    """Processing quality levels."""
    MAXIMUM = "maximum"
    HIGH = "high"
    NORMAL = "normal"
    REDUCED = "reduced"
    MINIMAL = "minimal"


@dataclass 
class PerformanceMetrics:
    """Real-time performance metrics."""
    latency_ms: float = 0.0
    throughput_hz: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0
    queue_depth: int = 0
    processing_errors: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingConstraints:
    """Real-time processing constraints."""
    max_latency_ms: float = 100.0
    min_throughput_hz: float = 100.0
    max_cpu_usage: float = 80.0
    max_memory_mb: float = 2048.0
    max_queue_depth: int = 50
    max_error_rate: float = 0.05
    quality_degradation_threshold: float = 0.7


@dataclass
class RealtimeGuardConfig:
    """Configuration for real-time processing guard."""
    monitoring_interval: float = 1.0
    latency_window_size: int = 100
    throughput_window_size: int = 50
    auto_scaling_enabled: bool = True
    quality_adaptation_enabled: bool = True
    load_shedding_enabled: bool = True
    predictive_scaling_enabled: bool = True
    resource_monitoring_enabled: bool = True


class RealtimeProcessingGuard:
    """Guard for real-time processing performance and constraints.
    
    Monitors latency, throughput, resource usage, and automatically
    optimizes processing to maintain real-time guarantees.
    """
    
    def __init__(self, config: Optional[RealtimeGuardConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or RealtimeGuardConfig()
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=self.config.latency_window_size)
        self.latency_history: deque = deque(maxlen=self.config.latency_window_size) 
        self.throughput_history: deque = deque(maxlen=self.config.throughput_window_size)
        
        # Processing constraints and state
        self.constraints = ProcessingConstraints()
        self.current_quality_level = QualityLevel.NORMAL
        self.current_priority = ProcessingPriority.NORMAL
        
        # Resource monitoring
        self.cpu_monitor = psutil
        self.performance_optimizer = PerformanceOptimizer()
        
        # Processing queues and state
        self.processing_queue: deque = deque()
        self.priority_queues: Dict[ProcessingPriority, deque] = {
            priority: deque() for priority in ProcessingPriority
        }
        
        # Monitoring and control
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.optimization_thread: Optional[threading.Thread] = None
        
        # Performance callbacks
        self.performance_callbacks: List[Callable] = []
        self.constraint_violation_callbacks: List[Callable] = []
        
        # Scaling and optimization state  
        self.scale_up_requests = 0
        self.scale_down_requests = 0
        self.last_scaling_action: Optional[datetime] = None
        self.optimization_actions_taken = 0
        
        # Load shedding state
        self.load_shedding_active = False
        self.load_shedding_start_time: Optional[datetime] = None
        self.dropped_requests = 0
        
        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.constraint_violations = 0
        self.guard_start_time = datetime.now()
        
    def set_processing_constraints(self, constraints: ProcessingConstraints) -> None:
        """Set processing constraints for real-time operation."""
        self.constraints = constraints
        self.logger.info(f"Processing constraints updated: max_latency={constraints.max_latency_ms}ms")
    
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        
        self.monitoring_thread.start()
        self.optimization_thread.start()
        
        self.logger.info("Real-time processing guard started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        
        self.logger.info("Real-time processing guard stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for real-time performance."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_performance_metrics()
                self.metrics_history.append(metrics)
                
                # Check constraint violations
                violations = self._check_constraint_violations(metrics)
                if violations:
                    self._handle_constraint_violations(violations, metrics)
                
                # Update performance trends
                self._update_performance_trends(metrics)
                
                # Trigger performance callbacks
                self._trigger_performance_callbacks(metrics)
                
                # Adaptive quality control
                if self.config.quality_adaptation_enabled:
                    self._adapt_processing_quality(metrics)
                
                # Load shedding if necessary
                if self.config.load_shedding_enabled:
                    self._manage_load_shedding(metrics)
                
            except Exception as e:
                self.logger.error(f"Real-time monitoring error: {e}")
            
            time.sleep(self.config.monitoring_interval)
    
    def _optimization_loop(self) -> None:
        """Optimization and scaling loop."""
        while self.monitoring_active:
            try:
                if len(self.metrics_history) >= 10:
                    # Auto-scaling decisions
                    if self.config.auto_scaling_enabled:
                        asyncio.run(self._auto_scale_resources())
                    
                    # Predictive optimization
                    if self.config.predictive_scaling_enabled:
                        self._predictive_optimization()
                    
                    # Resource optimization
                    self._optimize_resource_usage()
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
            
            time.sleep(5.0)  # Less frequent optimization checks
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # CPU and memory usage
        cpu_usage = self.cpu_monitor.cpu_percent()
        memory = self.cpu_monitor.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        
        # GPU usage (simplified - would use nvidia-ml-py in real implementation)
        gpu_usage = 0.0  # Placeholder
        
        # Processing queue depth
        queue_depth = len(self.processing_queue)
        
        # Calculate current throughput
        current_time = datetime.now()
        if len(self.throughput_history) > 0:
            time_window = (current_time - self.throughput_history[0].timestamp).total_seconds()
            if time_window > 0:
                throughput = len(self.throughput_history) / time_window
            else:
                throughput = 0.0
        else:
            throughput = 0.0
        
        # Calculate average latency from recent measurements
        if len(self.latency_history) > 0:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
        else:
            avg_latency = 0.0
        
        return PerformanceMetrics(
            latency_ms=avg_latency,
            throughput_hz=throughput,
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            gpu_usage=gpu_usage,
            queue_depth=queue_depth,
            processing_errors=self.total_errors
        )
    
    def _check_constraint_violations(self, metrics: PerformanceMetrics) -> List[str]:
        """Check for real-time constraint violations."""
        violations = []
        
        # Latency constraint
        if metrics.latency_ms > self.constraints.max_latency_ms:
            violations.append(f"latency_exceeded:{metrics.latency_ms:.1f}ms")
        
        # Throughput constraint
        if metrics.throughput_hz < self.constraints.min_throughput_hz:
            violations.append(f"throughput_low:{metrics.throughput_hz:.1f}Hz")
        
        # CPU usage constraint
        if metrics.cpu_usage > self.constraints.max_cpu_usage:
            violations.append(f"cpu_high:{metrics.cpu_usage:.1f}%")
        
        # Memory constraint
        if metrics.memory_usage_mb > self.constraints.max_memory_mb:
            violations.append(f"memory_high:{metrics.memory_usage_mb:.1f}MB")
        
        # Queue depth constraint
        if metrics.queue_depth > self.constraints.max_queue_depth:
            violations.append(f"queue_overflow:{metrics.queue_depth}")
        
        # Error rate constraint
        error_rate = self.total_errors / max(self.total_processed, 1)
        if error_rate > self.constraints.max_error_rate:
            violations.append(f"error_rate_high:{error_rate:.3f}")
        
        return violations
    
    def _handle_constraint_violations(self, violations: List[str], metrics: PerformanceMetrics) -> None:
        """Handle constraint violations with immediate actions."""
        self.constraint_violations += len(violations)
        
        self.logger.warning(f"Constraint violations detected: {violations}")
        
        # Immediate mitigation actions
        for violation in violations:
            violation_type = violation.split(':')[0]
            
            if violation_type == "latency_exceeded":
                self._mitigate_latency_violation(metrics)
            elif violation_type == "throughput_low":
                self._mitigate_throughput_violation(metrics)
            elif violation_type == "cpu_high":
                self._mitigate_cpu_violation(metrics)
            elif violation_type == "memory_high":
                self._mitigate_memory_violation(metrics)
            elif violation_type == "queue_overflow":
                self._mitigate_queue_violation(metrics)
        
        # Notify callbacks
        for callback in self.constraint_violation_callbacks:
            try:
                callback(violations, metrics)
            except Exception as e:
                self.logger.error(f"Constraint violation callback error: {e}")
    
    def _mitigate_latency_violation(self, metrics: PerformanceMetrics) -> None:
        """Mitigate latency constraint violation."""
        self.logger.info("Mitigating latency violation")
        
        # Reduce processing quality if possible
        if self.current_quality_level != QualityLevel.MINIMAL:
            self._reduce_quality_level()
        
        # Enable load shedding
        if not self.load_shedding_active:
            self._enable_load_shedding("latency_violation")
        
        # Request resource scaling
        self._request_scale_up("latency")
    
    def _mitigate_throughput_violation(self, metrics: PerformanceMetrics) -> None:
        """Mitigate throughput constraint violation."""
        self.logger.info("Mitigating throughput violation")
        
        # Optimize processing pipeline
        asyncio.run(self._optimize_processing_pipeline())
        
        # Request resource scaling
        self._request_scale_up("throughput")
    
    def _mitigate_cpu_violation(self, metrics: PerformanceMetrics) -> None:
        """Mitigate CPU usage violation."""
        self.logger.info("Mitigating CPU usage violation")
        
        # Reduce processing complexity
        self._reduce_quality_level()
        
        # Enable load shedding
        if not self.load_shedding_active:
            self._enable_load_shedding("cpu_overload")
    
    def _mitigate_memory_violation(self, metrics: PerformanceMetrics) -> None:
        """Mitigate memory usage violation."""
        self.logger.info("Mitigating memory usage violation")
        
        # Clear unnecessary buffers
        self._cleanup_memory_buffers()
        
        # Request memory scaling
        self._request_scale_up("memory")
    
    def _mitigate_queue_violation(self, metrics: PerformanceMetrics) -> None:
        """Mitigate processing queue overflow."""
        self.logger.info("Mitigating queue overflow")
        
        # Enable aggressive load shedding
        if not self.load_shedding_active:
            self._enable_load_shedding("queue_overflow")
        
        # Process highest priority items first
        self._prioritize_critical_processing()
    
    def _reduce_quality_level(self) -> None:
        """Reduce processing quality to improve performance."""
        quality_levels = list(QualityLevel)
        current_index = quality_levels.index(self.current_quality_level)
        
        if current_index < len(quality_levels) - 1:
            self.current_quality_level = quality_levels[current_index + 1]
            self.logger.info(f"Reduced quality level to: {self.current_quality_level.value}")
    
    def _enable_load_shedding(self, reason: str) -> None:
        """Enable load shedding to reduce processing load."""
        self.load_shedding_active = True
        self.load_shedding_start_time = datetime.now()
        self.logger.info(f"Load shedding enabled due to: {reason}")
    
    def _disable_load_shedding(self) -> None:
        """Disable load shedding when conditions improve."""
        if self.load_shedding_active:
            self.load_shedding_active = False
            duration = datetime.now() - self.load_shedding_start_time if self.load_shedding_start_time else timedelta()
            self.logger.info(f"Load shedding disabled after {duration.total_seconds():.1f}s")
    
    def _request_scale_up(self, reason: str) -> None:
        """Request resource scaling up."""
        self.scale_up_requests += 1
        self.last_scaling_action = datetime.now()
        self.logger.info(f"Scaling up requested due to: {reason}")
    
    def _update_performance_trends(self, metrics: PerformanceMetrics) -> None:
        """Update performance trends for predictive optimization."""
        # Store latency and throughput history
        self.latency_history.append(metrics.latency_ms)
        
        # Create throughput tracking entry
        self.throughput_history.append(metrics)
        
        # Clean up old throughput entries
        current_time = datetime.now()
        while (self.throughput_history and 
               (current_time - self.throughput_history[0].timestamp).total_seconds() > 60):
            self.throughput_history.popleft()
    
    def _adapt_processing_quality(self, metrics: PerformanceMetrics) -> None:
        """Adapt processing quality based on performance."""
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        # Adjust quality based on performance
        if performance_score < self.constraints.quality_degradation_threshold:
            if self.current_quality_level != QualityLevel.MINIMAL:
                self._reduce_quality_level()
        else:
            # Increase quality if performance allows
            quality_levels = list(QualityLevel)
            current_index = quality_levels.index(self.current_quality_level)
            
            if current_index > 0 and performance_score > 0.9:
                self.current_quality_level = quality_levels[current_index - 1]
                self.logger.info(f"Increased quality level to: {self.current_quality_level.value}")
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-1)."""
        # Latency score
        latency_score = max(0.0, 1.0 - metrics.latency_ms / self.constraints.max_latency_ms)
        
        # Throughput score  
        throughput_score = min(1.0, metrics.throughput_hz / self.constraints.min_throughput_hz)
        
        # Resource usage score
        cpu_score = max(0.0, 1.0 - metrics.cpu_usage / self.constraints.max_cpu_usage)
        memory_score = max(0.0, 1.0 - metrics.memory_usage_mb / self.constraints.max_memory_mb)
        
        # Queue score
        queue_score = max(0.0, 1.0 - metrics.queue_depth / self.constraints.max_queue_depth)
        
        # Weighted average
        performance_score = (
            latency_score * 0.3 +
            throughput_score * 0.3 + 
            cpu_score * 0.2 +
            memory_score * 0.1 +
            queue_score * 0.1
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _manage_load_shedding(self, metrics: PerformanceMetrics) -> None:
        """Manage load shedding based on current conditions."""
        performance_score = self._calculate_performance_score(metrics)
        
        # Enable load shedding if performance is poor
        if performance_score < 0.3 and not self.load_shedding_active:
            self._enable_load_shedding("poor_performance")
        
        # Disable load shedding if performance improves
        elif performance_score > 0.8 and self.load_shedding_active:
            self._disable_load_shedding()
    
    async def _auto_scale_resources(self) -> None:
        """Automatically scale resources based on demand."""
        if not self.metrics_history:
            return
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate scaling needs
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_queue = np.mean([m.queue_depth for m in recent_metrics])
        
        # Scale up conditions
        scale_up_needed = (
            avg_latency > self.constraints.max_latency_ms * 0.8 or
            avg_cpu > self.constraints.max_cpu_usage * 0.8 or
            avg_queue > self.constraints.max_queue_depth * 0.8
        )
        
        # Scale down conditions
        scale_down_needed = (
            avg_latency < self.constraints.max_latency_ms * 0.3 and
            avg_cpu < self.constraints.max_cpu_usage * 0.3 and
            avg_queue < self.constraints.max_queue_depth * 0.3
        )
        
        # Prevent too frequent scaling
        if self.last_scaling_action:
            time_since_scaling = (datetime.now() - self.last_scaling_action).total_seconds()
            if time_since_scaling < 30:  # 30 second cooldown
                return
        
        if scale_up_needed:
            await self._scale_up_resources()
        elif scale_down_needed:
            await self._scale_down_resources()
    
    async def _scale_up_resources(self) -> None:
        """Scale up processing resources."""
        self.logger.info("Scaling up processing resources")
        self.scale_up_requests += 1
        self.last_scaling_action = datetime.now()
        
        # Note: In real implementation, this would scale actual resources
        await asyncio.sleep(0.1)  # Simulated scaling time
    
    async def _scale_down_resources(self) -> None:
        """Scale down processing resources."""
        self.logger.info("Scaling down processing resources")
        self.scale_down_requests += 1
        self.last_scaling_action = datetime.now()
        
        # Note: In real implementation, this would scale down resources
        await asyncio.sleep(0.1)  # Simulated scaling time
    
    def _predictive_optimization(self) -> None:
        """Perform predictive optimization based on trends."""
        if len(self.latency_history) < 20:
            return
        
        # Simple trend prediction
        recent_latencies = list(self.latency_history)[-20:]
        latency_trend = np.polyfit(range(len(recent_latencies)), recent_latencies, 1)[0]
        
        # If latency is trending up, preemptively optimize
        if latency_trend > 1.0:  # Increasing by more than 1ms per measurement
            self.logger.info("Preemptive optimization due to latency trend")
            asyncio.run(self._preemptive_optimization())
    
    async def _preemptive_optimization(self) -> None:
        """Perform preemptive optimization."""
        # Optimize processing pipeline
        await self._optimize_processing_pipeline()
        
        # Request scaling if needed
        self._request_scale_up("predictive")
    
    def _optimize_resource_usage(self) -> None:
        """Optimize current resource usage."""
        try:
            # Use performance optimizer for resource optimization
            optimization_results = self.performance_optimizer.optimize_system_resources()
            self.optimization_actions_taken += 1
            
            self.logger.debug(f"Resource optimization completed: {optimization_results}")
        except Exception as e:
            self.logger.error(f"Resource optimization failed: {e}")
    
    async def _optimize_processing_pipeline(self) -> None:
        """Optimize the processing pipeline for performance."""
        try:
            self.logger.info("Optimizing processing pipeline")
            
            # Note: In real implementation, this would optimize the actual pipeline
            await asyncio.sleep(0.5)  # Simulated optimization time
            
            self.logger.info("Processing pipeline optimization completed")
        except Exception as e:
            self.logger.error(f"Pipeline optimization failed: {e}")
    
    def _cleanup_memory_buffers(self) -> None:
        """Clean up memory buffers to free resources."""
        # Clear old metrics history
        if len(self.metrics_history) > 50:
            # Keep only recent 50 entries
            while len(self.metrics_history) > 50:
                self.metrics_history.popleft()
        
        self.logger.info("Memory buffers cleaned up")
    
    def _prioritize_critical_processing(self) -> None:
        """Prioritize critical processing items."""
        # Move critical items to front of queue
        critical_items = []
        normal_items = []
        
        while self.processing_queue:
            item = self.processing_queue.popleft()
            if getattr(item, 'priority', ProcessingPriority.NORMAL) == ProcessingPriority.CRITICAL:
                critical_items.append(item)
            else:
                normal_items.append(item)
        
        # Re-add with critical items first
        self.processing_queue.extend(critical_items + normal_items)
    
    def record_processing_latency(self, latency_ms: float) -> None:
        """Record processing latency measurement."""
        self.latency_history.append(latency_ms)
        self.total_processed += 1
    
    def record_processing_error(self) -> None:
        """Record a processing error."""
        self.total_errors += 1
    
    def should_drop_request(self) -> bool:
        """Check if request should be dropped due to load shedding."""
        if not self.load_shedding_active:
            return False
        
        # Simple load shedding - drop every other request
        self.dropped_requests += 1
        return self.dropped_requests % 2 == 0
    
    async def optimize_processing(self) -> bool:
        """Optimize processing performance."""
        try:
            await self._optimize_processing_pipeline()
            return True
        except Exception as e:
            self.logger.error(f"Processing optimization failed: {e}")
            return False
    
    async def scale_resources(self) -> bool:
        """Scale processing resources."""
        try:
            await self._scale_up_resources()
            return True
        except Exception as e:
            self.logger.error(f"Resource scaling failed: {e}")
            return False
    
    async def reduce_quality_temporarily(self) -> bool:
        """Temporarily reduce processing quality."""
        try:
            original_level = self.current_quality_level
            self._reduce_quality_level()
            
            # Schedule quality restoration
            asyncio.create_task(self._restore_quality_after_delay(original_level, 30.0))
            
            return True
        except Exception as e:
            self.logger.error(f"Quality reduction failed: {e}")
            return False
    
    async def _restore_quality_after_delay(self, target_level: QualityLevel, delay_seconds: float) -> None:
        """Restore quality level after a delay."""
        await asyncio.sleep(delay_seconds)
        
        # Only restore if conditions allow
        if len(self.metrics_history) > 0:
            latest_metrics = self.metrics_history[-1]
            performance_score = self._calculate_performance_score(latest_metrics)
            
            if performance_score > 0.8:
                self.current_quality_level = target_level
                self.logger.info(f"Quality level restored to: {target_level.value}")
    
    def register_performance_callback(self, callback: Callable) -> None:
        """Register callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    def register_constraint_violation_callback(self, callback: Callable) -> None:
        """Register callback for constraint violations."""
        self.constraint_violation_callbacks.append(callback)
    
    def _trigger_performance_callbacks(self, metrics: PerformanceMetrics) -> None:
        """Trigger performance callbacks."""
        for callback in self.performance_callbacks:
            try:
                callback(self._metrics_to_dict(metrics))
            except Exception as e:
                self.logger.error(f"Performance callback error: {e}")
    
    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "latency_ms": metrics.latency_ms,
            "throughput_hz": metrics.throughput_hz,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage_mb": metrics.memory_usage_mb,
            "gpu_usage": metrics.gpu_usage,
            "queue_depth": metrics.queue_depth,
            "processing_errors": metrics.processing_errors,
            "performance_score": self._calculate_performance_score(metrics),
            "timestamp": metrics.timestamp.isoformat()
        }
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get comprehensive performance status."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        uptime = (datetime.now() - self.guard_start_time).total_seconds()
        
        return {
            "performance_score": self._calculate_performance_score(latest_metrics) if latest_metrics else 0.0,
            "current_quality_level": self.current_quality_level.value,
            "load_shedding_active": self.load_shedding_active,
            "constraints": {
                "max_latency_ms": self.constraints.max_latency_ms,
                "min_throughput_hz": self.constraints.min_throughput_hz,
                "max_cpu_usage": self.constraints.max_cpu_usage,
                "max_memory_mb": self.constraints.max_memory_mb
            },
            "current_metrics": self._metrics_to_dict(latest_metrics) if latest_metrics else {},
            "trends": {
                "avg_latency_ms": np.mean(list(self.latency_history)) if self.latency_history else 0.0,
                "latency_trend": np.polyfit(range(len(self.latency_history)), list(self.latency_history), 1)[0] if len(self.latency_history) > 1 else 0.0,
            },
            "statistics": {
                "total_processed": self.total_processed,
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / max(self.total_processed, 1),
                "constraint_violations": self.constraint_violations,
                "scale_up_requests": self.scale_up_requests,
                "scale_down_requests": self.scale_down_requests,
                "dropped_requests": self.dropped_requests,
                "optimization_actions": self.optimization_actions_taken,
                "uptime_seconds": uptime
            },
            "config": {
                "auto_scaling_enabled": self.config.auto_scaling_enabled,
                "quality_adaptation_enabled": self.config.quality_adaptation_enabled,
                "load_shedding_enabled": self.config.load_shedding_enabled,
                "predictive_scaling_enabled": self.config.predictive_scaling_enabled
            }
        }