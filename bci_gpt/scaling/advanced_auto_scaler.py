"""
Advanced Auto-Scaling System for BCI-GPT

Provides intelligent auto-scaling with predictive algorithms,
multi-dimensional resource monitoring, and cost optimization.
"""

import time
import threading
import logging
import statistics
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling directions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingReason(Enum):
    """Reasons for scaling decisions"""
    CPU_PRESSURE = "cpu_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    QUEUE_BACKLOG = "queue_backlog"
    RESPONSE_TIME = "response_time"
    PREDICTIVE = "predictive"
    COST_OPTIMIZATION = "cost_optimization"
    MANUAL = "manual"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    queue_size: int
    response_time_ms: float
    active_requests: int
    errors_per_minute: float
    throughput_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'queue_size': self.queue_size,
            'response_time_ms': self.response_time_ms,
            'active_requests': self.active_requests,
            'errors_per_minute': self.errors_per_minute,
            'throughput_per_second': self.throughput_per_second
        }


@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: float
    direction: ScalingDirection
    reason: ScalingReason
    old_instances: int
    new_instances: int
    metrics: ResourceMetrics
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'direction': self.direction.value,
            'reason': self.reason.value,
            'old_instances': self.old_instances,
            'new_instances': self.new_instances,
            'metrics': self.metrics.to_dict(),
            'success': self.success,
            'error_message': self.error_message
        }


class ScalingPolicy(ABC):
    """Abstract base class for scaling policies"""
    
    @abstractmethod
    def should_scale(self, metrics: ResourceMetrics, current_instances: int) -> Tuple[ScalingDirection, ScalingReason]:
        """Determine if scaling is needed"""
        pass
    
    @abstractmethod
    def calculate_target_instances(self, metrics: ResourceMetrics, current_instances: int) -> int:
        """Calculate target number of instances"""
        pass


class ThresholdScalingPolicy(ScalingPolicy):
    """Traditional threshold-based scaling policy"""
    
    def __init__(
        self,
        cpu_scale_up_threshold: float = 70.0,
        cpu_scale_down_threshold: float = 30.0,
        memory_scale_up_threshold: float = 75.0,
        memory_scale_down_threshold: float = 35.0,
        response_time_threshold_ms: float = 500.0,
        queue_size_threshold: int = 50
    ):
        self.cpu_scale_up_threshold = cpu_scale_up_threshold
        self.cpu_scale_down_threshold = cpu_scale_down_threshold
        self.memory_scale_up_threshold = memory_scale_up_threshold
        self.memory_scale_down_threshold = memory_scale_down_threshold
        self.response_time_threshold_ms = response_time_threshold_ms
        self.queue_size_threshold = queue_size_threshold
    
    def should_scale(self, metrics: ResourceMetrics, current_instances: int) -> Tuple[ScalingDirection, ScalingReason]:
        """Check if scaling is needed based on thresholds"""
        
        # Scale up conditions
        if metrics.cpu_usage > self.cpu_scale_up_threshold:
            return ScalingDirection.SCALE_UP, ScalingReason.CPU_PRESSURE
        
        if metrics.memory_usage > self.memory_scale_up_threshold:
            return ScalingDirection.SCALE_UP, ScalingReason.MEMORY_PRESSURE
        
        if metrics.queue_size > self.queue_size_threshold:
            return ScalingDirection.SCALE_UP, ScalingReason.QUEUE_BACKLOG
        
        if metrics.response_time_ms > self.response_time_threshold_ms:
            return ScalingDirection.SCALE_UP, ScalingReason.RESPONSE_TIME
        
        # Scale down conditions (only if we have more than 1 instance)
        if current_instances > 1:
            if (metrics.cpu_usage < self.cpu_scale_down_threshold and 
                metrics.memory_usage < self.memory_scale_down_threshold and
                metrics.queue_size < self.queue_size_threshold // 2 and
                metrics.response_time_ms < self.response_time_threshold_ms // 2):
                return ScalingDirection.SCALE_DOWN, ScalingReason.COST_OPTIMIZATION
        
        return ScalingDirection.MAINTAIN, ScalingReason.COST_OPTIMIZATION
    
    def calculate_target_instances(self, metrics: ResourceMetrics, current_instances: int) -> int:
        """Calculate target instances based on resource utilization"""
        
        # Calculate required instances for each resource
        cpu_instances = max(1, math.ceil(current_instances * metrics.cpu_usage / 60.0))
        memory_instances = max(1, math.ceil(current_instances * metrics.memory_usage / 65.0))
        queue_instances = max(1, math.ceil(metrics.queue_size / 25.0))
        
        # Take the maximum requirement
        target = max(cpu_instances, memory_instances, queue_instances)
        
        # Limit scaling changes
        max_change = max(1, current_instances // 2)
        target = max(current_instances - max_change, min(target, current_instances + max_change))
        
        return max(1, target)


class PredictiveScalingPolicy(ScalingPolicy):
    """Predictive scaling policy using time series analysis"""
    
    def __init__(self, prediction_window_minutes: int = 10):
        self.prediction_window_minutes = prediction_window_minutes
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 100
    
    def add_metrics(self, metrics: ResourceMetrics) -> None:
        """Add metrics to history for prediction"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def should_scale(self, metrics: ResourceMetrics, current_instances: int) -> Tuple[ScalingDirection, ScalingReason]:
        """Predict future resource needs"""
        self.add_metrics(metrics)
        
        if len(self.metrics_history) < 5:
            # Not enough data for prediction, use current metrics
            if metrics.cpu_usage > 80 or metrics.memory_usage > 80:
                return ScalingDirection.SCALE_UP, ScalingReason.CPU_PRESSURE
            return ScalingDirection.MAINTAIN, ScalingReason.COST_OPTIMIZATION
        
        # Predict future CPU usage
        predicted_cpu = self._predict_metric([m.cpu_usage for m in self.metrics_history[-10:]])
        predicted_memory = self._predict_metric([m.memory_usage for m in self.metrics_history[-10:]])
        
        # Make scaling decision based on predictions
        if predicted_cpu > 75 or predicted_memory > 75:
            return ScalingDirection.SCALE_UP, ScalingReason.PREDICTIVE
        elif predicted_cpu < 25 and predicted_memory < 25 and current_instances > 1:
            return ScalingDirection.SCALE_DOWN, ScalingReason.PREDICTIVE
        
        return ScalingDirection.MAINTAIN, ScalingReason.COST_OPTIMIZATION
    
    def calculate_target_instances(self, metrics: ResourceMetrics, current_instances: int) -> int:
        """Calculate target instances based on predictions"""
        if len(self.metrics_history) < 5:
            return current_instances
        
        predicted_cpu = self._predict_metric([m.cpu_usage for m in self.metrics_history[-10:]])
        predicted_memory = self._predict_metric([m.memory_usage for m in self.metrics_history[-10:]])
        
        # Calculate instances needed for predicted load
        cpu_instances = max(1, math.ceil(current_instances * predicted_cpu / 70.0))
        memory_instances = max(1, math.ceil(current_instances * predicted_memory / 70.0))
        
        target = max(cpu_instances, memory_instances)
        
        # Smooth scaling changes
        if target > current_instances:
            target = min(target, current_instances + max(1, current_instances // 4))
        elif target < current_instances:
            target = max(target, current_instances - max(1, current_instances // 4))
        
        return max(1, target)
    
    def _predict_metric(self, values: List[float]) -> float:
        """Simple linear trend prediction"""
        if len(values) < 3:
            return values[-1] if values else 0.0
        
        # Calculate simple linear regression
        n = len(values)
        x_values = list(range(n))
        
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return values[-1]
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict next value
        next_x = n
        predicted = slope * next_x + intercept
        
        # Ensure reasonable bounds
        return max(0, min(100, predicted))


class AdvancedAutoScaler:
    """
    Advanced auto-scaler with multiple policies and intelligent decision making
    
    Combines threshold-based and predictive scaling with cost optimization
    and performance monitoring.
    """
    
    def __init__(
        self,
        name: str,
        min_instances: int = 1,
        max_instances: int = 10,
        cooldown_period: float = 300.0,  # 5 minutes
        scale_up_cooldown: float = 180.0,  # 3 minutes
        scale_down_cooldown: float = 600.0,  # 10 minutes
    ):
        self.name = name
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.cooldown_period = cooldown_period
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        self.current_instances = min_instances
        self.policies: List[ScalingPolicy] = []
        self.scaling_events: List[ScalingEvent] = []
        self.metrics_history: List[ResourceMetrics] = []
        
        self._last_scale_up_time = 0.0
        self._last_scale_down_time = 0.0
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Default policies
        self.add_policy(ThresholdScalingPolicy())
        self.add_policy(PredictiveScalingPolicy())
        
        self.scaling_handlers: List[Callable[[int, int, ScalingReason], None]] = []
        
        logger.info(f"Advanced auto-scaler '{name}' initialized")
    
    def add_policy(self, policy: ScalingPolicy) -> None:
        """Add scaling policy"""
        self.policies.append(policy)
        logger.info(f"Added scaling policy: {type(policy).__name__}")
    
    def add_scaling_handler(self, handler: Callable[[int, int, ScalingReason], None]) -> None:
        """Add handler for scaling events"""
        self.scaling_handlers.append(handler)
    
    def update_metrics(self, metrics: ResourceMetrics) -> None:
        """Update resource metrics"""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Keep limited history
            if len(self.metrics_history) > 200:
                self.metrics_history = self.metrics_history[-200:]
        
        # Feed metrics to predictive policies
        for policy in self.policies:
            if hasattr(policy, 'add_metrics'):
                policy.add_metrics(metrics)
        
        # Check if scaling is needed
        self._evaluate_scaling(metrics)
    
    def _evaluate_scaling(self, metrics: ResourceMetrics) -> None:
        """Evaluate if scaling action is needed"""
        
        # Check cooldown periods
        current_time = time.time()
        
        if (current_time - self._last_scale_up_time < self.scale_up_cooldown or
            current_time - self._last_scale_down_time < self.scale_down_cooldown):
            return
        
        # Collect scaling recommendations from all policies
        recommendations = []
        
        for policy in self.policies:
            try:
                direction, reason = policy.should_scale(metrics, self.current_instances)
                target_instances = policy.calculate_target_instances(metrics, self.current_instances)
                recommendations.append((direction, reason, target_instances, policy))
            except Exception as e:
                logger.error(f"Error in scaling policy {type(policy).__name__}: {e}")
        
        if not recommendations:
            return
        
        # Make scaling decision based on policy consensus
        final_direction, final_reason, final_target = self._decide_scaling_action(recommendations)
        
        if final_direction != ScalingDirection.MAINTAIN:
            self._execute_scaling(final_target, final_reason, metrics)
    
    def _decide_scaling_action(
        self, 
        recommendations: List[Tuple[ScalingDirection, ScalingReason, int, ScalingPolicy]]
    ) -> Tuple[ScalingDirection, ScalingReason, int]:
        """Decide final scaling action based on policy recommendations"""
        
        scale_up_votes = []
        scale_down_votes = []
        maintain_votes = []
        
        for direction, reason, target, policy in recommendations:
            if direction == ScalingDirection.SCALE_UP:
                scale_up_votes.append((reason, target, policy))
            elif direction == ScalingDirection.SCALE_DOWN:
                scale_down_votes.append((reason, target, policy))
            else:
                maintain_votes.append((reason, target, policy))
        
        # Prioritize scale-up for system health
        if scale_up_votes:
            # Choose the most aggressive scale-up recommendation
            max_target = max(target for _, target, _ in scale_up_votes)
            reason = next(reason for reason, target, _ in scale_up_votes if target == max_target)
            return ScalingDirection.SCALE_UP, reason, max_target
        
        # Consider scale-down only if all policies agree
        if scale_down_votes and not maintain_votes:
            # Choose conservative scale-down
            min_target = min(target for _, target, _ in scale_down_votes)
            reason = next(reason for reason, target, _ in scale_down_votes if target == min_target)
            return ScalingDirection.SCALE_DOWN, reason, min_target
        
        return ScalingDirection.MAINTAIN, ScalingReason.COST_OPTIMIZATION, self.current_instances
    
    def _execute_scaling(self, target_instances: int, reason: ScalingReason, metrics: ResourceMetrics) -> None:
        """Execute scaling action"""
        
        # Ensure target is within bounds
        target_instances = max(self.min_instances, min(self.max_instances, target_instances))
        
        if target_instances == self.current_instances:
            return
        
        old_instances = self.current_instances
        direction = ScalingDirection.SCALE_UP if target_instances > old_instances else ScalingDirection.SCALE_DOWN
        
        logger.info(
            f"Scaling {self.name}: {old_instances} -> {target_instances} instances "
            f"(reason: {reason.value})"
        )
        
        # Execute scaling through handlers
        success = True
        error_message = None
        
        try:
            for handler in self.scaling_handlers:
                handler(old_instances, target_instances, reason)
            
            self.current_instances = target_instances
            
            # Update cooldown timers
            current_time = time.time()
            if direction == ScalingDirection.SCALE_UP:
                self._last_scale_up_time = current_time
            else:
                self._last_scale_down_time = current_time
            
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Scaling execution failed: {e}")
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            direction=direction,
            reason=reason,
            old_instances=old_instances,
            new_instances=target_instances if success else old_instances,
            metrics=metrics,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self.scaling_events.append(event)
            # Keep limited event history
            if len(self.scaling_events) > 100:
                self.scaling_events = self.scaling_events[-100:]
    
    def force_scale(self, target_instances: int) -> bool:
        """Force scaling to specific number of instances"""
        target_instances = max(self.min_instances, min(self.max_instances, target_instances))
        
        if target_instances == self.current_instances:
            return True
        
        logger.warning(f"Force scaling {self.name}: {self.current_instances} -> {target_instances}")
        
        try:
            for handler in self.scaling_handlers:
                handler(self.current_instances, target_instances, ScalingReason.MANUAL)
            
            old_instances = self.current_instances
            self.current_instances = target_instances
            
            # Record manual scaling event
            if self.metrics_history:
                current_metrics = self.metrics_history[-1]
            else:
                current_metrics = ResourceMetrics(
                    timestamp=time.time(),
                    cpu_usage=0, memory_usage=0, queue_size=0,
                    response_time_ms=0, active_requests=0,
                    errors_per_minute=0, throughput_per_second=0
                )
            
            event = ScalingEvent(
                timestamp=time.time(),
                direction=ScalingDirection.SCALE_UP if target_instances > old_instances else ScalingDirection.SCALE_DOWN,
                reason=ScalingReason.MANUAL,
                old_instances=old_instances,
                new_instances=target_instances,
                metrics=current_metrics,
                success=True
            )
            
            with self._lock:
                self.scaling_events.append(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Force scaling failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status"""
        with self._lock:
            recent_events = self.scaling_events[-10:] if self.scaling_events else []
            recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'name': self.name,
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'policies_count': len(self.policies),
            'total_scaling_events': len(self.scaling_events),
            'recent_events': [event.to_dict() for event in recent_events],
            'recent_metrics': recent_metrics.to_dict() if recent_metrics else None,
            'last_scale_up': self._last_scale_up_time,
            'last_scale_down': self._last_scale_down_time,
            'monitoring_active': self._monitoring
        }
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous monitoring and scaling"""
        if self._monitoring:
            logger.warning("Auto-scaler monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started auto-scaler monitoring (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped auto-scaler monitoring")
    
    def _monitoring_loop(self, interval: float) -> None:
        """Monitoring loop"""
        while self._monitoring:
            try:
                # Generate mock metrics (in production, would collect real metrics)
                mock_metrics = ResourceMetrics(
                    timestamp=time.time(),
                    cpu_usage=45.0 + (time.time() % 20),  # Varying load
                    memory_usage=60.0 + (time.time() % 10),
                    queue_size=int(20 + (time.time() % 30)),
                    response_time_ms=150.0 + (time.time() % 50),
                    active_requests=int(50 + (time.time() % 25)),
                    errors_per_minute=0.1,
                    throughput_per_second=100.0
                )
                
                self.update_metrics(mock_metrics)
                
            except Exception as e:
                logger.error(f"Error in auto-scaler monitoring loop: {e}")
            
            # Sleep in small increments for responsive shutdown
            elapsed = 0
            while elapsed < interval and self._monitoring:
                time.sleep(min(1.0, interval - elapsed))
                elapsed += 1.0


def create_bci_auto_scaler() -> AdvancedAutoScaler:
    """Create auto-scaler optimized for BCI workloads"""
    
    scaler = AdvancedAutoScaler(
        name="bci_processing",
        min_instances=2,  # Always maintain redundancy
        max_instances=20,  # High scale for EEG processing
        scale_up_cooldown=120.0,  # Fast scale-up for BCI
        scale_down_cooldown=900.0  # Conservative scale-down
    )
    
    # Add BCI-specific policy
    bci_policy = ThresholdScalingPolicy(
        cpu_scale_up_threshold=60.0,  # Lower threshold for real-time processing
        memory_scale_up_threshold=70.0,
        response_time_threshold_ms=100.0,  # Strict latency requirements
        queue_size_threshold=25  # Small queue tolerance
    )
    
    scaler.add_policy(bci_policy)
    
    def bci_scaling_handler(old_instances: int, new_instances: int, reason: ScalingReason):
        """BCI-specific scaling handler"""
        logger.info(f"BCI scaling: {old_instances} -> {new_instances} ({reason.value})")
        # In production, this would trigger Kubernetes scaling, etc.
    
    scaler.add_scaling_handler(bci_scaling_handler)
    
    return scaler