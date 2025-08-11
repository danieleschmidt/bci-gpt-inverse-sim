"""Adaptive scaling system for dynamic load management."""

import time
import threading
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import queue

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    queue_size: int
    response_time_ms: float
    timestamp: float


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    confidence: float
    reasoning: str
    target_instances: int
    priority: int


class AdaptiveScaler:
    """Intelligent auto-scaling based on system metrics and load patterns."""
    
    def __init__(self,
                 min_instances: int = 1,
                 max_instances: int = 10,
                 target_cpu_percent: float = 70.0,
                 target_memory_percent: float = 80.0,
                 target_gpu_percent: float = 75.0,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 cooldown_seconds: int = 300):
        """Initialize adaptive scaler.
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_cpu_percent: Target CPU utilization
            target_memory_percent: Target memory utilization
            target_gpu_percent: Target GPU utilization
            scale_up_threshold: Threshold for scaling up
            scale_down_threshold: Threshold for scaling down
            cooldown_seconds: Cooldown period between scaling actions
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu = target_cpu_percent
        self.target_memory = target_memory_percent
        self.target_gpu = target_gpu_percent
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: List[Tuple[float, str, int]] = []
        
        # Current state
        self.current_instances = min_instances
        self.last_scaling_time = 0
        self.is_running = False
        
        # Threading
        self._lock = threading.RLock()
        self._metrics_thread: Optional[threading.Thread] = None
        self._scaling_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Advanced features
        self.predictive_scaling = True
        self.learning_rate = 0.1
        self.performance_weights = {
            'cpu': 0.3,
            'memory': 0.25,
            'gpu': 0.35,
            'queue': 0.1
        }
    
    def start(self) -> None:
        """Start the adaptive scaling system."""
        self.is_running = True
        
        # Start metrics collection thread
        self._metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._metrics_thread.start()
        
        # Start scaling decision thread
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        
        logger.info("Adaptive scaler started")
    
    def stop(self) -> None:
        """Stop the adaptive scaling system."""
        self.is_running = False
        
        if self._metrics_thread:
            self._metrics_thread.join(timeout=5)
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5)
            
        logger.info("Adaptive scaler stopped")
    
    def set_callbacks(self, 
                     scale_up: Callable[[int], bool],
                     scale_down: Callable[[int], bool]) -> None:
        """Set scaling callbacks."""
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
    
    def _collect_metrics(self) -> None:
        """Collect system metrics continuously."""
        while self.is_running:
            try:
                metrics = self._get_current_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(30)  # Back off on error
    
    def _get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1) if HAS_PSUTIL else 50.0
        memory_percent = psutil.virtual_memory().percent if HAS_PSUTIL else 60.0
        
        # GPU metrics
        gpu_util = 0.0
        gpu_memory = 0.0
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_memory,
            queue_size=0,  # To be implemented by specific applications
            response_time_ms=100.0,  # Placeholder
            timestamp=time.time()
        )
    
    def _scaling_loop(self) -> None:
        """Main scaling decision loop."""
        while self.is_running:
            try:
                decision = self._make_scaling_decision()
                
                if decision and decision.action != 'maintain':
                    self._execute_scaling_decision(decision)
                
                time.sleep(30)  # Make decisions every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)  # Back off on error
    
    def _make_scaling_decision(self) -> Optional[ScalingDecision]:
        """Make intelligent scaling decision based on metrics and patterns."""
        with self._lock:
            if len(self.metrics_history) < 3:
                return None
            
            # Check cooldown
            if time.time() - self.last_scaling_time < self.cooldown_seconds:
                return None
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 readings
            
            # Compute weighted performance score
            avg_performance = self._compute_performance_score(recent_metrics)
            
            # Detect trends
            trend = self._detect_performance_trend(recent_metrics)
            
            # Predictive scaling
            predicted_load = self._predict_future_load(recent_metrics) if self.predictive_scaling else avg_performance
            
            # Make decision
            if predicted_load > self.scale_up_threshold and self.current_instances < self.max_instances:
                confidence = min(0.9, (predicted_load - self.scale_up_threshold) * 2)
                target_instances = min(self.max_instances, self.current_instances + self._calculate_scale_amount(predicted_load, 'up'))
                
                return ScalingDecision(
                    action='scale_up',
                    confidence=confidence,
                    reasoning=f"High load detected: {predicted_load:.2f} (trend: {trend:.2f})",
                    target_instances=target_instances,
                    priority=self._calculate_priority(predicted_load, 'up')
                )
            
            elif predicted_load < self.scale_down_threshold and self.current_instances > self.min_instances:
                confidence = min(0.9, (self.scale_down_threshold - predicted_load) * 2)
                target_instances = max(self.min_instances, self.current_instances - self._calculate_scale_amount(predicted_load, 'down'))
                
                return ScalingDecision(
                    action='scale_down',
                    confidence=confidence,
                    reasoning=f"Low load detected: {predicted_load:.2f} (trend: {trend:.2f})",
                    target_instances=target_instances,
                    priority=self._calculate_priority(predicted_load, 'down')
                )
            
            return ScalingDecision(
                action='maintain',
                confidence=0.8,
                reasoning=f"Load within acceptable range: {predicted_load:.2f}",
                target_instances=self.current_instances,
                priority=0
            )
    
    def _compute_performance_score(self, metrics_list: List[SystemMetrics]) -> float:
        """Compute weighted performance score."""
        if not metrics_list:
            return 0.5
        
        avg_cpu = sum(m.cpu_percent for m in metrics_list) / len(metrics_list) / 100.0
        avg_memory = sum(m.memory_percent for m in metrics_list) / len(metrics_list) / 100.0
        avg_gpu = sum(m.gpu_utilization for m in metrics_list) / len(metrics_list) / 100.0
        avg_queue = sum(m.queue_size for m in metrics_list) / len(metrics_list)
        
        # Normalize queue size (assume max 100 for normalization)
        normalized_queue = min(1.0, avg_queue / 100.0)
        
        # Weighted score
        score = (
            self.performance_weights['cpu'] * avg_cpu +
            self.performance_weights['memory'] * avg_memory +
            self.performance_weights['gpu'] * avg_gpu +
            self.performance_weights['queue'] * normalized_queue
        )
        
        return score
    
    def _detect_performance_trend(self, metrics_list: List[SystemMetrics]) -> float:
        """Detect performance trend over time."""
        if len(metrics_list) < 5:
            return 0.0
        
        # Simple linear trend calculation
        scores = [self._compute_performance_score([m]) for m in metrics_list]
        n = len(scores)
        
        # Calculate slope using least squares
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        
        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _predict_future_load(self, metrics_list: List[SystemMetrics]) -> float:
        """Predict future load based on trends and patterns."""
        current_load = self._compute_performance_score(metrics_list)
        trend = self._detect_performance_trend(metrics_list)
        
        # Simple prediction: current + trend * prediction_horizon
        prediction_horizon = 5.0  # 5 time steps ahead
        predicted_load = current_load + (trend * prediction_horizon)
        
        # Apply learning from past predictions (simplified)
        if len(self.scaling_history) > 5:
            # Adjust prediction based on recent scaling effectiveness
            recent_scalings = self.scaling_history[-5:]
            adjustment = self._calculate_prediction_adjustment(recent_scalings)
            predicted_load += adjustment
        
        return max(0.0, min(1.0, predicted_load))
    
    def _calculate_prediction_adjustment(self, recent_scalings: List[Tuple]) -> float:
        """Calculate adjustment based on past scaling effectiveness."""
        # Simplified learning mechanism
        effective_scalings = 0
        total_scalings = len(recent_scalings)
        
        for timestamp, action, instances in recent_scalings:
            # Check if scaling was effective (placeholder logic)
            if action == 'scale_up' and instances > 1:
                effective_scalings += 1
            elif action == 'scale_down' and instances < self.max_instances:
                effective_scalings += 1
        
        effectiveness_ratio = effective_scalings / total_scalings if total_scalings > 0 else 0.5
        
        # Adjust prediction based on effectiveness
        return (effectiveness_ratio - 0.5) * self.learning_rate
    
    def _calculate_scale_amount(self, load: float, direction: str) -> int:
        """Calculate how many instances to scale by."""
        if direction == 'up':
            # Aggressive scaling for high load
            if load > 0.9:
                return min(3, self.max_instances - self.current_instances)
            elif load > 0.8:
                return min(2, self.max_instances - self.current_instances)
            else:
                return 1
        else:  # scale down
            # Conservative scaling down
            if load < 0.2:
                return min(2, self.current_instances - self.min_instances)
            else:
                return 1
    
    def _calculate_priority(self, load: float, direction: str) -> int:
        """Calculate priority of scaling action."""
        if direction == 'up':
            if load > 0.95:
                return 10  # Critical
            elif load > 0.85:
                return 7   # High
            else:
                return 5   # Medium
        else:  # scale down
            if load < 0.1:
                return 3   # Low priority scale down
            else:
                return 1   # Very low priority
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute the scaling decision."""
        if decision.confidence < 0.3:
            logger.info(f"Skipping scaling action due to low confidence: {decision.confidence}")
            return
        
        logger.info(f"Executing scaling decision: {decision.action} to {decision.target_instances} instances")
        logger.info(f"Reasoning: {decision.reasoning}")
        logger.info(f"Confidence: {decision.confidence:.2f}, Priority: {decision.priority}")
        
        success = False
        
        try:
            if decision.action == 'scale_up' and self.scale_up_callback:
                success = self.scale_up_callback(decision.target_instances)
            elif decision.action == 'scale_down' and self.scale_down_callback:
                success = self.scale_down_callback(decision.target_instances)
            
            if success:
                with self._lock:
                    self.current_instances = decision.target_instances
                    self.last_scaling_time = time.time()
                    self.scaling_history.append((time.time(), decision.action, decision.target_instances))
                
                logger.info(f"Successfully scaled to {decision.target_instances} instances")
            else:
                logger.warning(f"Scaling action failed: {decision.action}")
                
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
    
    def get_current_status(self) -> Dict:
        """Get current scaler status."""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-1] if self.metrics_history else None
            
            return {
                'current_instances': self.current_instances,
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'is_running': self.is_running,
                'metrics_count': len(self.metrics_history),
                'scaling_actions': len(self.scaling_history),
                'last_scaling_time': self.last_scaling_time,
                'current_metrics': {
                    'cpu_percent': recent_metrics.cpu_percent if recent_metrics else 0,
                    'memory_percent': recent_metrics.memory_percent if recent_metrics else 0,
                    'gpu_utilization': recent_metrics.gpu_utilization if recent_metrics else 0,
                } if recent_metrics else None,
                'performance_score': self._compute_performance_score(list(self.metrics_history)[-10:]) if len(self.metrics_history) >= 10 else 0
            }
    
    def adjust_thresholds(self, 
                         scale_up_threshold: Optional[float] = None,
                         scale_down_threshold: Optional[float] = None) -> None:
        """Dynamically adjust scaling thresholds."""
        if scale_up_threshold is not None:
            self.scale_up_threshold = scale_up_threshold
        if scale_down_threshold is not None:
            self.scale_down_threshold = scale_down_threshold
        
        logger.info(f"Adjusted thresholds - Up: {self.scale_up_threshold}, Down: {self.scale_down_threshold}")
    
    def simulate_load_test(self, duration_minutes: int = 10) -> Dict:
        """Simulate different load patterns for testing."""
        logger.info(f"Starting load simulation for {duration_minutes} minutes")
        
        start_time = time.time()
        load_patterns = []
        
        while time.time() - start_time < duration_minutes * 60:
            # Simulate various load patterns
            elapsed = time.time() - start_time
            
            if elapsed < 120:  # First 2 minutes - gradual increase
                simulated_load = 0.3 + (elapsed / 120) * 0.5
            elif elapsed < 300:  # Next 3 minutes - high load
                simulated_load = 0.8 + 0.1 * (elapsed % 60) / 60
            elif elapsed < 480:  # Next 3 minutes - fluctuating load
                simulated_load = 0.6 + 0.3 * abs(math.sin(elapsed / 30))
            else:  # Final 2 minutes - decreasing load
                simulated_load = 0.8 - ((elapsed - 480) / 120) * 0.6
            
            # Create simulated metrics
            simulated_metrics = SystemMetrics(
                cpu_percent=simulated_load * 100,
                memory_percent=(simulated_load * 0.8) * 100,
                gpu_utilization=simulated_load * 100,
                gpu_memory_percent=(simulated_load * 0.9) * 100,
                queue_size=int(simulated_load * 50),
                response_time_ms=50 + simulated_load * 200,
                timestamp=time.time()
            )
            
            with self._lock:
                self.metrics_history.append(simulated_metrics)
            
            load_patterns.append((elapsed, simulated_load, self.current_instances))
            
            time.sleep(10)  # Update every 10 seconds
        
        return {
            'duration_minutes': duration_minutes,
            'load_patterns': load_patterns,
            'final_instances': self.current_instances,
            'scaling_actions': len(self.scaling_history),
            'final_status': self.get_current_status()
        }


class LoadBalancer:
    """Intelligent load balancer with health checking."""
    
    def __init__(self):
        self.instances: Dict[str, Dict] = {}
        self.request_queue = queue.Queue()
        self.health_check_interval = 30
        self._lock = threading.RLock()
        self.is_running = False
        
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0) -> None:
        """Add instance to load balancer."""
        with self._lock:
            self.instances[instance_id] = {
                'endpoint': endpoint,
                'weight': weight,
                'healthy': True,
                'request_count': 0,
                'response_time': 0.0,
                'last_health_check': 0
            }
    
    def remove_instance(self, instance_id: str) -> None:
        """Remove instance from load balancer."""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
    
    def get_next_instance(self) -> Optional[str]:
        """Get next instance using weighted round-robin."""
        with self._lock:
            healthy_instances = {k: v for k, v in self.instances.items() if v['healthy']}
            
            if not healthy_instances:
                return None
            
            # Weighted selection based on inverse response time and weight
            weights = []
            instance_ids = []
            
            for instance_id, info in healthy_instances.items():
                # Higher weight for faster instances
                effective_weight = info['weight'] / max(0.1, info['response_time'])
                weights.append(effective_weight)
                instance_ids.append(instance_id)
            
            # Simple weighted selection (could use more sophisticated algorithms)
            total_weight = sum(weights)
            if total_weight == 0:
                return list(healthy_instances.keys())[0]
            
            import random
            r = random.uniform(0, total_weight)
            cumulative_weight = 0
            
            for i, weight in enumerate(weights):
                cumulative_weight += weight
                if r <= cumulative_weight:
                    return instance_ids[i]
            
            return instance_ids[-1]  # Fallback
    
    def update_instance_stats(self, instance_id: str, response_time: float, success: bool) -> None:
        """Update instance performance statistics."""
        with self._lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                instance['request_count'] += 1
                
                # Exponential moving average of response time
                alpha = 0.1
                instance['response_time'] = (
                    alpha * response_time + 
                    (1 - alpha) * instance['response_time']
                )
                
                # Update health based on success rate
                if not success:
                    instance['healthy'] = False


# Singleton instances
adaptive_scaler = AdaptiveScaler()
load_balancer = LoadBalancer()


def auto_scale(min_instances: int = 1, max_instances: int = 10):
    """Decorator for auto-scaling function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record start time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                
                # Update load balancer stats if applicable
                # This would be implemented based on specific use case
                
                return result
                
            except Exception as e:
                # Handle errors and potentially trigger scaling
                logger.error(f"Function execution failed: {e}")
                raise
                
        return wrapper
    return decorator