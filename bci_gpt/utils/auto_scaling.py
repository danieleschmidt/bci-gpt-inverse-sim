"""Auto-scaling and load balancing system for BCI-GPT."""

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import queue
import warnings

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not available for system monitoring")

from .logging_config import get_logger, performance_monitor
from .monitoring import get_metrics_collector
from .performance_optimizer import get_performance_optimizer


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    active_requests: int
    request_rate: float
    response_time_ms: float
    error_rate: float
    queue_length: int
    throughput: float


@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    name: str
    metric: str  # cpu, memory, requests, response_time, etc.
    threshold_up: float
    threshold_down: float
    scale_up_action: str
    scale_down_action: str
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10


@dataclass
class WorkerInstance:
    """Worker instance for load balancing."""
    instance_id: str
    status: str  # active, busy, idle, stopping
    current_load: float
    max_capacity: int
    active_requests: int
    created_at: float
    last_health_check: float
    performance_score: float = 1.0


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: str = "least_connections"):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy ('round_robin', 'least_connections', 
                     'weighted_round_robin', 'response_time')
        """
        self.strategy = strategy
        self.workers: Dict[str, WorkerInstance] = {}
        self.request_counter = 0
        self.lock = threading.RLock()
        
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
        
        # Performance tracking
        self.response_times = {}
        self.error_counts = {}
    
    def add_worker(self, 
                   worker_id: str,
                   max_capacity: int = 10,
                   initial_score: float = 1.0):
        """Add worker to load balancer."""
        with self.lock:
            worker = WorkerInstance(
                instance_id=worker_id,
                status="active",
                current_load=0.0,
                max_capacity=max_capacity,
                active_requests=0,
                created_at=time.time(),
                last_health_check=time.time(),
                performance_score=initial_score
            )
            
            self.workers[worker_id] = worker
            self.response_times[worker_id] = []
            self.error_counts[worker_id] = 0
            
            self.logger.log_info(f"Added worker {worker_id} with capacity {max_capacity}")
    
    def remove_worker(self, worker_id: str):
        """Remove worker from load balancer."""
        with self.lock:
            if worker_id in self.workers:
                # Mark as stopping first
                self.workers[worker_id].status = "stopping"
                
                # Wait for active requests to finish (with timeout)
                timeout = time.time() + 30  # 30 second timeout
                while (self.workers[worker_id].active_requests > 0 and 
                       time.time() < timeout):
                    time.sleep(0.1)
                
                # Remove worker
                del self.workers[worker_id]
                del self.response_times[worker_id]
                del self.error_counts[worker_id]
                
                self.logger.log_info(f"Removed worker {worker_id}")
    
    def select_worker(self) -> Optional[str]:
        """Select best worker based on strategy."""
        with self.lock:
            available_workers = [
                w for w in self.workers.values() 
                if w.status == "active" and w.active_requests < w.max_capacity
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(available_workers)
            elif self.strategy == "least_connections":
                return self._least_connections_selection(available_workers)
            elif self.strategy == "weighted_round_robin":
                return self._weighted_selection(available_workers)
            elif self.strategy == "response_time":
                return self._response_time_selection(available_workers)
            else:
                # Default to least connections
                return self._least_connections_selection(available_workers)
    
    def _round_robin_selection(self, workers: List[WorkerInstance]) -> str:
        """Round-robin worker selection."""
        self.request_counter += 1
        return workers[self.request_counter % len(workers)].instance_id
    
    def _least_connections_selection(self, workers: List[WorkerInstance]) -> str:
        """Select worker with least active connections."""
        return min(workers, key=lambda w: w.active_requests).instance_id
    
    def _weighted_selection(self, workers: List[WorkerInstance]) -> str:
        """Weighted selection based on performance scores."""
        # Select based on performance score and available capacity
        weights = []
        for worker in workers:
            available_capacity = worker.max_capacity - worker.active_requests
            weight = worker.performance_score * (available_capacity / worker.max_capacity)
            weights.append(weight)
        
        # Select worker with highest weight
        best_worker = max(zip(workers, weights), key=lambda x: x[1])[0]
        return best_worker.instance_id
    
    def _response_time_selection(self, workers: List[WorkerInstance]) -> str:
        """Select worker with best response time."""
        best_worker = None
        best_score = float('inf')
        
        for worker in workers:
            response_times = self.response_times[worker.instance_id]
            if response_times:
                avg_response_time = statistics.mean(response_times[-10:])  # Last 10 requests
            else:
                avg_response_time = 0  # New worker, assume good performance
            
            # Consider both response time and current load
            load_factor = worker.active_requests / worker.max_capacity
            score = avg_response_time * (1 + load_factor)
            
            if score < best_score:
                best_score = score
                best_worker = worker
        
        return best_worker.instance_id if best_worker else workers[0].instance_id
    
    def start_request(self, worker_id: str):
        """Mark start of request on worker."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].active_requests += 1
                self.workers[worker_id].current_load = (
                    self.workers[worker_id].active_requests / 
                    self.workers[worker_id].max_capacity
                )
    
    def end_request(self, worker_id: str, response_time_ms: float, success: bool = True):
        """Mark end of request on worker."""
        with self.lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.active_requests = max(0, worker.active_requests - 1)
                worker.current_load = worker.active_requests / worker.max_capacity
                
                # Update performance metrics
                self.response_times[worker_id].append(response_time_ms)
                if len(self.response_times[worker_id]) > 100:
                    self.response_times[worker_id] = self.response_times[worker_id][-50:]
                
                if not success:
                    self.error_counts[worker_id] += 1
                
                # Update performance score
                self._update_performance_score(worker_id)
    
    def _update_performance_score(self, worker_id: str):
        """Update worker performance score based on metrics."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        response_times = self.response_times[worker_id]
        error_count = self.error_counts[worker_id]
        
        if not response_times:
            return
        
        # Calculate performance score based on response time and error rate
        avg_response_time = statistics.mean(response_times[-20:])  # Last 20 requests
        total_requests = len(response_times)
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        # Normalize scores (lower is better for both metrics)
        response_score = max(0.1, 1.0 / (1.0 + avg_response_time / 1000))  # Convert ms to seconds
        error_score = max(0.1, 1.0 - error_rate)
        
        # Combined performance score
        worker.performance_score = (response_score + error_score) / 2
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        with self.lock:
            total_capacity = sum(w.max_capacity for w in self.workers.values())
            active_requests = sum(w.active_requests for w in self.workers.values())
            
            return {
                "strategy": self.strategy,
                "workers": len(self.workers),
                "total_capacity": total_capacity,
                "active_requests": active_requests,
                "utilization": (active_requests / total_capacity) if total_capacity > 0 else 0,
                "worker_details": {
                    wid: {
                        "status": w.status,
                        "active_requests": w.active_requests,
                        "capacity": w.max_capacity,
                        "load": w.current_load,
                        "performance_score": w.performance_score
                    }
                    for wid, w in self.workers.items()
                }
            }


class AutoScaler:
    """Auto-scaling system for BCI-GPT workers."""
    
    def __init__(self, 
                 load_balancer: LoadBalancer,
                 check_interval: float = 30.0):
        """Initialize auto-scaler.
        
        Args:
            load_balancer: Load balancer instance
            check_interval: How often to check scaling conditions
        """
        self.load_balancer = load_balancer
        self.check_interval = check_interval
        
        self.scaling_rules: List[ScalingRule] = []
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scaling_action = {}
        
        self.running = False
        self.scaling_thread: Optional[threading.Thread] = None
        
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
        
        # Default scaling rules
        self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default scaling rules."""
        
        # CPU-based scaling
        self.add_scaling_rule(ScalingRule(
            name="cpu_scaling",
            metric="cpu_percent",
            threshold_up=75.0,
            threshold_down=25.0,
            scale_up_action="add_worker",
            scale_down_action="remove_worker",
            cooldown_seconds=300,
            min_instances=1,
            max_instances=5
        ))
        
        # Memory-based scaling
        self.add_scaling_rule(ScalingRule(
            name="memory_scaling", 
            metric="memory_percent",
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_action="add_worker",
            scale_down_action="remove_worker",
            cooldown_seconds=300,
            min_instances=1,
            max_instances=5
        ))
        
        # Request-based scaling
        self.add_scaling_rule(ScalingRule(
            name="request_scaling",
            metric="active_requests",
            threshold_up=8.0,  # Average requests per worker
            threshold_down=2.0,
            scale_up_action="add_worker", 
            scale_down_action="remove_worker",
            cooldown_seconds=180,
            min_instances=1,
            max_instances=8
        ))
        
        # Response time scaling
        self.add_scaling_rule(ScalingRule(
            name="response_time_scaling",
            metric="response_time_ms",
            threshold_up=2000.0,  # 2 seconds
            threshold_down=500.0,   # 0.5 seconds
            scale_up_action="add_worker",
            scale_down_action="remove_worker",
            cooldown_seconds=240,
            min_instances=1,
            max_instances=6
        ))
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule."""
        self.scaling_rules.append(rule)
        self.last_scaling_action[rule.name] = 0
        self.logger.log_info(f"Added scaling rule: {rule.name}")
    
    def start_scaling(self):
        """Start auto-scaling monitoring."""
        if self.running:
            self.logger.log_warning("Auto-scaling already running")
            return
        
        self.running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        self.logger.log_info("Started auto-scaling")
    
    def stop_scaling(self):
        """Stop auto-scaling monitoring."""
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        self.logger.log_info("Stopped auto-scaling")
    
    def _scaling_loop(self):
        """Main scaling monitoring loop."""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                self.metrics_history.append(current_metrics)
                
                # Keep limited history
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-50:]
                
                # Evaluate scaling rules
                for rule in self.scaling_rules:
                    self._evaluate_scaling_rule(rule, current_metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.log_error("Auto-scaling loop error", e)
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        current_time = time.time()
        
        # System metrics
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
        else:
            cpu_percent = 0.0
            memory_percent = 0.0
        
        # Load balancer metrics
        lb_status = self.load_balancer.get_status()
        active_requests = lb_status["active_requests"]
        total_capacity = lb_status["total_capacity"]
        
        # Calculate additional metrics
        request_rate = 0.0
        response_time_ms = 0.0
        error_rate = 0.0
        throughput = 0.0
        
        if len(self.metrics_history) > 1:
            # Calculate request rate from recent history
            recent_metrics = self.metrics_history[-5:]  # Last 5 samples
            if len(recent_metrics) >= 2:
                time_diff = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
                if time_diff > 0:
                    request_diff = recent_metrics[-1].active_requests - recent_metrics[0].active_requests
                    request_rate = abs(request_diff) / time_diff
        
        # Get performance optimizer stats for response time
        optimizer = get_performance_optimizer()
        if optimizer.profiles:
            # Average response time from profiled functions
            total_time = 0.0
            total_calls = 0
            for profile_data in optimizer.profiles.values():
                if profile_data['calls'] > 0:
                    total_time += profile_data['total_time']
                    total_calls += profile_data['calls']
            
            if total_calls > 0:
                response_time_ms = (total_time / total_calls) * 1000
        
        return ScalingMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_requests=active_requests,
            request_rate=request_rate,
            response_time_ms=response_time_ms,
            error_rate=error_rate,
            queue_length=0,  # Would be actual queue length in real implementation
            throughput=throughput
        )
    
    def _evaluate_scaling_rule(self, rule: ScalingRule, metrics: ScalingMetrics):
        """Evaluate whether scaling action is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action[rule.name] < rule.cooldown_seconds:
            return
        
        # Get metric value
        metric_value = getattr(metrics, rule.metric, 0.0)
        
        # Special handling for request-based metrics
        if rule.metric == "active_requests":
            # Convert to average requests per worker
            num_workers = len(self.load_balancer.workers)
            if num_workers > 0:
                metric_value = metric_value / num_workers
        
        current_instances = len(self.load_balancer.workers)
        
        # Determine scaling action
        scaling_action = None
        
        if (metric_value > rule.threshold_up and 
            current_instances < rule.max_instances):
            scaling_action = rule.scale_up_action
            
        elif (metric_value < rule.threshold_down and 
              current_instances > rule.min_instances):
            scaling_action = rule.scale_down_action
        
        # Execute scaling action
        if scaling_action:
            success = self._execute_scaling_action(scaling_action, rule, metric_value)
            
            if success:
                self.last_scaling_action[rule.name] = current_time
                self.logger.log_info(
                    f"Executed scaling action: {scaling_action} for rule {rule.name} "
                    f"(metric: {rule.metric}={metric_value:.2f})"
                )
    
    def _execute_scaling_action(self, action: str, rule: ScalingRule, metric_value: float) -> bool:
        """Execute scaling action."""
        try:
            if action == "add_worker":
                return self._add_worker()
            elif action == "remove_worker":
                return self._remove_worker()
            else:
                self.logger.log_warning(f"Unknown scaling action: {action}")
                return False
                
        except Exception as e:
            self.logger.log_error(f"Scaling action {action} failed", e)
            return False
    
    def _add_worker(self) -> bool:
        """Add new worker instance."""
        worker_id = f"worker_{int(time.time())}"
        
        try:
            # In a real implementation, this would spawn a new process/container
            # For now, just add to load balancer
            self.load_balancer.add_worker(worker_id, max_capacity=10)
            
            self.metrics.record_error(
                'info', 'autoscaler',
                f'Added worker {worker_id}',
                context={'action': 'scale_up', 'worker_id': worker_id}
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to add worker {worker_id}", e)
            return False
    
    def _remove_worker(self) -> bool:
        """Remove worker instance."""
        # Select worker with lowest load to remove
        workers = list(self.load_balancer.workers.values())
        if not workers:
            return False
        
        # Find worker with lowest current load
        worker_to_remove = min(workers, key=lambda w: w.current_load)
        
        try:
            self.load_balancer.remove_worker(worker_to_remove.instance_id)
            
            self.metrics.record_error(
                'info', 'autoscaler',
                f'Removed worker {worker_to_remove.instance_id}',
                context={'action': 'scale_down', 'worker_id': worker_to_remove.instance_id}
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to remove worker {worker_to_remove.instance_id}", e)
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "running": self.running,
            "rules_count": len(self.scaling_rules),
            "workers_count": len(self.load_balancer.workers),
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent if current_metrics else 0,
                "memory_percent": current_metrics.memory_percent if current_metrics else 0,
                "active_requests": current_metrics.active_requests if current_metrics else 0,
                "response_time_ms": current_metrics.response_time_ms if current_metrics else 0
            } if current_metrics else {},
            "last_scaling_actions": {
                name: datetime.fromtimestamp(timestamp).isoformat()
                for name, timestamp in self.last_scaling_action.items()
                if timestamp > 0
            },
            "load_balancer_status": self.load_balancer.get_status()
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: int = 60,
                 expected_exception: type = Exception):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before trying again
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.lock = threading.Lock()
        self.logger = get_logger()
    
    def __call__(self, func):
        """Decorator implementation."""
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        with self.lock:
            current_time = time.time()
            
            # Check if we should transition from OPEN to HALF_OPEN
            if (self.state == "OPEN" and self.last_failure_time and 
                current_time - self.last_failure_time >= self.timeout_seconds):
                self.state = "HALF_OPEN"
                self.logger.log_info(f"Circuit breaker for {func.__name__} moved to HALF_OPEN")
            
            # If circuit is OPEN, fail fast
            if self.state == "OPEN":
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        # Try to execute function
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker if it was HALF_OPEN
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.failure_count = 0
                    self.state = "CLOSED"
                    self.logger.log_info(f"Circuit breaker for {func.__name__} CLOSED after success")
            
            return result
            
        except self.expected_exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    self.logger.log_warning(
                        f"Circuit breaker OPENED for {func.__name__} "
                        f"after {self.failure_count} failures"
                    )
            
            raise


# Global instances
_global_load_balancer: Optional[LoadBalancer] = None
_global_auto_scaler: Optional[AutoScaler] = None


def get_load_balancer(strategy: str = "least_connections") -> LoadBalancer:
    """Get or create global load balancer."""
    global _global_load_balancer
    
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer(strategy=strategy)
    
    return _global_load_balancer


def get_auto_scaler() -> AutoScaler:
    """Get or create global auto-scaler."""
    global _global_auto_scaler
    
    if _global_auto_scaler is None:
        load_balancer = get_load_balancer()
        _global_auto_scaler = AutoScaler(load_balancer)
    
    return _global_auto_scaler


def circuit_breaker(failure_threshold: int = 5, timeout_seconds: int = 60):
    """Decorator for circuit breaker pattern."""
    def decorator(func):
        breaker = CircuitBreaker(failure_threshold, timeout_seconds)
        return breaker(func)
    return decorator