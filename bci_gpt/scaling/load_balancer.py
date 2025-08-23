"""
Advanced Load Balancer for BCI-GPT System

Provides intelligent load balancing with health-aware routing,
sticky sessions, and performance optimization for BCI workloads.
"""

import time
import random
import threading
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RESOURCE_USAGE = "resource_usage"
    HASH_BASED = "hash_based"
    ADAPTIVE = "adaptive"


class InstanceStatus(Enum):
    """Instance health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class InstanceMetrics:
    """Metrics for a service instance"""
    instance_id: str
    timestamp: float = field(default_factory=time.time)
    active_connections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    health_score: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'timestamp': self.timestamp,
            'active_connections': self.active_connections,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'response_time_ms': self.response_time_ms,
            'requests_per_second': self.requests_per_second,
            'error_rate': self.error_rate,
            'health_score': self.health_score
        }


@dataclass
class ServiceInstance:
    """Service instance configuration"""
    instance_id: str
    endpoint: str
    weight: float = 1.0
    max_connections: int = 100
    status: InstanceStatus = InstanceStatus.HEALTHY
    metrics: Optional[InstanceMetrics] = None
    last_health_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = InstanceMetrics(self.instance_id)
    
    def is_available(self) -> bool:
        """Check if instance is available for requests"""
        return self.status in [InstanceStatus.HEALTHY, InstanceStatus.DEGRADED]
    
    def is_overloaded(self) -> bool:
        """Check if instance is overloaded"""
        if self.metrics:
            return (self.metrics.active_connections >= self.max_connections or
                    self.metrics.cpu_usage > 90.0 or
                    self.metrics.memory_usage > 90.0)
        return False


class LoadBalancingAlgorithm(ABC):
    """Abstract base class for load balancing algorithms"""
    
    @abstractmethod
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Select an instance for the request"""
        pass


class RoundRobinAlgorithm(LoadBalancingAlgorithm):
    """Round robin load balancing"""
    
    def __init__(self):
        self._current_index = 0
        self._lock = threading.Lock()
    
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Select next instance in round-robin order"""
        available_instances = [inst for inst in instances if inst.is_available() and not inst.is_overloaded()]
        
        if not available_instances:
            return None
        
        with self._lock:
            instance = available_instances[self._current_index % len(available_instances)]
            self._current_index += 1
            
        return instance


class LeastConnectionsAlgorithm(LoadBalancingAlgorithm):
    """Least connections load balancing"""
    
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Select instance with least active connections"""
        available_instances = [inst for inst in instances if inst.is_available() and not inst.is_overloaded()]
        
        if not available_instances:
            return None
        
        # Find instance with minimum connections
        return min(available_instances, key=lambda inst: inst.metrics.active_connections)


class WeightedRoundRobinAlgorithm(LoadBalancingAlgorithm):
    """Weighted round robin load balancing"""
    
    def __init__(self):
        self._weights: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Select instance based on weights"""
        available_instances = [inst for inst in instances if inst.is_available() and not inst.is_overloaded()]
        
        if not available_instances:
            return None
        
        with self._lock:
            # Update weights if needed
            for inst in available_instances:
                if inst.instance_id not in self._weights:
                    self._weights[inst.instance_id] = int(inst.weight * 10)
            
            # Find instance with highest remaining weight
            best_instance = None
            highest_weight = -1
            
            for inst in available_instances:
                current_weight = self._weights.get(inst.instance_id, 0)
                if current_weight > highest_weight:
                    highest_weight = current_weight
                    best_instance = inst
            
            # Decrement weight and reset if all are zero
            if best_instance:
                self._weights[best_instance.instance_id] -= 1
                
                # Reset weights if all are zero
                if all(w <= 0 for w in self._weights.values()):
                    for inst in available_instances:
                        self._weights[inst.instance_id] = int(inst.weight * 10)
            
            return best_instance


class ResponseTimeAlgorithm(LoadBalancingAlgorithm):
    """Response time based load balancing"""
    
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Select instance with best response time"""
        available_instances = [inst for inst in instances if inst.is_available() and not inst.is_overloaded()]
        
        if not available_instances:
            return None
        
        # Select instance with lowest response time
        return min(available_instances, key=lambda inst: inst.metrics.response_time_ms)


class HashBasedAlgorithm(LoadBalancingAlgorithm):
    """Hash-based load balancing for sticky sessions"""
    
    def __init__(self, hash_key: str = "session_id"):
        self.hash_key = hash_key
    
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Select instance based on hash of request context"""
        available_instances = [inst for inst in instances if inst.is_available()]
        
        if not available_instances:
            return None
        
        # Use hash for consistent routing
        if request_context and self.hash_key in request_context:
            hash_value = hashlib.md5(str(request_context[self.hash_key]).encode()).hexdigest()
            index = int(hash_value, 16) % len(available_instances)
            
            selected = available_instances[index]
            
            # If selected instance is overloaded, fall back to least connections
            if selected.is_overloaded():
                fallback = LeastConnectionsAlgorithm()
                return fallback.select_instance(instances, request_context)
            
            return selected
        
        # Fall back to round robin if no hash key
        fallback = RoundRobinAlgorithm()
        return fallback.select_instance(instances, request_context)


class AdaptiveAlgorithm(LoadBalancingAlgorithm):
    """Adaptive load balancing that adjusts strategy based on conditions"""
    
    def __init__(self):
        self.algorithms = {
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsAlgorithm(),
            LoadBalancingStrategy.RESPONSE_TIME: ResponseTimeAlgorithm(),
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinAlgorithm()
        }
        self.current_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """Adaptively select best algorithm and instance"""
        
        # Analyze current system state
        self._adapt_strategy(instances)
        
        # Use selected algorithm
        algorithm = self.algorithms[self.current_strategy]
        return algorithm.select_instance(instances, request_context)
    
    def _adapt_strategy(self, instances: List[ServiceInstance]) -> None:
        """Adapt strategy based on current system state"""
        if not instances:
            return
        
        # Calculate system metrics
        total_instances = len(instances)
        healthy_instances = len([inst for inst in instances if inst.status == InstanceStatus.HEALTHY])
        avg_cpu = sum(inst.metrics.cpu_usage for inst in instances) / total_instances
        avg_response_time = sum(inst.metrics.response_time_ms for inst in instances) / total_instances
        
        # Choose strategy based on conditions
        if healthy_instances / total_instances < 0.5:
            # Low availability - use least connections
            self.current_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        elif avg_response_time > 500:
            # High latency - optimize for response time
            self.current_strategy = LoadBalancingStrategy.RESPONSE_TIME
        elif avg_cpu < 30:
            # Low utilization - simple round robin
            self.current_strategy = LoadBalancingStrategy.ROUND_ROBIN
        else:
            # Balanced load - use least connections
            self.current_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS


class LoadBalancer:
    """
    Advanced load balancer for BCI-GPT services
    
    Provides intelligent load balancing with health monitoring,
    multiple algorithms, and performance optimization.
    """
    
    def __init__(
        self,
        name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        health_check_interval: float = 30.0,
        sticky_sessions: bool = False
    ):
        self.name = name
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.sticky_sessions = sticky_sessions
        
        self.instances: Dict[str, ServiceInstance] = {}
        self.algorithm = self._create_algorithm(strategy)
        self.request_count = 0
        self.total_response_time = 0.0
        
        self._health_check_thread: Optional[threading.Thread] = None
        self._health_monitoring = False
        self._lock = threading.Lock()
        
        # Health check callbacks
        self.health_checkers: List[Callable[[ServiceInstance], Tuple[InstanceStatus, InstanceMetrics]]] = []
        
        logger.info(f"Load balancer '{name}' initialized with strategy: {strategy.value}")
    
    def _create_algorithm(self, strategy: LoadBalancingStrategy) -> LoadBalancingAlgorithm:
        """Create load balancing algorithm instance"""
        
        algorithms = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinAlgorithm(),
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsAlgorithm(),
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinAlgorithm(),
            LoadBalancingStrategy.RESPONSE_TIME: ResponseTimeAlgorithm(),
            LoadBalancingStrategy.HASH_BASED: HashBasedAlgorithm(),
            LoadBalancingStrategy.ADAPTIVE: AdaptiveAlgorithm()
        }
        
        return algorithms.get(strategy, AdaptiveAlgorithm())
    
    def add_instance(self, instance: ServiceInstance) -> None:
        """Add service instance"""
        with self._lock:
            self.instances[instance.instance_id] = instance
        logger.info(f"Added instance {instance.instance_id} to load balancer {self.name}")
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove service instance"""
        with self._lock:
            removed = self.instances.pop(instance_id, None)
        if removed:
            logger.info(f"Removed instance {instance_id} from load balancer {self.name}")
        return removed is not None
    
    def set_instance_status(self, instance_id: str, status: InstanceStatus) -> bool:
        """Set instance status"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].status = status
                logger.info(f"Set instance {instance_id} status to {status.value}")
                return True
        return False
    
    def drain_instance(self, instance_id: str) -> bool:
        """Drain instance (stop sending new requests)"""
        return self.set_instance_status(instance_id, InstanceStatus.DRAINING)
    
    def select_instance(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """Select best instance for request"""
        with self._lock:
            instances = list(self.instances.values())
        
        if not instances:
            logger.warning(f"No instances available in load balancer {self.name}")
            return None
        
        # Use algorithm to select instance
        selected = self.algorithm.select_instance(instances, request_context)
        
        if selected:
            # Update connection count
            selected.metrics.active_connections += 1
            logger.debug(f"Selected instance {selected.instance_id} for request")
        else:
            logger.warning(f"No suitable instance found in load balancer {self.name}")
        
        return selected
    
    def release_instance(self, instance_id: str, response_time_ms: float = 0.0) -> None:
        """Release instance after request completion"""
        with self._lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                if instance.metrics.active_connections > 0:
                    instance.metrics.active_connections -= 1
                
                # Update response time metrics
                if response_time_ms > 0:
                    instance.metrics.response_time_ms = (
                        instance.metrics.response_time_ms * 0.9 + response_time_ms * 0.1
                    )
                
                # Update global metrics
                self.request_count += 1
                self.total_response_time += response_time_ms
    
    def add_health_checker(self, checker: Callable[[ServiceInstance], Tuple[InstanceStatus, InstanceMetrics]]) -> None:
        """Add health check callback"""
        self.health_checkers.append(checker)
    
    def start_health_monitoring(self) -> None:
        """Start health monitoring"""
        if self._health_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._health_monitoring = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        logger.info(f"Started health monitoring for load balancer {self.name}")
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring"""
        if not self._health_monitoring:
            return
        
        self._health_monitoring = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")
    
    def _health_check_loop(self) -> None:
        """Health check monitoring loop"""
        while self._health_monitoring:
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            # Sleep in small increments for responsive shutdown
            elapsed = 0
            while elapsed < self.health_check_interval and self._health_monitoring:
                time.sleep(min(1.0, self.health_check_interval - elapsed))
                elapsed += 1.0
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all instances"""
        with self._lock:
            instances = list(self.instances.values())
        
        for instance in instances:
            try:
                # Default health check (mock implementation)
                new_status, new_metrics = self._default_health_check(instance)
                
                # Run custom health checkers
                for checker in self.health_checkers:
                    try:
                        status, metrics = checker(instance)
                        if status != InstanceStatus.HEALTHY:
                            new_status = status
                        # Merge metrics (would be more sophisticated in production)
                        if metrics:
                            new_metrics = metrics
                    except Exception as e:
                        logger.error(f"Health checker failed for {instance.instance_id}: {e}")
                
                # Update instance
                with self._lock:
                    if instance.instance_id in self.instances:
                        self.instances[instance.instance_id].status = new_status
                        self.instances[instance.instance_id].metrics = new_metrics
                        self.instances[instance.instance_id].last_health_check = time.time()
                
            except Exception as e:
                logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
    
    def _default_health_check(self, instance: ServiceInstance) -> Tuple[InstanceStatus, InstanceMetrics]:
        """Default health check implementation"""
        # Mock health check - in production would make actual HTTP requests
        
        # Simulate health check with some randomness
        health_score = 100.0 - random.uniform(0, 20)
        
        if health_score > 80:
            status = InstanceStatus.HEALTHY
        elif health_score > 60:
            status = InstanceStatus.DEGRADED
        else:
            status = InstanceStatus.UNHEALTHY
        
        # Generate mock metrics
        metrics = InstanceMetrics(
            instance_id=instance.instance_id,
            cpu_usage=random.uniform(20, 80),
            memory_usage=random.uniform(30, 90),
            response_time_ms=random.uniform(50, 300),
            requests_per_second=random.uniform(10, 100),
            error_rate=random.uniform(0, 0.05),
            health_score=health_score
        )
        
        return status, metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status"""
        with self._lock:
            instance_statuses = {
                instance_id: {
                    'status': instance.status.value,
                    'metrics': instance.metrics.to_dict(),
                    'weight': instance.weight,
                    'last_health_check': instance.last_health_check
                }
                for instance_id, instance in self.instances.items()
            }
        
        avg_response_time = (
            self.total_response_time / max(1, self.request_count)
        ) if self.request_count > 0 else 0.0
        
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'total_instances': len(self.instances),
            'healthy_instances': len([
                inst for inst in self.instances.values() 
                if inst.status == InstanceStatus.HEALTHY
            ]),
            'request_count': self.request_count,
            'average_response_time_ms': avg_response_time,
            'health_monitoring': self._health_monitoring,
            'instances': instance_statuses
        }


def create_bci_load_balancer() -> LoadBalancer:
    """Create load balancer optimized for BCI workloads"""
    
    balancer = LoadBalancer(
        name="bci_inference",
        strategy=LoadBalancingStrategy.ADAPTIVE,
        health_check_interval=15.0,  # Frequent health checks for BCI
        sticky_sessions=False  # BCI processing doesn't need sticky sessions
    )
    
    # Add BCI-specific health checker
    def bci_health_checker(instance: ServiceInstance) -> Tuple[InstanceStatus, InstanceMetrics]:
        """BCI-specific health check focusing on latency"""
        
        # Mock BCI-specific metrics
        eeg_processing_latency = random.uniform(10, 100)  # ms
        model_inference_latency = random.uniform(20, 150)  # ms
        total_latency = eeg_processing_latency + model_inference_latency
        
        # BCI has strict latency requirements
        if total_latency < 100:
            status = InstanceStatus.HEALTHY
        elif total_latency < 200:
            status = InstanceStatus.DEGRADED
        else:
            status = InstanceStatus.UNHEALTHY
        
        metrics = InstanceMetrics(
            instance_id=instance.instance_id,
            response_time_ms=total_latency,
            cpu_usage=random.uniform(30, 70),
            memory_usage=random.uniform(40, 80),
            health_score=max(0, 100 - total_latency)
        )
        
        return status, metrics
    
    balancer.add_health_checker(bci_health_checker)
    
    return balancer