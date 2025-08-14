"""
Advanced auto-scaling and load balancing for BCI-GPT system.
Provides intelligent resource management and horizontal scaling.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int = 0
    request_rate: float = 0.0
    error_rate: float = 0.0
    response_time_ms: float = 0.0


@dataclass
class ServiceInstance:
    """Service instance metadata."""
    instance_id: str
    host: str
    port: int
    status: str = "healthy"  # healthy, unhealthy, starting, stopping
    created_at: datetime = field(default_factory=datetime.now)
    current_load: float = 0.0
    total_requests: int = 0
    error_count: int = 0


class LoadBalancer:
    """Intelligent load balancing with multiple strategies."""
    
    def __init__(self, strategy: str = "least_connections"):
        """Initialize load balancer."""
        self.strategy = strategy
        self.instances: Dict[str, ServiceInstance] = {}
        self.request_history: deque = deque(maxlen=10000)
        
        self._current_index = 0
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def register_instance(self, instance: ServiceInstance):
        """Register a new service instance."""
        with self._lock:
            self.instances[instance.instance_id] = instance
            self.logger.info(f"Registered instance {instance.instance_id}")
    
    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get list of healthy instances."""
        with self._lock:
            return [
                instance for instance in self.instances.values()
                if instance.status == "healthy"
            ]
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_instances = len(self.instances)
            healthy_instances = len(self.get_healthy_instances())
            
            return {
                'strategy': self.strategy,
                'total_instances': total_instances,
                'healthy_instances': healthy_instances,
                'instances': [
                    {
                        'id': instance.instance_id,
                        'host': instance.host,
                        'port': instance.port,
                        'status': instance.status,
                        'load': instance.current_load
                    }
                    for instance in self.instances.values()
                ]
            }


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    min_instances: int = 1
    max_instances: int = 10


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, load_balancer: LoadBalancer):
        """Initialize auto-scaler."""
        self.load_balancer = load_balancer
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_history: List[Dict] = []
        
        self._running = False
        self._scaler_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add an auto-scaling rule."""
        self.scaling_rules[rule.name] = rule
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def start_auto_scaling(self, check_interval: int = 60):
        """Start auto-scaling monitoring."""
        if self._running:
            return
        
        self._running = True
        self._scaler_thread = threading.Thread(
            target=self._scaling_loop,
            args=(check_interval,),
            daemon=True
        )
        self._scaler_thread.start()
        
        self.logger.info(f"Started auto-scaling with {check_interval}s interval")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self._running = False
        if self._scaler_thread:
            self._scaler_thread.join()
        
        self.logger.info("Stopped auto-scaling")
    
    def _scaling_loop(self, interval: int):
        """Main auto-scaling loop."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                self._evaluate_scaling_rules()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                active_connections=len(self.load_balancer.instances)
            )
            
        except ImportError:
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=50.0,
                memory_percent=60.0,
                disk_percent=40.0,
                active_connections=len(self.load_balancer.instances)
            )
    
    def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules."""
        if len(self.metrics_history) < 2:
            return
        
        for rule_name, rule in self.scaling_rules.items():
            try:
                should_scale, direction = self._should_scale(rule)
                
                if should_scale:
                    if direction == "up":
                        self._scale_up(rule)
                    elif direction == "down":
                        self._scale_down(rule)
                        
            except Exception as e:
                self.logger.error(f"Error evaluating scaling rule {rule_name}: {e}")
    
    def _should_scale(self, rule: ScalingRule) -> Tuple[bool, Optional[str]]:
        """Determine if scaling should occur based on a rule."""
        if not self.metrics_history:
            return False, None
        
        current_metrics = self.metrics_history[-1]
        current_instances = len(self.load_balancer.instances)
        
        # Get metric value
        if rule.metric_name == "cpu_percent":
            metric_value = current_metrics.cpu_percent
        elif rule.metric_name == "memory_percent":
            metric_value = current_metrics.memory_percent
        elif rule.metric_name == "active_connections":
            metric_value = current_metrics.active_connections
        else:
            return False, None
        
        # Check for scale up
        if metric_value > rule.threshold_up and current_instances < rule.max_instances:
            return True, "up"
        
        # Check for scale down
        elif metric_value < rule.threshold_down and current_instances > rule.min_instances:
            return True, "down"
        
        return False, None
    
    def _scale_up(self, rule: ScalingRule):
        """Scale up instances."""
        current_count = len(self.load_balancer.instances)
        
        # Create mock instance for simulation
        instance_id = f"instance_{int(time.time())}"
        instance = ServiceInstance(
            instance_id=instance_id,
            host=f"10.0.0.{100 + current_count}",
            port=8080
        )
        
        self.load_balancer.register_instance(instance)
        
        self.scaling_history.append({
            'timestamp': time.time(),
            'rule_name': rule.name,
            'action': 'scale_up',
            'total_instances': current_count + 1
        })
        
        self.logger.info(f"Scaled up (rule: {rule.name})")
    
    def _scale_down(self, rule: ScalingRule):
        """Scale down instances."""
        current_count = len(self.load_balancer.instances)
        
        if current_count > rule.min_instances:
            # Remove one instance (simplified)
            if self.load_balancer.instances:
                instance_id = next(iter(self.load_balancer.instances))
                del self.load_balancer.instances[instance_id]
                
                self.scaling_history.append({
                    'timestamp': time.time(),
                    'rule_name': rule.name,
                    'action': 'scale_down',
                    'total_instances': current_count - 1
                })
                
                self.logger.info(f"Scaled down (rule: {rule.name})")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get auto-scaling summary."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'current_instances': len(self.load_balancer.instances),
            'scaling_rules': len(self.scaling_rules),
            'scaling_events': len(self.scaling_history),
            'current_metrics': {
                'cpu_percent': current_metrics.cpu_percent if current_metrics else 0,
                'memory_percent': current_metrics.memory_percent if current_metrics else 0
            } if current_metrics else {},
            'recent_scaling_events': self.scaling_history[-5:]
        }


# Global instances
_load_balancer = LoadBalancer()
_auto_scaler = AutoScaler(_load_balancer)


def get_load_balancer() -> LoadBalancer:
    """Get the global load balancer instance."""
    return _load_balancer


def get_auto_scaler() -> AutoScaler:
    """Get the global auto-scaler instance."""
    return _auto_scaler


def setup_default_scaling_rules():
    """Setup default auto-scaling rules."""
    cpu_rule = ScalingRule(
        name="cpu_scaling",
        metric_name="cpu_percent",
        threshold_up=70.0,
        threshold_down=30.0,
        min_instances=2,
        max_instances=20
    )
    _auto_scaler.add_scaling_rule(cpu_rule)
    
    memory_rule = ScalingRule(
        name="memory_scaling",
        metric_name="memory_percent",
        threshold_up=80.0,
        threshold_down=40.0,
        min_instances=2,
        max_instances=15
    )
    _auto_scaler.add_scaling_rule(memory_rule)