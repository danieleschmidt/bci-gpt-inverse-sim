"""Auto-scaling system for BCI-GPT."""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from ..utils.logging_config import get_logger
from ..optimization.performance_optimizer import PerformanceOptimizer


class ScalingDirection(Enum):
    """Scaling direction enumeration."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


@dataclass
class ScalingMetric:
    """Scaling metric definition."""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScalingAction:
    """Scaling action definition."""
    direction: ScalingDirection
    magnitude: float
    reason: str
    timestamp: datetime
    metrics: Dict[str, float]


class ResourceManager:
    """Manages system resources for scaling."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.logger = get_logger(__name__)
        self.current_resources = {
            "cpu_cores": 4,
            "memory_gb": 8,
            "worker_threads": 4,
            "connection_pool_size": 20,
            "cache_size": 1000,
            "batch_size": 16
        }
        self.min_resources = {
            "cpu_cores": 1,
            "memory_gb": 2,
            "worker_threads": 1,
            "connection_pool_size": 5,
            "cache_size": 100,
            "batch_size": 1
        }
        self.max_resources = {
            "cpu_cores": 16,
            "memory_gb": 64,
            "worker_threads": 32,
            "connection_pool_size": 100,
            "cache_size": 10000,
            "batch_size": 128
        }
        self.scaling_callbacks = {}
    
    def scale_resource(self, resource_name: str, factor: float) -> bool:
        """Scale a specific resource."""
        if resource_name not in self.current_resources:
            return False
        
        current = self.current_resources[resource_name]
        min_val = self.min_resources[resource_name]
        max_val = self.max_resources[resource_name]
        
        # Calculate new value
        if factor > 1.0:
            new_value = min(current * factor, max_val)
        else:
            new_value = max(current * factor, min_val)
        
        # Round to integer for certain resources
        if resource_name in ["cpu_cores", "worker_threads", "connection_pool_size", "cache_size", "batch_size"]:
            new_value = int(new_value)
        
        # Apply change if significant
        if abs(new_value - current) / current > 0.1:  # 10% threshold
            self.current_resources[resource_name] = new_value
            
            # Trigger callback if registered
            if resource_name in self.scaling_callbacks:
                try:
                    self.scaling_callbacks[resource_name](new_value)
                except Exception as e:
                    self.logger.log_error(f"Scaling callback failed for {resource_name}: {e}")
            
            self.logger.log_info(f"Scaled {resource_name} from {current} to {new_value}")
            return True
        
        return False
    
    def register_scaling_callback(self, resource_name: str, callback: Callable[[float], None]):
        """Register callback for resource scaling."""
        self.scaling_callbacks[resource_name] = callback
        self.logger.log_info(f"Registered scaling callback for {resource_name}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            "current": self.current_resources.copy(),
            "utilization": {
                name: (current - self.min_resources[name]) / (self.max_resources[name] - self.min_resources[name])
                for name, current in self.current_resources.items()
            },
            "scalable": {
                name: current < self.max_resources[name]
                for name, current in self.current_resources.items()
            }
        }


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, performance_optimizer: Optional[PerformanceOptimizer] = None):
        """Initialize auto-scaler."""
        self.logger = get_logger(__name__)
        self.performance_optimizer = performance_optimizer
        self.resource_manager = ResourceManager()
        
        # Scaling configuration
        self.scaling_enabled = True
        self.scaling_interval = 60.0  # seconds
        self.cooldown_period = 300.0  # 5 minutes
        self.last_scaling_action = None
        
        # Metrics for scaling decisions
        self.scaling_metrics = {
            "cpu_usage": ScalingMetric("cpu_usage", 0.0, 75.0, 25.0, 1.0),
            "memory_usage": ScalingMetric("memory_usage", 0.0, 80.0, 30.0, 1.0),
            "request_rate": ScalingMetric("request_rate", 0.0, 100.0, 20.0, 0.8),
            "response_time": ScalingMetric("response_time", 0.0, 200.0, 50.0, 1.2),
            "queue_length": ScalingMetric("queue_length", 0.0, 50.0, 5.0, 0.9),
            "error_rate": ScalingMetric("error_rate", 0.0, 5.0, 1.0, 1.5)
        }
        
        # Scaling history
        self.scaling_history = deque(maxlen=1000)
        
        # Auto-scaling thread
        self.scaling_active = False
        self.scaling_thread = None
    
    def start_auto_scaling(self):
        """Start auto-scaling."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        self.logger.log_info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling."""
        self.scaling_active = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        self.logger.log_info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling loop."""
        while self.scaling_active:
            try:
                if self.scaling_enabled:
                    self._evaluate_scaling_needs()
            except Exception as e:
                self.logger.log_error(f"Scaling evaluation error: {e}")
            
            time.sleep(self.scaling_interval)
    
    def _evaluate_scaling_needs(self):
        """Evaluate if scaling is needed."""
        # Check cooldown period
        if self._in_cooldown_period():
            return
        
        # Collect current metrics
        current_metrics = self._collect_metrics()
        
        # Update scaling metrics
        for name, value in current_metrics.items():
            if name in self.scaling_metrics:
                self.scaling_metrics[name].current_value = value
        
        # Make scaling decision
        scaling_decision = self._make_scaling_decision()
        
        if scaling_decision.direction != ScalingDirection.MAINTAIN:
            self._execute_scaling_action(scaling_decision)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # Simulate metric collection - in real implementation,
        # this would integrate with monitoring systems
        metrics = {
            "cpu_usage": 45.0,
            "memory_usage": 65.0,
            "request_rate": 75.0,
            "response_time": 120.0,
            "queue_length": 15.0,
            "error_rate": 2.0
        }
        
        # Add some realistic variation
        import random
        for key in metrics:
            metrics[key] += random.uniform(-5, 5)
            metrics[key] = max(0, metrics[key])
        
        return metrics
    
    def _make_scaling_decision(self) -> ScalingAction:
        """Make intelligent scaling decision."""
        scale_up_score = 0.0
        scale_down_score = 0.0
        reasons = []
        
        for metric in self.scaling_metrics.values():
            if not metric.enabled:
                continue
            
            if metric.current_value > metric.threshold_up:
                score = ((metric.current_value - metric.threshold_up) / metric.threshold_up) * metric.weight
                scale_up_score += score
                reasons.append(f"{metric.name} high ({metric.current_value:.1f} > {metric.threshold_up})")
            
            elif metric.current_value < metric.threshold_down:
                score = ((metric.threshold_down - metric.current_value) / metric.threshold_down) * metric.weight
                scale_down_score += score
                reasons.append(f"{metric.name} low ({metric.current_value:.1f} < {metric.threshold_down})")
        
        # Determine scaling direction and magnitude
        if scale_up_score > scale_down_score and scale_up_score > 1.0:
            direction = ScalingDirection.UP
            magnitude = min(2.0, 1.0 + (scale_up_score - 1.0) * 0.5)  # Cap at 2x scaling
            reason = f"Scale up needed: {', '.join(reasons[:3])}"
        elif scale_down_score > scale_up_score and scale_down_score > 1.0:
            direction = ScalingDirection.DOWN
            magnitude = max(0.5, 1.0 - (scale_down_score - 1.0) * 0.2)  # Cap at 0.5x scaling
            reason = f"Scale down possible: {', '.join(reasons[:3])}"
        else:
            direction = ScalingDirection.MAINTAIN
            magnitude = 1.0
            reason = "Metrics within acceptable ranges"
        
        return ScalingAction(
            direction=direction,
            magnitude=magnitude,
            reason=reason,
            timestamp=datetime.now(),
            metrics={metric.name: metric.current_value for metric in self.scaling_metrics.values()}
        )
    
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute scaling action."""
        self.logger.log_info(f"Executing scaling action: {action.direction.value} by {action.magnitude:.2f}x - {action.reason}")
        
        # Apply scaling to relevant resources
        if action.direction == ScalingDirection.UP:
            self._scale_up(action.magnitude)
        elif action.direction == ScalingDirection.DOWN:
            self._scale_down(action.magnitude)
        
        # Record action
        self.scaling_history.append(action)
        self.last_scaling_action = action
    
    def _scale_up(self, magnitude: float):
        """Scale up resources."""
        resources_to_scale = ["worker_threads", "connection_pool_size", "cache_size"]
        
        for resource in resources_to_scale:
            self.resource_manager.scale_resource(resource, magnitude)
        
        # Increase batch size if CPU usage is high
        cpu_metric = self.scaling_metrics.get("cpu_usage")
        if cpu_metric and cpu_metric.current_value > 70:
            self.resource_manager.scale_resource("batch_size", min(magnitude, 1.5))
    
    def _scale_down(self, magnitude: float):
        """Scale down resources."""
        resources_to_scale = ["worker_threads", "connection_pool_size", "cache_size"]
        
        for resource in resources_to_scale:
            self.resource_manager.scale_resource(resource, magnitude)
        
        # Decrease batch size if memory usage is low
        memory_metric = self.scaling_metrics.get("memory_usage")
        if memory_metric and memory_metric.current_value < 40:
            self.resource_manager.scale_resource("batch_size", max(magnitude, 0.7))
    
    def _in_cooldown_period(self) -> bool:
        """Check if we're in cooldown period."""
        if not self.last_scaling_action:
            return False
        
        time_since_last = (datetime.now() - self.last_scaling_action.timestamp).total_seconds()
        return time_since_last < self.cooldown_period
    
    def update_metric(self, name: str, value: float):
        """Update a specific metric value."""
        if name in self.scaling_metrics:
            self.scaling_metrics[name].current_value = value
    
    def configure_metric(self, name: str, threshold_up: Optional[float] = None,
                        threshold_down: Optional[float] = None, weight: Optional[float] = None,
                        enabled: Optional[bool] = None):
        """Configure scaling metric."""
        if name not in self.scaling_metrics:
            return False
        
        metric = self.scaling_metrics[name]
        
        if threshold_up is not None:
            metric.threshold_up = threshold_up
        if threshold_down is not None:
            metric.threshold_down = threshold_down
        if weight is not None:
            metric.weight = weight
        if enabled is not None:
            metric.enabled = enabled
        
        self.logger.log_info(f"Updated scaling metric configuration for {name}")
        return True
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status."""
        return {
            "enabled": self.scaling_enabled,
            "active": self.scaling_active,
            "cooldown_remaining": max(0, self.cooldown_period - (
                (datetime.now() - self.last_scaling_action.timestamp).total_seconds()
                if self.last_scaling_action else self.cooldown_period
            )),
            "metrics": {
                name: {
                    "current": metric.current_value,
                    "threshold_up": metric.threshold_up,
                    "threshold_down": metric.threshold_down,
                    "weight": metric.weight,
                    "enabled": metric.enabled
                }
                for name, metric in self.scaling_metrics.items()
            },
            "resources": self.resource_manager.get_resource_status(),
            "recent_actions": [
                {
                    "direction": action.direction.value,
                    "magnitude": action.magnitude,
                    "reason": action.reason,
                    "timestamp": action.timestamp.isoformat()
                }
                for action in list(self.scaling_history)[-10:]
            ]
        }


class LoadBalancer:
    """Intelligent load balancer for distributed processing."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.logger = get_logger(__name__)
        self.nodes = {}
        self.load_balancing_strategy = "least_connections"
        self.health_check_interval = 30.0
        self.unhealthy_threshold = 3
    
    def register_node(self, node_id: str, endpoint: str, capacity: float = 1.0):
        """Register a processing node."""
        self.nodes[node_id] = {
            "endpoint": endpoint,
            "capacity": capacity,
            "current_load": 0.0,
            "connections": 0,
            "health_status": "healthy",
            "failed_checks": 0,
            "last_used": datetime.now()
        }
        
        self.logger.log_info(f"Registered node {node_id} with capacity {capacity}")
    
    def select_node(self, request_weight: float = 1.0) -> Optional[str]:
        """Select best node for request."""
        healthy_nodes = [
            (node_id, node) for node_id, node in self.nodes.items()
            if node["health_status"] == "healthy"
        ]
        
        if not healthy_nodes:
            return None
        
        if self.load_balancing_strategy == "least_connections":
            selected = min(healthy_nodes, key=lambda x: x[1]["connections"])
        elif self.load_balancing_strategy == "least_load":
            selected = min(healthy_nodes, key=lambda x: x[1]["current_load"])
        elif self.load_balancing_strategy == "weighted_round_robin":
            selected = self._weighted_round_robin_selection(healthy_nodes)
        else:
            # Default to round robin
            selected = min(healthy_nodes, key=lambda x: x[1]["last_used"])
        
        # Update node state
        node_id, node = selected
        node["connections"] += 1
        node["current_load"] += request_weight
        node["last_used"] = datetime.now()
        
        return node_id
    
    def release_node(self, node_id: str, request_weight: float = 1.0):
        """Release node after request completion."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node["connections"] = max(0, node["connections"] - 1)
            node["current_load"] = max(0, node["current_load"] - request_weight)
    
    def _weighted_round_robin_selection(self, healthy_nodes: List[Tuple[str, Dict]]) -> Tuple[str, Dict]:
        """Weighted round robin node selection."""
        # Simple implementation - in practice would be more sophisticated
        total_capacity = sum(node["capacity"] for _, node in healthy_nodes)
        
        # Find node with lowest load relative to capacity
        return min(healthy_nodes, key=lambda x: x[1]["current_load"] / x[1]["capacity"])
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        return {
            "strategy": self.load_balancing_strategy,
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n["health_status"] == "healthy"]),
            "total_connections": sum(n["connections"] for n in self.nodes.values()),
            "total_load": sum(n["current_load"] for n in self.nodes.values()),
            "nodes": {
                node_id: {
                    "capacity": node["capacity"],
                    "current_load": node["current_load"],
                    "connections": node["connections"],
                    "health_status": node["health_status"],
                    "utilization": node["current_load"] / node["capacity"] if node["capacity"] > 0 else 0
                }
                for node_id, node in self.nodes.items()
            }
        }


# Global instances
global_auto_scaler = AutoScaler()
global_load_balancer = LoadBalancer()