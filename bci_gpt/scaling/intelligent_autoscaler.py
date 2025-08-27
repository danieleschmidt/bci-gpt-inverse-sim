#!/usr/bin/env python3
"""
Intelligent Auto-Scaling System for BCI-GPT
Generation 3: Adaptive scaling based on load, performance, and resource utilization
"""

import time
import json
import logging
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import subprocess
import asyncio

class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ResourceType(Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"

@dataclass
class ScalingMetric:
    """Metric for scaling decisions."""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    history: List[float] = field(default_factory=list)
    
    def add_value(self, value: float):
        """Add new metric value."""
        self.current_value = value
        self.history.append(value)
        
        # Keep last 100 values
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_trend(self) -> str:
        """Get metric trend."""
        if len(self.history) < 5:
            return "insufficient_data"
        
        recent = self.history[-5:]
        older = self.history[-10:-5] if len(self.history) >= 10 else self.history[:-5]
        
        if older:
            recent_avg = statistics.mean(recent)
            older_avg = statistics.mean(older)
            
            if recent_avg > older_avg * 1.1:
                return "increasing"
            elif recent_avg < older_avg * 0.9:
                return "decreasing"
        
        return "stable"
    
    def should_scale_up(self) -> bool:
        """Check if metric suggests scaling up."""
        return self.current_value > self.threshold_up
    
    def should_scale_down(self) -> bool:
        """Check if metric suggests scaling down."""
        return self.current_value < self.threshold_down

@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    direction: ScalingDirection
    reason: str
    metrics: Dict[str, float]
    previous_replicas: int
    new_replicas: int
    success: bool = False
    error_message: Optional[str] = None

class ResourceMonitor:
    """Monitor system resources for scaling decisions."""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": ScalingMetric("cpu_usage", 0.0, 0.8, 0.3, weight=1.0),
            "memory_usage": ScalingMetric("memory_usage", 0.0, 0.8, 0.3, weight=1.0),
            "gpu_usage": ScalingMetric("gpu_usage", 0.0, 0.8, 0.2, weight=1.2),
            "request_latency": ScalingMetric("request_latency", 0.0, 200.0, 50.0, weight=1.5),
            "queue_length": ScalingMetric("queue_length", 0.0, 10.0, 2.0, weight=1.3),
            "error_rate": ScalingMetric("error_rate", 0.0, 0.05, 0.01, weight=2.0),
            "throughput": ScalingMetric("throughput", 0.0, 1000.0, 100.0, weight=0.8)
        }
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval: int = 30):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Resource monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[Dict[str, ScalingMetric]], None]):
        """Add callback for metric updates."""
        self.callbacks.append(callback)
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._collect_application_metrics()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.metrics.copy())
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Back off on errors
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage (mock - would use psutil in production)
            import os
            cpu_usage = min(1.0, len(os.listdir('/proc')) / 1000.0)  # Rough estimate
            self.metrics["cpu_usage"].add_value(cpu_usage)
            
            # Memory usage (mock)
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                total_memory = None
                free_memory = None
                
                for line in lines:
                    if line.startswith('MemTotal:'):
                        total_memory = int(line.split()[1]) * 1024  # Convert to bytes
                    elif line.startswith('MemAvailable:'):
                        free_memory = int(line.split()[1]) * 1024
                
                if total_memory and free_memory:
                    memory_usage = 1.0 - (free_memory / total_memory)
                    self.metrics["memory_usage"].add_value(memory_usage)
            
            # GPU usage (mock)
            gpu_usage = 0.3  # Would use nvidia-ml-py in production
            self.metrics["gpu_usage"].add_value(gpu_usage)
            
        except Exception as e:
            self.logger.warning(f"System metrics collection failed: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Mock application metrics
            import random
            
            # Request latency (ms)
            latency = random.uniform(20, 150)
            self.metrics["request_latency"].add_value(latency)
            
            # Queue length
            queue_length = random.randint(0, 15)
            self.metrics["queue_length"].add_value(queue_length)
            
            # Error rate
            error_rate = random.uniform(0, 0.1)
            self.metrics["error_rate"].add_value(error_rate)
            
            # Throughput (requests/minute)
            throughput = random.uniform(50, 800)
            self.metrics["throughput"].add_value(throughput)
            
        except Exception as e:
            self.logger.warning(f"Application metrics collection failed: {e}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {name: metric.current_value for name, metric in self.metrics.items()}
    
    def get_scaling_recommendation(self) -> Tuple[ScalingDirection, float, str]:
        """Get scaling recommendation based on current metrics."""
        scale_up_score = 0.0
        scale_down_score = 0.0
        reasons = []
        
        for name, metric in self.metrics.items():
            if metric.should_scale_up():
                scale_up_score += metric.weight
                reasons.append(f"{name} high ({metric.current_value:.2f})")
            elif metric.should_scale_down():
                scale_down_score += metric.weight
                reasons.append(f"{name} low ({metric.current_value:.2f})")
        
        if scale_up_score > scale_down_score and scale_up_score > 2.0:
            return ScalingDirection.UP, scale_up_score, "; ".join(reasons)
        elif scale_down_score > scale_up_score and scale_down_score > 2.0:
            return ScalingDirection.DOWN, scale_down_score, "; ".join(reasons)
        else:
            return ScalingDirection.STABLE, 0.0, "metrics within normal ranges"

class AutoScaler:
    """Intelligent auto-scaling controller."""
    
    def __init__(self,
                 min_replicas: int = 1,
                 max_replicas: int = 10,
                 target_replicas: int = 3,
                 scale_up_cooldown: int = 300,  # 5 minutes
                 scale_down_cooldown: int = 600):  # 10 minutes
        
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = target_replicas
        self.target_replicas = target_replicas
        
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        
        self.scaling_events = []
        self.resource_monitor = ResourceMonitor()
        self.scaling_callbacks = []
        
        # Scaling configuration
        self.scaling_policies = {
            "aggressive": {"scale_factor": 2, "threshold_multiplier": 0.8},
            "conservative": {"scale_factor": 1, "threshold_multiplier": 1.2},
            "balanced": {"scale_factor": 1, "threshold_multiplier": 1.0}
        }
        self.current_policy = "balanced"
        
        self.logger = logging.getLogger(__name__)
    
    def start_autoscaling(self, monitoring_interval: int = 30):
        """Start auto-scaling system."""
        self.resource_monitor.add_callback(self._evaluate_scaling)
        self.resource_monitor.start_monitoring(monitoring_interval)
        
        self.logger.info(f"Auto-scaling started: {self.current_replicas} replicas")
    
    def stop_autoscaling(self):
        """Stop auto-scaling system."""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Auto-scaling stopped")
    
    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]):
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
    
    def _evaluate_scaling(self, metrics: Dict[str, ScalingMetric]):
        """Evaluate if scaling is needed."""
        try:
            direction, confidence, reason = self.resource_monitor.get_scaling_recommendation()
            
            if direction == ScalingDirection.UP:
                self._attempt_scale_up(reason, confidence, metrics)
            elif direction == ScalingDirection.DOWN:
                self._attempt_scale_down(reason, confidence, metrics)
            
        except Exception as e:
            self.logger.error(f"Scaling evaluation error: {e}")
    
    def _attempt_scale_up(self, reason: str, confidence: float, metrics: Dict[str, ScalingMetric]):
        """Attempt to scale up."""
        now = datetime.now()
        
        # Check cooldown
        if (now - self.last_scale_up).seconds < self.scale_up_cooldown:
            return
        
        # Check max replicas
        if self.current_replicas >= self.max_replicas:
            self.logger.warning(f"Cannot scale up: at max replicas ({self.max_replicas})")
            return
        
        # Calculate new replica count
        policy = self.scaling_policies[self.current_policy]
        scale_factor = policy["scale_factor"]
        new_replicas = min(self.current_replicas + scale_factor, self.max_replicas)
        
        # Execute scaling
        success = self._execute_scaling(new_replicas)
        
        # Record event
        event = ScalingEvent(
            timestamp=now,
            direction=ScalingDirection.UP,
            reason=reason,
            metrics={name: m.current_value for name, m in metrics.items()},
            previous_replicas=self.current_replicas,
            new_replicas=new_replicas,
            success=success
        )
        
        if success:
            self.current_replicas = new_replicas
            self.last_scale_up = now
            self.logger.info(f"Scaled up: {event.previous_replicas} → {new_replicas} ({reason})")
        else:
            self.logger.error(f"Scale up failed: {reason}")
        
        self.scaling_events.append(event)
        self._notify_scaling_callbacks(event)
    
    def _attempt_scale_down(self, reason: str, confidence: float, metrics: Dict[str, ScalingMetric]):
        """Attempt to scale down."""
        now = datetime.now()
        
        # Check cooldown
        if (now - self.last_scale_down).seconds < self.scale_down_cooldown:
            return
        
        # Check min replicas
        if self.current_replicas <= self.min_replicas:
            return
        
        # Calculate new replica count
        policy = self.scaling_policies[self.current_policy]
        scale_factor = policy["scale_factor"]
        new_replicas = max(self.current_replicas - scale_factor, self.min_replicas)
        
        # Execute scaling
        success = self._execute_scaling(new_replicas)
        
        # Record event
        event = ScalingEvent(
            timestamp=now,
            direction=ScalingDirection.DOWN,
            reason=reason,
            metrics={name: m.current_value for name, m in metrics.items()},
            previous_replicas=self.current_replicas,
            new_replicas=new_replicas,
            success=success
        )
        
        if success:
            self.current_replicas = new_replicas
            self.last_scale_down = now
            self.logger.info(f"Scaled down: {event.previous_replicas} → {new_replicas} ({reason})")
        else:
            self.logger.error(f"Scale down failed: {reason}")
        
        self.scaling_events.append(event)
        self._notify_scaling_callbacks(event)
    
    def _execute_scaling(self, target_replicas: int) -> bool:
        """Execute scaling operation."""
        try:
            # Mock scaling execution - would use Kubernetes API, Docker Swarm, etc.
            self.logger.info(f"Executing scaling to {target_replicas} replicas")
            
            # Simulate scaling delay
            time.sleep(1)
            
            # In production, would execute:
            # kubectl scale deployment/bci-gpt --replicas={target_replicas}
            # or use Kubernetes API, Docker API, etc.
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
            return False
    
    def _notify_scaling_callbacks(self, event: ScalingEvent):
        """Notify all scaling callbacks."""
        for callback in self.scaling_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Scaling callback error: {e}")
    
    def manual_scale(self, target_replicas: int, reason: str = "manual") -> bool:
        """Manually scale to target replicas."""
        if not (self.min_replicas <= target_replicas <= self.max_replicas):
            self.logger.error(f"Invalid replica count: {target_replicas}")
            return False
        
        success = self._execute_scaling(target_replicas)
        
        if success:
            event = ScalingEvent(
                timestamp=datetime.now(),
                direction=ScalingDirection.UP if target_replicas > self.current_replicas else ScalingDirection.DOWN,
                reason=reason,
                metrics=self.resource_monitor.get_current_metrics(),
                previous_replicas=self.current_replicas,
                new_replicas=target_replicas,
                success=True
            )
            
            self.current_replicas = target_replicas
            self.scaling_events.append(event)
            self._notify_scaling_callbacks(event)
            
            self.logger.info(f"Manual scaling successful: {target_replicas} replicas")
        
        return success
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "current_replicas": self.current_replicas,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "policy": self.current_policy,
            "last_scale_up": self.last_scale_up.isoformat() if self.last_scale_up != datetime.min else None,
            "last_scale_down": self.last_scale_down.isoformat() if self.last_scale_down != datetime.min else None,
            "total_scaling_events": len(self.scaling_events),
            "current_metrics": self.resource_monitor.get_current_metrics()
        }
    
    def set_policy(self, policy: str):
        """Set scaling policy."""
        if policy in self.scaling_policies:
            self.current_policy = policy
            self.logger.info(f"Scaling policy changed to: {policy}")
        else:
            self.logger.error(f"Unknown scaling policy: {policy}")

# Example usage and testing
if __name__ == "__main__":
    print("⚡ Testing Auto-Scaling System...")
    
    # Test resource monitoring
    monitor = ResourceMonitor()
    
    def metric_callback(metrics):
        direction, confidence, reason = monitor.get_scaling_recommendation()
        if direction != ScalingDirection.STABLE:
            print(f"📊 Scaling recommendation: {direction.value} (confidence: {confidence:.1f}) - {reason}")
    
    monitor.add_callback(metric_callback)
    monitor.start_monitoring(5)  # 5-second intervals for testing
    
    # Test auto-scaler
    autoscaler = AutoScaler(
        min_replicas=1,
        max_replicas=5,
        target_replicas=2,
        scale_up_cooldown=10,  # Short cooldown for testing
        scale_down_cooldown=20
    )
    
    def scaling_callback(event):
        print(f"🔄 Scaling event: {event.direction.value} from {event.previous_replicas} to {event.new_replicas}")
        print(f"   Reason: {event.reason}")
    
    autoscaler.add_scaling_callback(scaling_callback)
    autoscaler.start_autoscaling(5)
    
    # Test manual scaling
    print("🔧 Testing manual scaling...")
    success = autoscaler.manual_scale(4, "testing_manual_scale")
    print(f"✅ Manual scaling result: {success}")
    
    # Let it run for a bit
    print("⏳ Monitoring for 30 seconds...")
    time.sleep(30)
    
    # Get status
    status = autoscaler.get_status()
    print(f"📊 Final status: {status['current_replicas']} replicas, {status['total_scaling_events']} events")
    
    # Cleanup
    autoscaler.stop_autoscaling()
    
    print("\n🚀 Auto-Scaling System Ready!")
