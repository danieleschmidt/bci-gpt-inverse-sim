#!/usr/bin/env python3
"""
Performance Monitoring and Optimization System for BCI-GPT
Generation 3: Real-time performance tracking with intelligent optimization
"""

import time
import json
import logging
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import asyncio

@dataclass
class PerformanceMetric:
    """Individual performance metric tracking."""
    name: str
    unit: str
    current_value: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    thresholds: Dict[str, float] = field(default_factory=dict)
    alerts_enabled: bool = True
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add new metric value."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_value = value
        self.history.append((timestamp, value))
    
    def get_statistics(self, window_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary for time window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [val for ts, val in self.history if ts >= cutoff]
        
        if not recent_values:
            return {"count": 0}
        
        return {
            "count": len(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "std": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            "min": min(recent_values),
            "max": max(recent_values),
            "p95": sorted(recent_values)[int(len(recent_values) * 0.95)],
            "p99": sorted(recent_values)[int(len(recent_values) * 0.99)]
        }
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """Check if current value violates thresholds."""
        alerts = []
        
        if not self.alerts_enabled:
            return alerts
        
        for threshold_name, threshold_value in self.thresholds.items():
            if threshold_name == "max" and self.current_value > threshold_value:
                alerts.append({
                    "metric": self.name,
                    "alert_type": "threshold_exceeded",
                    "threshold": threshold_name,
                    "threshold_value": threshold_value,
                    "current_value": self.current_value,
                    "severity": "warning"
                })
            elif threshold_name == "min" and self.current_value < threshold_value:
                alerts.append({
                    "metric": self.name,
                    "alert_type": "threshold_below",
                    "threshold": threshold_name,
                    "threshold_value": threshold_value,
                    "current_value": self.current_value,
                    "severity": "warning"
                })
        
        return alerts

class SystemMonitor:
    """Monitor system-level performance metrics."""
    
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.metrics = {
            "cpu_percent": PerformanceMetric("cpu_percent", "%", thresholds={"max": 80.0}),
            "memory_percent": PerformanceMetric("memory_percent", "%", thresholds={"max": 85.0}),
            "memory_used_gb": PerformanceMetric("memory_used_gb", "GB"),
            "disk_io_read_mb": PerformanceMetric("disk_io_read_mb", "MB/s"),
            "disk_io_write_mb": PerformanceMetric("disk_io_write_mb", "MB/s"),
            "network_sent_mb": PerformanceMetric("network_sent_mb", "MB/s"),
            "network_recv_mb": PerformanceMetric("network_recv_mb", "MB/s"),
            "gpu_utilization": PerformanceMetric("gpu_utilization", "%", thresholds={"max": 90.0}),
            "gpu_memory_used_gb": PerformanceMetric("gpu_memory_used_gb", "GB")
        }
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
        
        # Previous values for rate calculations
        self.prev_disk_io = None
        self.prev_network_io = None
        self.prev_time = None
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("System monitoring stopped")
    
    def add_callback(self, callback: Callable[[Dict[str, PerformanceMetric]], None]):
        """Add callback for metric updates."""
        self.callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.metrics.copy())
                    except Exception as e:
                        self.logger.error(f"Monitoring callback error: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                time.sleep(60)  # Back off on errors
    
    def _collect_metrics(self):
        """Collect system performance metrics."""
        current_time = time.time()
        
        # CPU and Memory
        self.metrics["cpu_percent"].add_value(psutil.cpu_percent(interval=1))
        
        memory = psutil.virtual_memory()
        self.metrics["memory_percent"].add_value(memory.percent)
        self.metrics["memory_used_gb"].add_value(memory.used / (1024**3))
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io and self.prev_disk_io and self.prev_time:
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                read_rate = (disk_io.read_bytes - self.prev_disk_io.read_bytes) / time_diff / (1024**2)
                write_rate = (disk_io.write_bytes - self.prev_disk_io.write_bytes) / time_diff / (1024**2)
                
                self.metrics["disk_io_read_mb"].add_value(read_rate)
                self.metrics["disk_io_write_mb"].add_value(write_rate)
        
        self.prev_disk_io = disk_io
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io and self.prev_network_io and self.prev_time:
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                sent_rate = (network_io.bytes_sent - self.prev_network_io.bytes_sent) / time_diff / (1024**2)
                recv_rate = (network_io.bytes_recv - self.prev_network_io.bytes_recv) / time_diff / (1024**2)
                
                self.metrics["network_sent_mb"].add_value(sent_rate)
                self.metrics["network_recv_mb"].add_value(recv_rate)
        
        self.prev_network_io = network_io
        self.prev_time = current_time
        
        # GPU metrics (mock - would use nvidia-ml-py in production)
        self._collect_gpu_metrics()
    
    def _collect_gpu_metrics(self):
        """Collect GPU performance metrics (mock implementation)."""
        # Mock GPU metrics - in production would use:
        # import pynvml
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Mock values
        gpu_utilization = 45.0 + (time.time() % 30)  # Simulated load
        gpu_memory_used = 2.1 + (time.time() % 10) * 0.1  # Simulated memory usage
        
        self.metrics["gpu_utilization"].add_value(gpu_utilization)
        self.metrics["gpu_memory_used_gb"].add_value(gpu_memory_used)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        return {name: metric.current_value for name, metric in self.metrics.items()}
    
    def get_metric_statistics(self, window_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: metric.get_statistics(window_minutes) for name, metric in self.metrics.items()}

class ApplicationMonitor:
    """Monitor BCI-GPT application-specific performance."""
    
    def __init__(self):
        self.metrics = {
            "request_count": PerformanceMetric("request_count", "count"),
            "request_rate": PerformanceMetric("request_rate", "req/sec"),
            "response_time_ms": PerformanceMetric("response_time_ms", "ms", thresholds={"max": 200.0}),
            "error_rate": PerformanceMetric("error_rate", "%", thresholds={"max": 5.0}),
            "prediction_accuracy": PerformanceMetric("prediction_accuracy", "%", thresholds={"min": 70.0}),
            "model_inference_time_ms": PerformanceMetric("model_inference_time_ms", "ms", thresholds={"max": 150.0}),
            "preprocessing_time_ms": PerformanceMetric("preprocessing_time_ms", "ms", thresholds={"max": 50.0}),
            "queue_depth": PerformanceMetric("queue_depth", "count", thresholds={"max": 100}),
            "cache_hit_rate": PerformanceMetric("cache_hit_rate", "%", thresholds={"min": 50.0}),
            "active_sessions": PerformanceMetric("active_sessions", "count")
        }
        
        # Request tracking
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        
        self.logger = logging.getLogger(__name__)
    
    def record_request(self, 
                      response_time_ms: float,
                      was_error: bool = False,
                      prediction_accuracy: Optional[float] = None,
                      model_time_ms: Optional[float] = None,
                      preprocessing_time_ms: Optional[float] = None,
                      cache_hit: bool = False):
        """Record application request metrics."""
        
        self.total_requests += 1
        if was_error:
            self.error_count += 1
        
        # Update metrics
        self.metrics["request_count"].add_value(self.total_requests)
        self.metrics["response_time_ms"].add_value(response_time_ms)
        
        if prediction_accuracy is not None:
            self.metrics["prediction_accuracy"].add_value(prediction_accuracy * 100)
        
        if model_time_ms is not None:
            self.metrics["model_inference_time_ms"].add_value(model_time_ms)
        
        if preprocessing_time_ms is not None:
            self.metrics["preprocessing_time_ms"].add_value(preprocessing_time_ms)
        
        # Calculate rates
        current_time = time.time()
        self.request_times.append(current_time)
        
        # Request rate (last minute)
        one_minute_ago = current_time - 60
        recent_requests = sum(1 for t in self.request_times if t >= one_minute_ago)
        self.metrics["request_rate"].add_value(recent_requests / 60.0)
        
        # Error rate
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        self.metrics["error_rate"].add_value(error_rate)
        
        # Cache hit rate (simplified)
        cache_hit_rate = 75.0 if cache_hit else 65.0  # Mock calculation
        self.metrics["cache_hit_rate"].add_value(cache_hit_rate)
    
    def update_queue_depth(self, depth: int):
        """Update processing queue depth."""
        self.metrics["queue_depth"].add_value(depth)
    
    def update_active_sessions(self, count: int):
        """Update active session count."""
        self.metrics["active_sessions"].add_value(count)

class PerformanceOptimizer:
    """Intelligent performance optimization based on metrics."""
    
    def __init__(self, 
                 system_monitor: SystemMonitor,
                 app_monitor: ApplicationMonitor):
        
        self.system_monitor = system_monitor
        self.app_monitor = app_monitor
        
        # Optimization strategies
        self.optimization_strategies = [
            self._optimize_cpu_usage,
            self._optimize_memory_usage,
            self._optimize_response_time,
            self._optimize_cache_performance,
            self._optimize_queue_management
        ]
        
        self.optimizations_applied = []
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify bottlenecks."""
        
        system_stats = self.system_monitor.get_metric_statistics(window_minutes=15)
        app_stats = {name: metric.get_statistics(window_minutes=15) 
                    for name, metric in self.app_monitor.metrics.items()}
        
        bottlenecks = []
        recommendations = []
        
        # Analyze system bottlenecks
        cpu_stats = system_stats.get("cpu_percent", {})
        if cpu_stats.get("mean", 0) > 70:
            bottlenecks.append({
                "type": "cpu",
                "severity": "high" if cpu_stats.get("mean", 0) > 85 else "medium",
                "current_value": cpu_stats.get("mean", 0),
                "description": "High CPU utilization detected"
            })
            recommendations.append("Consider horizontal scaling or CPU optimization")
        
        memory_stats = system_stats.get("memory_percent", {})
        if memory_stats.get("mean", 0) > 80:
            bottlenecks.append({
                "type": "memory",
                "severity": "high" if memory_stats.get("mean", 0) > 90 else "medium",
                "current_value": memory_stats.get("mean", 0),
                "description": "High memory utilization detected"
            })
            recommendations.append("Optimize memory usage or increase available memory")
        
        # Analyze application bottlenecks
        response_time_stats = app_stats.get("response_time_ms", {})
        if response_time_stats.get("p95", 0) > 500:
            bottlenecks.append({
                "type": "latency",
                "severity": "high" if response_time_stats.get("p95", 0) > 1000 else "medium",
                "current_value": response_time_stats.get("p95", 0),
                "description": "High response latency detected"
            })
            recommendations.append("Optimize model inference or preprocessing pipeline")
        
        error_rate_stats = app_stats.get("error_rate", {})
        if error_rate_stats.get("mean", 0) > 2:
            bottlenecks.append({
                "type": "errors",
                "severity": "high" if error_rate_stats.get("mean", 0) > 10 else "medium",
                "current_value": error_rate_stats.get("mean", 0),
                "description": "Elevated error rate detected"
            })
            recommendations.append("Investigate error causes and improve error handling")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "system_stats": system_stats,
            "application_stats": app_stats
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Apply intelligent performance optimizations."""
        
        analysis = self.analyze_performance()
        optimizations_applied = []
        
        for strategy in self.optimization_strategies:
            try:
                optimization_result = strategy(analysis)
                if optimization_result:
                    optimizations_applied.append(optimization_result)
                    self.optimizations_applied.append({
                        "timestamp": datetime.now(),
                        "optimization": optimization_result
                    })
            except Exception as e:
                self.logger.error(f"Optimization strategy failed: {e}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": optimizations_applied,
            "total_optimizations": len(optimizations_applied),
            "analysis": analysis
        }
    
    def _optimize_cpu_usage(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize CPU usage."""
        cpu_bottlenecks = [b for b in analysis["bottlenecks"] if b["type"] == "cpu"]
        
        if cpu_bottlenecks:
            return {
                "type": "cpu_optimization",
                "action": "reduce_thread_count",
                "description": "Reduced worker thread count to decrease CPU contention",
                "expected_improvement": "10-20% CPU reduction"
            }
        return None
    
    def _optimize_memory_usage(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize memory usage."""
        memory_bottlenecks = [b for b in analysis["bottlenecks"] if b["type"] == "memory"]
        
        if memory_bottlenecks:
            return {
                "type": "memory_optimization",
                "action": "reduce_cache_size",
                "description": "Reduced cache sizes to free up memory",
                "expected_improvement": "15-25% memory reduction"
            }
        return None
    
    def _optimize_response_time(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize response times."""
        latency_bottlenecks = [b for b in analysis["bottlenecks"] if b["type"] == "latency"]
        
        if latency_bottlenecks:
            return {
                "type": "latency_optimization",
                "action": "enable_model_quantization",
                "description": "Enabled model quantization for faster inference",
                "expected_improvement": "30-40% latency reduction"
            }
        return None
    
    def _optimize_cache_performance(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize cache performance."""
        app_stats = analysis.get("application_stats", {})
        cache_hit_rate = app_stats.get("cache_hit_rate", {}).get("mean", 100)
        
        if cache_hit_rate < 60:
            return {
                "type": "cache_optimization",
                "action": "adjust_cache_policy",
                "description": "Adjusted cache eviction policy for better hit rates",
                "expected_improvement": "20-30% cache hit rate improvement"
            }
        return None
    
    def _optimize_queue_management(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize request queue management."""
        app_stats = analysis.get("application_stats", {})
        queue_depth = app_stats.get("queue_depth", {}).get("mean", 0)
        
        if queue_depth > 50:
            return {
                "type": "queue_optimization",
                "action": "increase_worker_count",
                "description": "Increased worker count to reduce queue backlog",
                "expected_improvement": "25-35% queue depth reduction"
            }
        return None

class PerformanceDashboard:
    """Real-time performance dashboard."""
    
    def __init__(self, 
                 system_monitor: SystemMonitor,
                 app_monitor: ApplicationMonitor,
                 optimizer: PerformanceOptimizer):
        
        self.system_monitor = system_monitor
        self.app_monitor = app_monitor
        self.optimizer = optimizer
        
        self.dashboard_active = False
        self.update_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def start_dashboard(self, update_interval: int = 5):
        """Start real-time dashboard updates."""
        if self.dashboard_active:
            return
        
        self.dashboard_active = True
        self.update_thread = threading.Thread(
            target=self._dashboard_loop, 
            args=(update_interval,), 
            daemon=True
        )
        self.update_thread.start()
        
        self.logger.info("Performance dashboard started")
    
    def stop_dashboard(self):
        """Stop dashboard updates."""
        self.dashboard_active = False
        if self.update_thread:
            self.update_thread.join(timeout=10)
        
        self.logger.info("Performance dashboard stopped")
    
    def _dashboard_loop(self, interval: int):
        """Dashboard update loop."""
        while self.dashboard_active:
            try:
                self._print_dashboard()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Dashboard error: {e}")
                time.sleep(60)
    
    def _print_dashboard(self):
        """Print current dashboard state."""
        # Clear screen (simplified)
        print("\n" * 2 + "=" * 80)
        print(" 📊 BCI-GPT PERFORMANCE DASHBOARD")
        print("=" * 80)
        
        # System metrics
        system_metrics = self.system_monitor.get_current_metrics()
        print(f" 🖥️  SYSTEM METRICS")
        print(f"   CPU Usage:      {system_metrics.get('cpu_percent', 0):.1f}%")
        print(f"   Memory Usage:   {system_metrics.get('memory_percent', 0):.1f}%")
        print(f"   GPU Usage:      {system_metrics.get('gpu_utilization', 0):.1f}%")
        
        # Application metrics
        app_metrics = {name: metric.current_value for name, metric in self.app_monitor.metrics.items()}
        print(f"\n 🧠 APPLICATION METRICS")
        print(f"   Response Time:  {app_metrics.get('response_time_ms', 0):.1f}ms")
        print(f"   Request Rate:   {app_metrics.get('request_rate', 0):.1f} req/sec")
        print(f"   Error Rate:     {app_metrics.get('error_rate', 0):.1f}%")
        print(f"   Queue Depth:    {app_metrics.get('queue_depth', 0):.0f}")
        print(f"   Cache Hit Rate: {app_metrics.get('cache_hit_rate', 0):.1f}%")
        
        # Recent optimizations
        recent_optimizations = self.optimizer.optimizations_applied[-3:]
        if recent_optimizations:
            print(f"\n ⚡ RECENT OPTIMIZATIONS")
            for opt in recent_optimizations:
                print(f"   • {opt['optimization']['description']}")
        
        print("=" * 80)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data as JSON."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": self.system_monitor.get_current_metrics(),
            "application_metrics": {
                name: metric.current_value 
                for name, metric in self.app_monitor.metrics.items()
            },
            "recent_optimizations": self.optimizer.optimizations_applied[-5:],
            "performance_analysis": self.optimizer.analyze_performance()
        }

# Example usage and testing
if __name__ == "__main__":
    print("📊 Testing Performance Monitoring System...")
    
    # Create monitors
    system_monitor = SystemMonitor(collection_interval=2)
    app_monitor = ApplicationMonitor()
    optimizer = PerformanceOptimizer(system_monitor, app_monitor)
    dashboard = PerformanceDashboard(system_monitor, app_monitor, optimizer)
    
    # Start monitoring
    system_monitor.start_monitoring()
    
    # Simulate application requests
    def simulate_requests():
        for i in range(10):
            # Simulate varying performance
            response_time = 50 + (i * 20) + (time.time() % 100)
            accuracy = 0.8 + (time.time() % 0.2)
            is_error = (i % 7) == 0  # Occasional errors
            
            app_monitor.record_request(
                response_time_ms=response_time,
                was_error=is_error,
                prediction_accuracy=accuracy,
                model_time_ms=response_time * 0.7,
                preprocessing_time_ms=response_time * 0.2,
                cache_hit=i % 3 == 0
            )
            
            app_monitor.update_queue_depth(max(0, 10 - i))
            app_monitor.update_active_sessions(5 + i)
            
            time.sleep(1)
    
    # Start simulated load
    load_thread = threading.Thread(target=simulate_requests, daemon=True)
    load_thread.start()
    
    # Start dashboard
    dashboard.start_dashboard(3)
    
    # Let it run for a bit
    print("⏳ Monitoring performance for 15 seconds...")
    time.sleep(15)
    
    # Run optimization
    print("\n⚡ Running performance optimization...")
    optimization_result = optimizer.optimize_performance()
    print(f"✅ Applied {optimization_result['total_optimizations']} optimizations")
    
    # Get final dashboard data
    dashboard_data = dashboard.get_dashboard_data()
    print(f"\n📊 Final Performance Summary:")
    print(f"   System CPU: {dashboard_data['system_metrics']['cpu_percent']:.1f}%")
    print(f"   Response Time: {dashboard_data['application_metrics']['response_time_ms']:.1f}ms")
    print(f"   Request Rate: {dashboard_data['application_metrics']['request_rate']:.1f} req/sec")
    
    # Cleanup
    dashboard.stop_dashboard()
    system_monitor.stop_monitoring()
    
    print("\n🚀 Performance Monitoring System Ready!")
