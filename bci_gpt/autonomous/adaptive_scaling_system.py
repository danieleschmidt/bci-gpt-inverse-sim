"""
Adaptive Performance Optimization System v4.0
Intelligent resource management, auto-scaling, and performance optimization.
"""

import asyncio
import json
import logging
import psutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import resource

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    load_average: List[float] = field(default_factory=list)
    process_count: int = 0


@dataclass
class PerformanceProfile:
    """Performance optimization profile."""
    name: str
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 85.0
    max_workers: int = 4
    cache_size_mb: int = 512
    batch_size: int = 32
    optimization_level: OptimizationStrategy = OptimizationStrategy.BALANCED


class AdaptiveScalingSystem:
    """
    Intelligent adaptive scaling system that optimizes performance
    based on real-time resource usage and workload patterns.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("adaptive_scaling_config.json")
        self.metrics_path = Path("quality_reports/performance_metrics.json")
        self.metrics_path.parent.mkdir(exist_ok=True)
        
        self.current_profile = self._get_default_profile()
        self.metrics_history: List[ResourceMetrics] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance caches
        self.result_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            "cpu_threshold": 75.0,
            "memory_threshold": 80.0,
            "response_time_target": 100.0,  # ms
            "throughput_target": 1000.0     # ops/sec
        }
        
        self._load_config()
    
    def _get_default_profile(self) -> PerformanceProfile:
        """Get default performance profile based on system capabilities."""
        cpu_count = mp.cpu_count()
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        if cpu_count >= 8 and total_memory >= 16:
            return PerformanceProfile(
                name="high_performance",
                max_cpu_usage=85.0,
                max_memory_usage=80.0,
                max_workers=min(cpu_count, 8),
                cache_size_mb=1024,
                batch_size=64,
                optimization_level=OptimizationStrategy.AGGRESSIVE
            )
        elif cpu_count >= 4 and total_memory >= 8:
            return PerformanceProfile(
                name="balanced",
                max_cpu_usage=75.0,
                max_memory_usage=75.0,
                max_workers=min(cpu_count, 4),
                cache_size_mb=512,
                batch_size=32,
                optimization_level=OptimizationStrategy.BALANCED
            )
        else:
            return PerformanceProfile(
                name="conservative",
                max_cpu_usage=65.0,
                max_memory_usage=70.0,
                max_workers=2,
                cache_size_mb=256,
                batch_size=16,
                optimization_level=OptimizationStrategy.CONSERVATIVE
            )
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)
                    
                    # Update adaptive thresholds
                    if "adaptive_thresholds" in config:
                        self.adaptive_thresholds.update(config["adaptive_thresholds"])
                    
                    # Load performance profile
                    if "current_profile" in config:
                        profile_data = config["current_profile"]
                        self.current_profile = PerformanceProfile(**profile_data)
                        
            except Exception as e:
                logger.warning(f"Failed to load adaptive scaling config: {e}")
    
    def save_config(self):
        """Save current configuration."""
        config = {
            "adaptive_thresholds": self.adaptive_thresholds,
            "current_profile": {
                "name": self.current_profile.name,
                "max_cpu_usage": self.current_profile.max_cpu_usage,
                "max_memory_usage": self.current_profile.max_memory_usage,
                "max_workers": self.current_profile.max_workers,
                "cache_size_mb": self.current_profile.cache_size_mb,
                "batch_size": self.current_profile.batch_size,
                "optimization_level": self.current_profile.optimization_level.value
            },
            "last_updated": time.time()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
            
            # Load average (Unix/Linux only)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [cpu_percent / 100.0]  # Fallback for Windows
            
            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage=disk.percent,
                network_io=network_io,
                load_average=load_avg,
                process_count=len(psutil.pids())
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ResourceMetrics()  # Return empty metrics
    
    def analyze_performance_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze performance trends from recent metrics."""
        if len(self.metrics_history) < window_size:
            return {"trend": "insufficient_data", "recommendation": "continue_monitoring"}
        
        recent_metrics = self.metrics_history[-window_size:]
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        
        # Average resource usage
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Performance assessment
        performance_score = self._calculate_performance_score(avg_cpu, avg_memory)
        
        return {
            "trend": {
                "cpu": cpu_trend,
                "memory": memory_trend
            },
            "averages": {
                "cpu": avg_cpu,
                "memory": avg_memory
            },
            "performance_score": performance_score,
            "recommendation": self._get_optimization_recommendation(
                avg_cpu, avg_memory, cpu_trend, memory_trend
            )
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression slope
        n = len(values)
        x_avg = (n - 1) / 2
        y_avg = sum(values) / n
        
        numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
        denominator = sum((i - x_avg) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_score(self, cpu: float, memory: float) -> float:
        """Calculate overall performance score (0-100)."""
        # Lower resource usage = higher performance score
        cpu_score = max(0, 100 - cpu)
        memory_score = max(0, 100 - memory)
        
        # Weighted average (CPU weighted more heavily)
        return (cpu_score * 0.6 + memory_score * 0.4)
    
    def _get_optimization_recommendation(self, cpu: float, memory: float, 
                                       cpu_trend: str, memory_trend: str) -> str:
        """Get optimization recommendation based on metrics."""
        if cpu > 90 or memory > 95:
            return "scale_down_immediately"
        elif cpu > 80 or memory > 85:
            return "scale_down"
        elif cpu < 30 and memory < 40 and cpu_trend == "stable":
            return "scale_up"
        elif cpu_trend == "increasing" and memory_trend == "increasing":
            return "prepare_scale_down"
        else:
            return "maintain_current"
    
    async def adaptive_optimization(self) -> Dict[str, Any]:
        """Perform adaptive optimization based on current metrics."""
        metrics = self.collect_metrics()
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 100 measurements)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Analyze trends
        analysis = self.analyze_performance_trend()
        
        # Apply optimizations based on analysis
        optimizations_applied = []
        
        if analysis["recommendation"] == "scale_down":
            optimizations_applied.extend(await self._scale_down())
        elif analysis["recommendation"] == "scale_up":
            optimizations_applied.extend(await self._scale_up())
        elif analysis["recommendation"] == "scale_down_immediately":
            optimizations_applied.extend(await self._emergency_scale_down())
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds(metrics, analysis)
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "metrics": {
                "cpu": metrics.cpu_percent,
                "memory": metrics.memory_percent,
                "disk": metrics.disk_usage
            },
            "analysis": analysis,
            "optimizations_applied": optimizations_applied,
            "performance_score": analysis["performance_score"]
        }
        
        self.optimization_history.append(optimization_record)
        
        # Save metrics
        await self._save_metrics()
        
        return optimization_record
    
    async def _scale_down(self) -> List[str]:
        """Scale down resource usage."""
        optimizations = []
        
        # Reduce worker count
        if self.current_profile.max_workers > 2:
            self.current_profile.max_workers = max(2, self.current_profile.max_workers - 1)
            optimizations.append(f"reduced_workers_to_{self.current_profile.max_workers}")
        
        # Reduce cache size
        if self.current_profile.cache_size_mb > 128:
            self.current_profile.cache_size_mb = max(128, int(self.current_profile.cache_size_mb * 0.8))
            optimizations.append(f"reduced_cache_to_{self.current_profile.cache_size_mb}mb")
        
        # Reduce batch size
        if self.current_profile.batch_size > 8:
            self.current_profile.batch_size = max(8, int(self.current_profile.batch_size * 0.8))
            optimizations.append(f"reduced_batch_size_to_{self.current_profile.batch_size}")
        
        # Clear some cache
        if len(self.result_cache) > 100:
            # Remove oldest entries
            cache_items = list(self.result_cache.items())
            self.result_cache = dict(cache_items[-50:])
            optimizations.append("cleared_cache")
        
        return optimizations
    
    async def _scale_up(self) -> List[str]:
        """Scale up resource usage for better performance."""
        optimizations = []
        max_workers = min(mp.cpu_count(), 8)
        
        # Increase worker count
        if self.current_profile.max_workers < max_workers:
            self.current_profile.max_workers = min(max_workers, self.current_profile.max_workers + 1)
            optimizations.append(f"increased_workers_to_{self.current_profile.max_workers}")
        
        # Increase cache size
        if self.current_profile.cache_size_mb < 1024:
            self.current_profile.cache_size_mb = min(1024, int(self.current_profile.cache_size_mb * 1.2))
            optimizations.append(f"increased_cache_to_{self.current_profile.cache_size_mb}mb")
        
        # Increase batch size
        if self.current_profile.batch_size < 64:
            self.current_profile.batch_size = min(64, int(self.current_profile.batch_size * 1.2))
            optimizations.append(f"increased_batch_size_to_{self.current_profile.batch_size}")
        
        return optimizations
    
    async def _emergency_scale_down(self) -> List[str]:
        """Emergency scaling when resources are critically high."""
        optimizations = []
        
        # Aggressive scaling
        self.current_profile.max_workers = 1
        self.current_profile.cache_size_mb = 64
        self.current_profile.batch_size = 4
        
        # Clear all cache
        self.result_cache.clear()
        
        optimizations.extend([
            "emergency_scale_workers_to_1",
            "emergency_reduce_cache_to_64mb",
            "emergency_reduce_batch_to_4",
            "emergency_clear_cache"
        ])
        
        return optimizations
    
    def _update_adaptive_thresholds(self, metrics: ResourceMetrics, analysis: Dict[str, Any]):
        """Update adaptive thresholds based on performance."""
        # Adjust thresholds based on performance trends
        performance_score = analysis["performance_score"]
        
        if performance_score > 80:  # System performing well
            # Slightly increase thresholds to allow more resource usage
            self.adaptive_thresholds["cpu_threshold"] = min(85, self.adaptive_thresholds["cpu_threshold"] + 0.5)
            self.adaptive_thresholds["memory_threshold"] = min(85, self.adaptive_thresholds["memory_threshold"] + 0.5)
        elif performance_score < 50:  # System struggling
            # Decrease thresholds to be more conservative
            self.adaptive_thresholds["cpu_threshold"] = max(60, self.adaptive_thresholds["cpu_threshold"] - 1.0)
            self.adaptive_thresholds["memory_threshold"] = max(65, self.adaptive_thresholds["memory_threshold"] - 1.0)
    
    async def _save_metrics(self):
        """Save performance metrics to file."""
        # Save last 50 metrics to avoid large files
        recent_metrics = self.metrics_history[-50:] if len(self.metrics_history) > 50 else self.metrics_history
        
        metrics_data = {
            "timestamp": time.time(),
            "current_profile": {
                "name": self.current_profile.name,
                "max_workers": self.current_profile.max_workers,
                "cache_size_mb": self.current_profile.cache_size_mb,
                "batch_size": self.current_profile.batch_size
            },
            "adaptive_thresholds": self.adaptive_thresholds,
            "recent_metrics": [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_usage": m.disk_usage,
                    "process_count": m.process_count
                }
                for m in recent_metrics
            ],
            "cache_stats": {
                "cache_size": len(self.result_cache),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            }
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor():
            asyncio.run(self._monitor_loop(interval))
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started adaptive performance monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped adaptive performance monitoring")
    
    async def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self.adaptive_optimization()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"status": "no_data", "monitoring_active": self.monitoring_active}
        
        latest_metrics = self.metrics_history[-1]
        analysis = self.analyze_performance_trend()
        
        return {
            "status": "active",
            "monitoring_active": self.monitoring_active,
            "current_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "disk_usage": latest_metrics.disk_usage,
                "process_count": latest_metrics.process_count
            },
            "current_profile": {
                "name": self.current_profile.name,
                "max_workers": self.current_profile.max_workers,
                "cache_size_mb": self.current_profile.cache_size_mb,
                "batch_size": self.current_profile.batch_size
            },
            "performance_analysis": analysis,
            "cache_stats": {
                "size": len(self.result_cache),
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            },
            "optimization_count": len(self.optimization_history),
            "adaptive_thresholds": self.adaptive_thresholds
        }


# Integration function
async def optimize_system_performance(monitoring_duration: int = 60) -> Dict[str, Any]:
    """Run adaptive performance optimization for specified duration."""
    scaler = AdaptiveScalingSystem()
    
    # Start monitoring
    scaler.start_monitoring(interval=10)
    
    # Wait for monitoring period
    await asyncio.sleep(monitoring_duration)
    
    # Stop monitoring and get summary
    scaler.stop_monitoring()
    summary = scaler.get_performance_summary()
    
    # Save configuration
    scaler.save_config()
    
    return summary