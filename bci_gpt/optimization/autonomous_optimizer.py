"""Autonomous optimization system for BCI-GPT with adaptive performance tuning."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationConfig:
    """Configuration for autonomous optimization."""
    target_latency_ms: float = 100.0
    target_throughput: float = 1000.0
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 85.0
    max_gpu_usage: float = 90.0
    optimization_interval: float = 30.0
    adaptation_factor: float = 0.1


class AutonomousOptimizer:
    """Autonomous system optimizer with adaptive performance tuning."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize autonomous optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[SystemMetrics] = []
        self.optimization_strategies = {}
        self.active_optimizations = set()
        self._monitoring = False
        self._monitor_thread = None
        
        # Initialize optimization strategies
        self._init_optimization_strategies()
    
    def _init_optimization_strategies(self):
        """Initialize available optimization strategies."""
        self.optimization_strategies = {
            "batch_size_scaling": self._optimize_batch_size,
            "memory_management": self._optimize_memory,
            "cpu_affinity": self._optimize_cpu_affinity,
            "inference_optimization": self._optimize_inference,
            "cache_tuning": self._optimize_cache,
            "load_balancing": self._optimize_load_balancing
        }
    
    def start_monitoring(self):
        """Start autonomous performance monitoring and optimization."""
        if self._monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Autonomous optimization monitoring started")
    
    def stop_monitoring(self):
        """Stop autonomous monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Autonomous optimization monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and optimization loop."""
        while self._monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Analyze performance and optimize
                self._analyze_and_optimize(metrics)
                
                # Sleep until next optimization cycle
                time.sleep(self.config.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(self.config.optimization_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        metrics = SystemMetrics()
        
        try:
            # CPU and memory
            metrics.cpu_usage = psutil.cpu_percent(interval=1)
            metrics.memory_usage = psutil.virtual_memory().percent
            
            # GPU metrics (if available)
            if HAS_TORCH and torch.cuda.is_available():
                try:
                    metrics.gpu_usage = torch.cuda.utilization()
                    metrics.gpu_memory = torch.cuda.memory_percent()
                except:
                    pass
            
            self.logger.debug(f"Collected metrics: CPU={metrics.cpu_usage}%, "
                            f"Memory={metrics.memory_usage}%, "
                            f"GPU={metrics.gpu_usage}%")
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _analyze_and_optimize(self, current_metrics: SystemMetrics):
        """Analyze metrics and apply optimizations."""
        try:
            # Check if optimization is needed
            optimizations_needed = self._identify_optimizations_needed(current_metrics)
            
            if not optimizations_needed:
                return
            
            self.logger.info(f"Applying optimizations: {optimizations_needed}")
            
            # Apply optimizations
            for optimization in optimizations_needed:
                if optimization in self.optimization_strategies:
                    try:
                        self.optimization_strategies[optimization](current_metrics)
                        self.active_optimizations.add(optimization)
                    except Exception as e:
                        self.logger.error(f"Error applying {optimization}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in analysis and optimization: {e}")
    
    def _identify_optimizations_needed(self, metrics: SystemMetrics) -> List[str]:
        """Identify which optimizations are needed based on metrics."""
        needed = []
        
        # High CPU usage
        if metrics.cpu_usage > self.config.max_cpu_usage:
            if "cpu_affinity" not in self.active_optimizations:
                needed.append("cpu_affinity")
            if "batch_size_scaling" not in self.active_optimizations:
                needed.append("batch_size_scaling")
        
        # High memory usage
        if metrics.memory_usage > self.config.max_memory_usage:
            needed.append("memory_management")
            needed.append("cache_tuning")
        
        # High GPU usage
        if metrics.gpu_usage > self.config.max_gpu_usage:
            needed.append("inference_optimization")
        
        # High latency
        if metrics.latency > self.config.target_latency_ms:
            needed.append("load_balancing")
            needed.append("inference_optimization")
        
        # Low throughput
        if 0 < metrics.throughput < self.config.target_throughput:
            needed.append("batch_size_scaling")
            needed.append("load_balancing")
        
        return needed
    
    def _optimize_batch_size(self, metrics: SystemMetrics):
        """Dynamically optimize batch size based on system resources."""
        self.logger.info("Optimizing batch size based on system resources")
        
        # Calculate optimal batch size based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate batch size (rough heuristic)
        if available_memory_gb > 8:
            optimal_batch_size = 64
        elif available_memory_gb > 4:
            optimal_batch_size = 32
        elif available_memory_gb > 2:
            optimal_batch_size = 16
        else:
            optimal_batch_size = 8
        
        # Adjust based on CPU usage
        if metrics.cpu_usage > 80:
            optimal_batch_size = max(8, optimal_batch_size // 2)
        
        self.logger.info(f"Recommended batch size: {optimal_batch_size}")
        
        # Store recommendation (would be used by training/inference systems)
        self._store_optimization_param("batch_size", optimal_batch_size)
    
    def _optimize_memory(self, metrics: SystemMetrics):
        """Optimize memory usage and trigger garbage collection."""
        self.logger.info("Optimizing memory usage")
        
        # Python garbage collection
        import gc
        collected = gc.collect()
        self.logger.debug(f"Garbage collected: {collected} objects")
        
        # PyTorch cache cleanup
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Cleared CUDA cache")
        
        # Set memory optimization flags
        self._store_optimization_param("memory_optimized", True)
    
    def _optimize_cpu_affinity(self, metrics: SystemMetrics):
        """Optimize CPU affinity for better performance."""
        self.logger.info("Optimizing CPU affinity")
        
        try:
            # Get available CPUs
            cpu_count = psutil.cpu_count(logical=False)
            available_cpus = list(range(cpu_count))
            
            # Reserve some CPUs for system processes
            if cpu_count > 4:
                available_cpus = available_cpus[:-2]  # Reserve 2 CPUs
            
            # Set CPU affinity for current process
            current_process = psutil.Process()
            current_process.cpu_affinity(available_cpus)
            
            self.logger.info(f"Set CPU affinity to cores: {available_cpus}")
            self._store_optimization_param("cpu_affinity", available_cpus)
            
        except Exception as e:
            self.logger.error(f"Error setting CPU affinity: {e}")
    
    def _optimize_inference(self, metrics: SystemMetrics):
        """Optimize inference performance."""
        self.logger.info("Optimizing inference performance")
        
        optimizations = {}
        
        # Mixed precision if GPU available
        if HAS_TORCH and torch.cuda.is_available():
            optimizations["use_mixed_precision"] = True
            optimizations["use_cuda_graphs"] = True
        
        # Batch processing optimization
        optimizations["enable_batch_processing"] = True
        optimizations["inference_batch_size"] = 16 if metrics.gpu_usage < 70 else 8
        
        # Model optimization flags
        optimizations["use_torch_compile"] = True
        optimizations["use_torch_jit"] = True
        
        for key, value in optimizations.items():
            self._store_optimization_param(key, value)
        
        self.logger.info(f"Applied inference optimizations: {list(optimizations.keys())}")
    
    def _optimize_cache(self, metrics: SystemMetrics):
        """Optimize caching strategy."""
        self.logger.info("Optimizing cache configuration")
        
        # Calculate cache size based on available memory
        available_memory_mb = psutil.virtual_memory().available / (1024**2)
        
        # Use 10% of available memory for cache, max 1GB
        cache_size_mb = min(1024, available_memory_mb * 0.1)
        
        cache_config = {
            "cache_size_mb": cache_size_mb,
            "cache_ttl_seconds": 300,  # 5 minutes
            "enable_lru_cache": True,
            "enable_memory_cache": True
        }
        
        for key, value in cache_config.items():
            self._store_optimization_param(key, value)
        
        self.logger.info(f"Optimized cache: {cache_size_mb:.1f}MB")
    
    def _optimize_load_balancing(self, metrics: SystemMetrics):
        """Optimize load balancing parameters."""
        self.logger.info("Optimizing load balancing")
        
        # Calculate optimal worker count
        cpu_count = psutil.cpu_count()
        
        if metrics.cpu_usage > 80:
            optimal_workers = max(1, cpu_count // 2)
        elif metrics.cpu_usage < 40:
            optimal_workers = min(cpu_count, cpu_count + 2)
        else:
            optimal_workers = cpu_count
        
        load_balance_config = {
            "worker_count": optimal_workers,
            "request_timeout": 30.0,
            "max_requests_per_worker": 1000,
            "enable_request_queuing": True
        }
        
        for key, value in load_balance_config.items():
            self._store_optimization_param(key, value)
        
        self.logger.info(f"Optimized load balancing: {optimal_workers} workers")
    
    def _store_optimization_param(self, key: str, value: Any):
        """Store optimization parameter for use by other systems."""
        # This would typically store in a shared configuration or message queue
        # For now, we'll just log it
        self.logger.debug(f"Optimization parameter: {key} = {value}")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations."""
        if not self.metrics_history:
            return {}
        
        current_metrics = self.metrics_history[-1]
        recommendations = {}
        
        # Analyze recent performance
        if len(self.metrics_history) > 10:
            recent_metrics = self.metrics_history[-10:]
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            
            recommendations.update({
                "average_cpu_usage": avg_cpu,
                "average_memory_usage": avg_memory,
                "performance_trend": "stable",  # Could be more sophisticated
                "active_optimizations": list(self.active_optimizations)
            })
        
        return recommendations


def get_autonomous_optimizer() -> AutonomousOptimizer:
    """Get global autonomous optimizer instance."""
    global _optimizer
    if '_optimizer' not in globals():
        _optimizer = AutonomousOptimizer()
    return _optimizer