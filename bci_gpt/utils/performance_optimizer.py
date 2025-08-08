"""Performance optimization utilities for BCI-GPT."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import warnings
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from functools import lru_cache, wraps
from dataclasses import dataclass
import hashlib
import pickle
import os
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .logging_config import get_logger, performance_monitor
from .monitoring import get_metrics_collector


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    function_name: str
    total_calls: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0


class AdaptiveCache:
    """Adaptive caching system with automatic eviction and hit-rate optimization."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: float = 3600,
                 cleanup_interval: float = 300):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items
            cleanup_interval: How often to clean expired items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
        
        self.logger = get_logger()
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup of expired cache entries."""
        while self.running:
            try:
                self._cleanup_expired()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.log_error("Cache cleanup error", e)
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, creation_time in self.creation_times.items():
                if current_time - creation_time > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self.logger.log_info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_key(self, key: str):
        """Remove key from all cache structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used items."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Find LRU item
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                self._remove_key(lru_key)
                self.logger.log_info(f"Evicted LRU cache entry: {lru_key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Update access time
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            current_time = time.time()
            
            # Evict if necessary
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def invalidate(self, key: str):
        """Invalidate specific cache entry."""
        with self.lock:
            self._remove_key(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "fill_ratio": len(self.cache) / self.max_size
            }
    
    def stop(self):
        """Stop cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1.0)


class BatchProcessor:
    """Batch processing system for efficient EEG data processing."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 max_workers: int = None,
                 use_gpu: bool = True):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of processing batches
            max_workers: Maximum number of worker threads
            use_gpu: Whether to use GPU acceleration
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, mp.cpu_count() + 4)
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()
        
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
        
        # Setup device
        if self.use_gpu:
            self.device = torch.device("cuda")
            self.logger.log_info(f"Using GPU for batch processing: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            self.logger.log_info("Using CPU for batch processing")
    
    @performance_monitor("batch_process_eeg")
    def process_eeg_batch(self, 
                         eeg_data: np.ndarray,
                         process_func: Callable,
                         **kwargs) -> List[Any]:
        """Process EEG data in batches.
        
        Args:
            eeg_data: EEG data array (samples x channels x time_points)
            process_func: Function to apply to each batch
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processing results
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required for batch processing")
        
        num_samples = eeg_data.shape[0]
        results = []
        
        # Process in batches
        for i in range(0, num_samples, self.batch_size):
            batch_end = min(i + self.batch_size, num_samples)
            batch_data = eeg_data[i:batch_end]
            
            batch_result = process_func(batch_data, **kwargs)
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            # Log progress for large batches
            if num_samples > 100 and (i + self.batch_size) % (self.batch_size * 10) == 0:
                progress = (batch_end / num_samples) * 100
                self.logger.log_info(f"Batch processing progress: {progress:.1f}%")
        
        return results
    
    @performance_monitor("parallel_process")
    def parallel_process(self, 
                        items: List[Any],
                        process_func: Callable,
                        use_threads: bool = True) -> List[Any]:
        """Process items in parallel.
        
        Args:
            items: Items to process
            process_func: Function to apply to each item
            use_threads: Use ThreadPoolExecutor vs ProcessPoolExecutor
            
        Returns:
            List of processing results
        """
        results = [None] * len(items)
        
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_func, item): i
                for i, item in enumerate(items)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.log_error(f"Parallel processing error for item {index}", e)
                    results[index] = None
        
        return results


class ModelOptimizer:
    """Model optimization utilities for improved inference speed."""
    
    def __init__(self):
        self.logger = get_logger()
        self.optimized_models = {}
    
    def optimize_model(self, 
                      model: Any,
                      optimization_level: str = "medium",
                      compile_model: bool = False) -> Any:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            optimization_level: "low", "medium", "high"
            compile_model: Whether to use torch.compile (PyTorch 2.0+)
            
        Returns:
            Optimized model
        """
        if not HAS_TORCH:
            self.logger.log_warning("PyTorch not available, returning original model")
            return model
        
        model_id = id(model)
        if model_id in self.optimized_models:
            return self.optimized_models[model_id]
        
        optimized_model = model
        
        try:
            # Set to evaluation mode
            if hasattr(optimized_model, 'eval'):
                optimized_model.eval()
            
            # Apply optimizations based on level
            if optimization_level in ["medium", "high"]:
                # Fuse operations where possible
                if hasattr(torch, 'jit') and hasattr(optimized_model, 'forward'):
                    try:
                        optimized_model = torch.jit.optimize_for_inference(optimized_model)
                        self.logger.log_info("Applied TorchScript optimizations")
                    except Exception as e:
                        self.logger.log_warning(f"TorchScript optimization failed: {e}")
            
            if optimization_level == "high":
                # Additional high-level optimizations
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                    self.logger.log_info("Enabled cuDNN benchmark mode")
            
            # Compile model if requested (PyTorch 2.0+)
            if compile_model and hasattr(torch, 'compile'):
                try:
                    optimized_model = torch.compile(optimized_model)
                    self.logger.log_info("Compiled model with torch.compile")
                except Exception as e:
                    self.logger.log_warning(f"Model compilation failed: {e}")
            
            # Cache optimized model
            self.optimized_models[model_id] = optimized_model
            
            return optimized_model
            
        except Exception as e:
            self.logger.log_error("Model optimization failed", e)
            return model
    
    def setup_mixed_precision(self, model: Any) -> Tuple[Any, Any]:
        """Setup mixed precision training/inference.
        
        Returns:
            Tuple of (model, scaler)
        """
        if not HAS_TORCH or not torch.cuda.is_available():
            return model, None
        
        try:
            # Enable automatic mixed precision
            if hasattr(model, 'half'):
                # Convert model to half precision for inference
                model = model.half()
                self.logger.log_info("Enabled half precision for model")
            
            # Create gradient scaler for training
            scaler = GradScaler()
            
            return model, scaler
            
        except Exception as e:
            self.logger.log_error("Mixed precision setup failed", e)
            return model, None
    
    def optimize_memory_usage(self, model: Any):
        """Optimize GPU memory usage."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return
        
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
                self.logger.log_info("Enabled flash attention")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if high memory usage detected
            memory_info = torch.cuda.mem_get_info()
            free_memory = memory_info[0]
            total_memory = memory_info[1]
            usage_percent = (1 - free_memory / total_memory) * 100
            
            if usage_percent > 85:
                torch.cuda.set_per_process_memory_fraction(0.8)
                self.logger.log_warning(f"High GPU memory usage ({usage_percent:.1f}%), limited memory fraction")
            
        except Exception as e:
            self.logger.log_error("Memory optimization failed", e)


class ConnectionPool:
    """Connection pooling for external resources."""
    
    def __init__(self, 
                 create_connection: Callable,
                 max_connections: int = 10,
                 timeout: float = 30.0):
        """Initialize connection pool.
        
        Args:
            create_connection: Function to create new connections
            max_connections: Maximum number of connections in pool
            timeout: Connection timeout in seconds
        """
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.timeout = timeout
        
        self.pool = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        self.logger = get_logger()
    
    def get_connection(self):
        """Get connection from pool."""
        try:
            # Try to get existing connection
            connection = self.pool.get_nowait()
            return connection
        except queue.Empty:
            # Create new connection if under limit
            with self.lock:
                if self.active_connections < self.max_connections:
                    connection = self.create_connection()
                    self.active_connections += 1
                    return connection
                else:
                    # Wait for available connection
                    try:
                        connection = self.pool.get(timeout=self.timeout)
                        return connection
                    except queue.Empty:
                        raise TimeoutError("Connection pool timeout")
    
    def return_connection(self, connection):
        """Return connection to pool."""
        try:
            self.pool.put_nowait(connection)
        except queue.Full:
            # Pool is full, close connection
            if hasattr(connection, 'close'):
                connection.close()
            with self.lock:
                self.active_connections -= 1
    
    def close_all(self):
        """Close all connections in pool."""
        while not self.pool.empty():
            try:
                connection = self.pool.get_nowait()
                if hasattr(connection, 'close'):
                    connection.close()
            except queue.Empty:
                break
        
        with self.lock:
            self.active_connections = 0


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.logger = get_logger()
        self.cache = AdaptiveCache()
        self.batch_processor = BatchProcessor()
        self.model_optimizer = ModelOptimizer()
        
        # Performance profiles
        self.profiles = {}
        self.optimization_suggestions = {}
    
    def cached_function(self, 
                       ttl: float = 3600,
                       key_func: Optional[Callable] = None):
        """Decorator for caching function results.
        
        Args:
            ttl: Time-to-live for cached results
            key_func: Custom function to generate cache keys
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                    cache_key = hashlib.md5(key_data.encode()).hexdigest()
                
                # Check cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache.put(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def profile_function(self, func: Callable) -> PerformanceProfile:
        """Profile function performance."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        if func_name not in self.profiles:
            self.profiles[func_name] = {
                'calls': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'times': []
            }
        
        # Wrapper to collect metrics
        @wraps(func)
        def profiled_func(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                # Update profile
                profile = self.profiles[func_name]
                profile['calls'] += 1
                profile['total_time'] += duration
                profile['min_time'] = min(profile['min_time'], duration)
                profile['max_time'] = max(profile['max_time'], duration)
                profile['times'].append(duration)
                
                # Keep only recent times
                if len(profile['times']) > 1000:
                    profile['times'] = profile['times'][-500:]
            
            return result
        
        return profiled_func
    
    def get_performance_profile(self, func_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for function."""
        if func_name not in self.profiles:
            return None
        
        profile_data = self.profiles[func_name]
        
        if profile_data['calls'] == 0:
            return None
        
        avg_time = profile_data['total_time'] / profile_data['calls']
        
        # Get cache stats for this function
        cache_stats = self.cache.get_stats()
        
        return PerformanceProfile(
            function_name=func_name,
            total_calls=profile_data['calls'],
            total_time=profile_data['total_time'],
            avg_time=avg_time,
            min_time=profile_data['min_time'],
            max_time=profile_data['max_time'],
            cache_hits=cache_stats.get('hit_count', 0),
            cache_misses=cache_stats.get('miss_count', 0)
        )
    
    def optimize_system(self) -> Dict[str, Any]:
        """Perform system-wide optimization."""
        optimizations = {}
        
        try:
            # Cache optimization
            cache_stats = self.cache.get_stats()
            if cache_stats['hit_rate'] < 0.5:
                # Low hit rate, increase cache size
                self.cache.max_size = int(self.cache.max_size * 1.5)
                optimizations['cache'] = f"Increased cache size to {self.cache.max_size}"
            
            # Memory optimization
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimizations['memory'] = "Cleared GPU cache"
            
            # Thread optimization
            optimal_threads = min(32, mp.cpu_count() + 4)
            if self.batch_processor.max_workers != optimal_threads:
                self.batch_processor.max_workers = optimal_threads
                optimizations['threads'] = f"Set optimal thread count: {optimal_threads}"
            
            self.logger.log_info(f"System optimization completed: {optimizations}")
            return optimizations
            
        except Exception as e:
            self.logger.log_error("System optimization failed", e)
            return {"error": str(e)}
    
    def get_optimization_report(self) -> str:
        """Generate optimization recommendations report."""
        report = "=== BCI-GPT Performance Optimization Report ===\n\n"
        
        # Cache statistics
        cache_stats = self.cache.get_stats()
        report += f"Cache Performance:\n"
        report += f"  Hit Rate: {cache_stats['hit_rate']:.2%}\n"
        report += f"  Size: {cache_stats['size']}/{cache_stats['max_size']} ({cache_stats['fill_ratio']:.1%} full)\n"
        report += f"  Hits/Misses: {cache_stats['hit_count']}/{cache_stats['miss_count']}\n\n"
        
        # Function profiles
        if self.profiles:
            report += "Function Performance:\n"
            for func_name, profile_data in self.profiles.items():
                if profile_data['calls'] > 0:
                    avg_time = profile_data['total_time'] / profile_data['calls']
                    report += f"  {func_name}:\n"
                    report += f"    Calls: {profile_data['calls']}\n"
                    report += f"    Avg Time: {avg_time*1000:.2f}ms\n"
                    report += f"    Total Time: {profile_data['total_time']:.2f}s\n"
        
        # Optimization suggestions
        report += "\nOptimization Suggestions:\n"
        
        if cache_stats['hit_rate'] < 0.3:
            report += "  • Low cache hit rate - consider increasing cache size or TTL\n"
        
        if cache_stats['fill_ratio'] > 0.9:
            report += "  • Cache nearly full - consider increasing max_size\n"
        
        # Check for slow functions
        slow_functions = []
        for func_name, profile_data in self.profiles.items():
            if profile_data['calls'] > 0:
                avg_time = profile_data['total_time'] / profile_data['calls']
                if avg_time > 0.1:  # Functions taking > 100ms
                    slow_functions.append((func_name, avg_time))
        
        if slow_functions:
            report += "  • Slow functions detected:\n"
            for func_name, avg_time in sorted(slow_functions, key=lambda x: x[1], reverse=True)[:5]:
                report += f"    - {func_name}: {avg_time*1000:.1f}ms average\n"
        
        if not slow_functions and cache_stats['hit_rate'] > 0.8:
            report += "  • Performance looks good! No major optimizations needed.\n"
        
        return report


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    
    return _global_optimizer


def cached(ttl: float = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    return get_performance_optimizer().cached_function(ttl=ttl, key_func=key_func)


def optimized(func: Callable):
    """Decorator for performance optimization and profiling."""
    optimizer = get_performance_optimizer()
    return optimizer.profile_function(func)