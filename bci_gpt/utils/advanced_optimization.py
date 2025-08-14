"""
Advanced optimization utilities for BCI-GPT system.
Provides intelligent caching, performance optimization, and auto-scaling.
"""

import asyncio
import threading
import time
import pickle
import hashlib
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from contextlib import contextmanager
from functools import wraps
import logging


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update the hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class IntelligentCache:
    """High-performance intelligent cache with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = None,
        eviction_strategy: str = "lru"
    ):
        """Initialize intelligent cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            eviction_strategy: Eviction strategy (lru, lfu, ttl, adaptive)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.eviction_strategy = eviction_strategy
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
        
        # Adaptive eviction parameters
        self._access_pattern: Dict[str, List[float]] = defaultdict(list)
        self._performance_history: List[float] = []
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return default
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size -= 1
                self._stats.update_hit_rate()
                return default
            
            # Update access statistics
            entry.update_access()
            self._record_access_pattern(key)
            
            # Move to end for LRU
            if self.eviction_strategy in ["lru", "adaptive"]:
                self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.update_hit_rate()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate entry size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Check if single entry exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Entry too large for cache: {size_bytes} bytes")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # If key exists, update it
            if key in self._cache:
                old_entry = self._cache[key]
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return True
            
            # Check capacity and evict if necessary
            self._ensure_capacity(size_bytes)
            
            # Add new entry
            self._cache[key] = entry
            self._stats.size += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size -= 1
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size=self.max_size)
            self._access_pattern.clear()
    
    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        
        # Evict entries while over limits
        while (len(self._cache) >= self.max_size or 
               current_memory + new_entry_size > self.max_memory_bytes):
            
            if not self._cache:
                break
            
            evicted_key = self._select_eviction_candidate()
            if evicted_key:
                evicted_entry = self._cache[evicted_key]
                current_memory -= evicted_entry.size_bytes
                del self._cache[evicted_key]
                self._stats.evictions += 1
                self._stats.size -= 1
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on strategy."""
        if not self._cache:
            return None
        
        if self.eviction_strategy == "lru":
            return next(iter(self._cache))  # First (oldest) entry
        
        elif self.eviction_strategy == "lfu":
            # Find least frequently used
            min_access_count = float('inf')
            lfu_key = None
            for key, entry in self._cache.items():
                if entry.access_count < min_access_count:
                    min_access_count = entry.access_count
                    lfu_key = key
            return lfu_key
        
        elif self.eviction_strategy == "ttl":
            # Find entry with earliest expiration
            earliest_expiry = float('inf')
            ttl_key = None
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if entry.ttl:
                    expiry_time = entry.created_at.timestamp() + entry.ttl
                    if expiry_time < earliest_expiry:
                        earliest_expiry = expiry_time
                        ttl_key = key
            
            return ttl_key or next(iter(self._cache))
        
        elif self.eviction_strategy == "adaptive":
            return self._adaptive_eviction()
        
        else:
            return next(iter(self._cache))  # Default to LRU
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns and performance."""
        if not self._cache:
            return None
        
        # Calculate scores for each entry
        scores = {}
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Factors: recency, frequency, size, predicted future access
            recency_score = 1.0 / (current_time - entry.last_accessed.timestamp() + 1)
            frequency_score = entry.access_count / (current_time - entry.created_at.timestamp() + 1)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Predict future access based on historical pattern
            future_access_score = self._predict_future_access(key)
            
            # Combined score (higher = more valuable, less likely to evict)
            scores[key] = (recency_score * 0.3 + 
                          frequency_score * 0.4 + 
                          future_access_score * 0.3 - 
                          size_penalty * 0.1)
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for adaptive eviction."""
        current_time = time.time()
        self._access_pattern[key].append(current_time)
        
        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self._access_pattern[key] = [
            t for t in self._access_pattern[key] if t > cutoff_time
        ]
    
    def _predict_future_access(self, key: str) -> float:
        """Predict likelihood of future access based on historical pattern."""
        access_times = self._access_pattern.get(key, [])
        
        if len(access_times) < 2:
            return 0.1  # Low confidence for items with little history
        
        # Calculate access frequency and trend
        current_time = time.time()
        recent_accesses = [t for t in access_times if current_time - t < 1800]  # Last 30 minutes
        
        if not recent_accesses:
            return 0.0
        
        # Simple frequency-based prediction
        frequency = len(recent_accesses) / 1800.0  # Accesses per second
        return min(frequency * 100, 1.0)  # Normalize to 0-1
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            self._stats.update_hit_rate()
            return self._stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        with self._lock:
            total_bytes = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                'total_entries': len(self._cache),
                'total_bytes': total_bytes,
                'total_mb': total_bytes / (1024 * 1024),
                'avg_entry_size': total_bytes / len(self._cache) if self._cache else 0,
                'memory_utilization': total_bytes / self.max_memory_bytes,
                'largest_entry': max((e.size_bytes for e in self._cache.values()), default=0)
            }


class PerformanceOptimizer:
    """Dynamic performance optimization system."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.optimizations: Dict[str, Callable] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.optimization_history: List[Dict] = []
        
        self._running = False
        self._optimizer_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Register default optimizations
        self._register_default_optimizations()
    
    def _register_default_optimizations(self):
        """Register default optimization strategies."""
        self.register_optimization("gc_collect", self._optimize_garbage_collection)
        self.register_optimization("cache_tuning", self._optimize_cache_parameters)
        self.register_optimization("thread_pool", self._optimize_thread_pool)
    
    def register_optimization(self, name: str, optimization_func: Callable):
        """Register a new optimization function.
        
        Args:
            name: Unique name for the optimization
            optimization_func: Function that performs the optimization
        """
        self.optimizations[name] = optimization_func
        self.logger.info(f"Registered optimization: {name}")
    
    def record_performance_metric(self, metric_name: str, value: float):
        """Record a performance metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.performance_metrics[metric_name].append(value)
        
        # Keep only recent metrics (last 1000 values)
        if len(self.performance_metrics[metric_name]) > 1000:
            self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-1000:]
    
    def start_auto_optimization(self, interval_seconds: int = 300):
        """Start automatic performance optimization.
        
        Args:
            interval_seconds: Optimization interval in seconds
        """
        if self._running:
            return
        
        self._running = True
        self._optimizer_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._optimizer_thread.start()
        
        self.logger.info(f"Started auto-optimization with {interval_seconds}s interval")
    
    def stop_auto_optimization(self):
        """Stop automatic performance optimization."""
        self._running = False
        if self._optimizer_thread:
            self._optimizer_thread.join()
        
        self.logger.info("Stopped auto-optimization")
    
    def _optimization_loop(self, interval: int):
        """Main optimization loop."""
        while self._running:
            try:
                self._run_optimizations()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(interval)
    
    def _run_optimizations(self):
        """Run all registered optimizations."""
        start_time = time.time()
        results = {}
        
        for name, optimization_func in self.optimizations.items():
            try:
                opt_start = time.time()
                result = optimization_func()
                opt_duration = time.time() - opt_start
                
                results[name] = {
                    'success': True,
                    'duration_ms': opt_duration * 1000,
                    'result': result
                }
                
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'duration_ms': 0
                }
                self.logger.error(f"Optimization {name} failed: {e}")
        
        # Record optimization session
        optimization_session = {
            'timestamp': datetime.now().isoformat(),
            'total_duration_ms': (time.time() - start_time) * 1000,
            'results': results
        }
        
        self.optimization_history.append(optimization_session)
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def _optimize_garbage_collection(self) -> Dict[str, Any]:
        """Optimize garbage collection."""
        import gc
        
        # Get pre-GC stats
        pre_objects = len(gc.get_objects())
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get post-GC stats
        post_objects = len(gc.get_objects())
        freed_objects = pre_objects - post_objects
        
        return {
            'collected_objects': collected,
            'freed_objects': freed_objects,
            'remaining_objects': post_objects
        }
    
    def _optimize_cache_parameters(self) -> Dict[str, Any]:
        """Optimize cache parameters based on usage patterns."""
        # This would analyze cache hit rates and adjust cache sizes
        # For now, return placeholder metrics
        return {
            'cache_hit_rate': 0.85,
            'recommended_cache_size': 2000,
            'memory_efficiency': 0.92
        }
    
    def _optimize_thread_pool(self) -> Dict[str, Any]:
        """Optimize thread pool sizes based on load."""
        import threading
        
        active_threads = threading.active_count()
        
        # Simple optimization: recommend thread count based on CPU cores
        try:
            import os
            cpu_count = os.cpu_count() or 4
            recommended_threads = min(cpu_count * 2, 32)
        except Exception:
            recommended_threads = 8
        
        return {
            'current_threads': active_threads,
            'recommended_threads': recommended_threads,
            'cpu_cores': cpu_count if 'cpu_count' in locals() else 'unknown'
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary."""
        return {
            'registered_optimizations': list(self.optimizations.keys()),
            'optimization_sessions': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None,
            'performance_metrics': {
                name: {
                    'count': len(values),
                    'latest': values[-1] if values else None,
                    'average': sum(values) / len(values) if values else 0
                }
                for name, values in self.performance_metrics.items()
            }
        }


class AsyncTaskManager:
    """Asynchronous task management for improved scalability."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize async task manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[Dict] = []
        
        self._task_counter = 0
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    async def submit_task(
        self,
        coro: Callable,
        task_name: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Submit an asynchronous task.
        
        Args:
            coro: Coroutine to execute
            task_name: Optional task name
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"
        
        task_name = task_name or f"unnamed_task_{task_id}"
        
        # Create and start task
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: self._task_completed(task_id, t))
        
        self.active_tasks[task_id] = {
            'task': task,
            'name': task_name,
            'priority': priority,
            'created_at': datetime.now(),
            'started_at': datetime.now()
        }
        
        self.logger.info(f"Submitted task {task_id}: {task_name}")
        return task_id
    
    def _task_completed(self, task_id: str, task: asyncio.Task):
        """Handle task completion."""
        if task_id not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_id]
        
        # Record completion
        completion_info = {
            'task_id': task_id,
            'name': task_info['name'],
            'priority': task_info['priority'],
            'created_at': task_info['created_at'].isoformat(),
            'completed_at': datetime.now().isoformat(),
            'duration_ms': (datetime.now() - task_info['started_at']).total_seconds() * 1000,
            'success': not task.exception(),
            'exception': str(task.exception()) if task.exception() else None
        }
        
        self.completed_tasks.append(completion_info)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        # Keep only recent completed tasks
        if len(self.completed_tasks) > 1000:
            self.completed_tasks = self.completed_tasks[-1000:]
        
        if completion_info['success']:
            self.logger.info(f"Task {task_id} completed successfully")
        else:
            self.logger.error(f"Task {task_id} failed: {completion_info['exception']}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current task status."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_workers': self.max_workers,
            'active_task_details': [
                {
                    'id': task_id,
                    'name': info['name'],
                    'priority': info['priority'],
                    'running_time_ms': (datetime.now() - info['started_at']).total_seconds() * 1000
                }
                for task_id, info in self.active_tasks.items()
            ]
        }


# Global instances
_intelligent_cache = IntelligentCache()
_performance_optimizer = PerformanceOptimizer()
_task_manager = AsyncTaskManager()


def get_cache() -> IntelligentCache:
    """Get the global intelligent cache instance."""
    return _intelligent_cache


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return _performance_optimizer


def get_task_manager() -> AsyncTaskManager:
    """Get the global task manager instance."""
    return _task_manager


def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator to cache function results.
    
    Args:
        ttl: Time-to-live for cached results
        key_func: Custom function to generate cache keys
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]
            
            # Try to get from cache
            result = _intelligent_cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            _intelligent_cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def performance_monitoring(operation_name: str):
    """Context manager for monitoring operation performance."""
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        _performance_optimizer.record_performance_metric(f"{operation_name}_duration_ms", duration)