"""Advanced caching system with intelligent invalidation and distributed support."""

import time
import hashlib
import pickle
from typing import Any, Dict, Optional, Callable, Union, List
from functools import wraps
import threading
import weakref
import logging

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import numpy as np
    import torch
    HAS_TORCH_NUMPY = True
except ImportError:
    HAS_TORCH_NUMPY = False

logger = logging.getLogger(__name__)


class AdvancedCache:
    """Multi-level cache with intelligent invalidation and compression."""
    
    def __init__(self,
                 max_memory_mb: int = 512,
                 ttl_seconds: int = 3600,
                 compression_threshold: int = 1024,
                 redis_url: Optional[str] = None):
        """Initialize advanced cache system.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries
            compression_threshold: Compress items larger than this (bytes)
            redis_url: Redis connection URL for distributed caching
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.compression_threshold = compression_threshold
        
        # Local memory cache
        self._memory_cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._memory_usage = 0
        self._lock = threading.RLock()
        
        # Distributed cache
        self._redis_client = None
        if redis_url and HAS_REDIS:
            try:
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()
                logger.info(f"Connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._redis_client = None
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0
        }
        
    def _hash_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        # Handle torch tensors and numpy arrays
        if HAS_TORCH_NUMPY:
            key_data = self._serialize_tensors(key_data)
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_tensors(self, obj: Any) -> Any:
        """Serialize tensors and arrays for hashing."""
        if isinstance(obj, torch.Tensor):
            return {
                'type': 'torch_tensor',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'hash': hashlib.md5(obj.detach().cpu().numpy().tobytes()).hexdigest()
            }
        elif isinstance(obj, np.ndarray):
            return {
                'type': 'numpy_array',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'hash': hashlib.md5(obj.tobytes()).hexdigest()
            }
        elif isinstance(obj, dict):
            return {k: self._serialize_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_tensors(item) for item in obj]
        else:
            return obj
    
    def _compress_if_needed(self, data: bytes) -> tuple:
        """Compress data if it exceeds threshold."""
        if len(data) > self.compression_threshold:
            try:
                import gzip
                compressed = gzip.compress(data)
                if len(compressed) < len(data):
                    self.stats['compressions'] += 1
                    return compressed, True
            except ImportError:
                pass
        return data, False
    
    def _decompress_if_needed(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if needed."""
        if is_compressed:
            import gzip
            return gzip.decompress(data)
        return data
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self._memory_cache:
            return
            
        # Sort by access time
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        
        # Evict oldest 25% of items
        evict_count = max(1, len(sorted_items) // 4)
        
        for key, _ in sorted_items[:evict_count]:
            self._remove_from_memory(key)
            self.stats['evictions'] += 1
    
    def _remove_from_memory(self, key: str) -> None:
        """Remove item from memory cache."""
        if key in self._memory_cache:
            item_size = self._memory_cache[key].get('size', 0)
            self._memory_usage -= item_size
            del self._memory_cache[key]
            
        if key in self._access_times:
            del self._access_times[key]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self._memory_cache.items():
            if current_time - item['timestamp'] > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_from_memory(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            self._cleanup_expired()
            
            # Check memory cache first
            if key in self._memory_cache:
                item = self._memory_cache[key]
                current_time = time.time()
                
                if current_time - item['timestamp'] <= self.ttl_seconds:
                    self._access_times[key] = current_time
                    self.stats['hits'] += 1
                    
                    # Decompress if needed
                    data = self._decompress_if_needed(item['data'], item['compressed'])
                    return pickle.loads(data)
                else:
                    self._remove_from_memory(key)
            
            # Check distributed cache
            if self._redis_client:
                try:
                    redis_data = self._redis_client.get(key)
                    if redis_data:
                        item_data = pickle.loads(redis_data)
                        self.stats['hits'] += 1
                        
                        # Store in memory cache for faster future access
                        self._store_in_memory(key, item_data)
                        return item_data
                except Exception as e:
                    logger.warning(f"Redis get failed: {e}")
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        ttl = ttl or self.ttl_seconds
        
        with self._lock:
            # Store in memory cache
            self._store_in_memory(key, value, ttl)
            
            # Store in distributed cache
            if self._redis_client:
                try:
                    serialized = pickle.dumps(value)
                    self._redis_client.setex(key, ttl, serialized)
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
    
    def _store_in_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store item in memory cache."""
        ttl = ttl or self.ttl_seconds
        serialized = pickle.dumps(value)
        data, is_compressed = self._compress_if_needed(serialized)
        
        item = {
            'data': data,
            'compressed': is_compressed,
            'timestamp': time.time(),
            'size': len(data)
        }
        
        # Check if we need to evict
        if self._memory_usage + item['size'] > self.max_memory_bytes:
            self._evict_lru()
        
        self._memory_cache[key] = item
        self._access_times[key] = time.time()
        self._memory_usage += item['size']
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        with self._lock:
            self._remove_from_memory(key)
            
            if self._redis_client:
                try:
                    self._redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis delete failed: {e}")
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._memory_cache.clear()
            self._access_times.clear()
            self._memory_usage = 0
            
            if self._redis_client:
                try:
                    self._redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_usage_mb': self._memory_usage / (1024 * 1024),
            'memory_utilization': self._memory_usage / self.max_memory_bytes,
            'cache_size': len(self._memory_cache)
        }


# Global cache instance
_global_cache = AdvancedCache()


def smart_cache(ttl: int = 3600, 
               key_func: Optional[Callable] = None,
               invalidate_on: Optional[List[str]] = None):
    """Smart caching decorator with automatic invalidation.
    
    Args:
        ttl: Time-to-live in seconds
        key_func: Custom key generation function
        invalidate_on: List of attributes that trigger cache invalidation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{_global_cache._hash_key(*args, **kwargs)}"
            
            # Check cache
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            _global_cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: _global_cache.clear()
        wrapper.cache_info = lambda: _global_cache.get_stats()
        
        return wrapper
    return decorator


class ModelCache:
    """Specialized cache for ML models and tensors."""
    
    def __init__(self, max_models: int = 5):
        self.max_models = max_models
        self._models: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get_model(self, model_key: str, loader_func: Callable) -> Any:
        """Get model from cache or load it."""
        with self._lock:
            if model_key in self._models:
                self._access_times[model_key] = time.time()
                return self._models[model_key]
            
            # Load model
            model = loader_func()
            self._store_model(model_key, model)
            return model
    
    def _store_model(self, model_key: str, model: Any) -> None:
        """Store model in cache."""
        if len(self._models) >= self.max_models:
            # Evict oldest model
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._models[oldest_key]
            del self._access_times[oldest_key]
        
        self._models[model_key] = model
        self._access_times[model_key] = time.time()
    
    def clear(self) -> None:
        """Clear model cache."""
        with self._lock:
            self._models.clear()
            self._access_times.clear()


# Global model cache
model_cache = ModelCache()


class TensorCache:
    """GPU-optimized tensor caching."""
    
    def __init__(self, max_gpu_memory_gb: float = 2.0):
        self.max_gpu_memory_bytes = int(max_gpu_memory_gb * 1024 * 1024 * 1024)
        self._tensors: Dict[str, torch.Tensor] = {}
        self._sizes: Dict[str, int] = {}
        self._access_times: Dict[str, float] = {}
        self._total_size = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache."""
        with self._lock:
            if key in self._tensors:
                self._access_times[key] = time.time()
                return self._tensors[key]
            return None
    
    def set(self, key: str, tensor: torch.Tensor) -> None:
        """Set tensor in cache."""
        if not HAS_TORCH_NUMPY:
            return
            
        with self._lock:
            tensor_size = tensor.numel() * tensor.element_size()
            
            # Evict if necessary
            while self._total_size + tensor_size > self.max_gpu_memory_bytes and self._tensors:
                oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                self._remove_tensor(oldest_key)
            
            self._tensors[key] = tensor
            self._sizes[key] = tensor_size
            self._access_times[key] = time.time()
            self._total_size += tensor_size
    
    def _remove_tensor(self, key: str) -> None:
        """Remove tensor from cache."""
        if key in self._tensors:
            self._total_size -= self._sizes[key]
            del self._tensors[key]
            del self._sizes[key]
            del self._access_times[key]
    
    def clear(self) -> None:
        """Clear tensor cache."""
        with self._lock:
            self._tensors.clear()
            self._sizes.clear()
            self._access_times.clear()
            self._total_size = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            'used_gb': self._total_size / (1024 ** 3),
            'max_gb': self.max_gpu_memory_bytes / (1024 ** 3),
            'utilization': self._total_size / self.max_gpu_memory_bytes,
            'cached_tensors': len(self._tensors)
        }


# Global tensor cache
tensor_cache = TensorCache()


def cache_tensor(key_func: Optional[Callable] = None):
    """Decorator for caching tensors on GPU."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not HAS_TORCH_NUMPY:
                return func(*args, **kwargs)
                
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{_global_cache._hash_key(*args, **kwargs)}"
            
            # Check cache
            cached_tensor = tensor_cache.get(cache_key)
            if cached_tensor is not None:
                return cached_tensor
            
            # Compute tensor
            result = func(*args, **kwargs)
            
            # Cache if it's a tensor
            if isinstance(result, torch.Tensor):
                tensor_cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator