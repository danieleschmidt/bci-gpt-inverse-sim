"""Intelligent caching systems for EEG processing and inference."""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import hashlib
import pickle
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
import warnings

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    warnings.warn("Redis not available for distributed caching")

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False
    warnings.warn("SQLite not available for persistent caching")


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.memory_usage = 0.0
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def _get_item_size(self, item: Any) -> float:
        """Estimate item size in MB."""
        try:
            if isinstance(item, torch.Tensor):
                return item.element_size() * item.nelement() / (1024 ** 2)
            elif isinstance(item, np.ndarray):
                return item.nbytes / (1024 ** 2)
            else:
                # Fallback to pickle size
                return len(pickle.dumps(item)) / (1024 ** 2)
        except:
            return 0.1  # Default estimate
    
    def _evict_if_needed(self):
        """Evict items if cache is over limits."""
        while (len(self.cache) > self.max_size or 
               self.memory_usage > self.max_memory_mb):
            if not self.cache:
                break
            
            # Remove oldest item
            key, value = self.cache.popitem(last=False)
            self.memory_usage -= self._get_item_size(value)
            self.stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats.hits += 1
                return value
            else:
                self.stats.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            item_size = self._get_item_size(value)
            
            # Remove existing item if present
            if key in self.cache:
                old_value = self.cache.pop(key)
                self.memory_usage -= self._get_item_size(old_value)
            
            # Add new item
            self.cache[key] = value
            self.memory_usage += item_size
            
            # Update stats
            self.stats.memory_usage_mb = self.memory_usage
            
            # Evict if necessary
            self._evict_if_needed()
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.memory_usage = 0.0
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            self.stats.memory_usage_mb = self.memory_usage
            return self.stats


class EEGCache:
    """Specialized cache for EEG processing results."""
    
    def __init__(self, 
                 max_size: int = 500,
                 max_memory_mb: float = 50.0,
                 enable_persistence: bool = False,
                 db_path: str = "eeg_cache.db"):
        """Initialize EEG cache.
        
        Args:
            max_size: Maximum cache size
            max_memory_mb: Maximum memory usage
            enable_persistence: Whether to persist cache to disk
            db_path: Path to SQLite database for persistence
        """
        self.memory_cache = LRUCache(max_size, max_memory_mb)
        self.enable_persistence = enable_persistence and HAS_SQLITE
        self.db_path = db_path
        
        if self.enable_persistence:
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS eeg_cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                timestamp REAL,
                access_count INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
    
    def _compute_eeg_hash(self, eeg_data: np.ndarray, processing_params: Dict[str, Any]) -> str:
        """Compute hash for EEG data and processing parameters."""
        # Hash EEG data
        eeg_hash = hashlib.md5(eeg_data.tobytes()).hexdigest()
        
        # Hash processing parameters
        param_str = str(sorted(processing_params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return f"eeg_{eeg_hash}_{param_hash}"
    
    def get_processed_eeg(self, 
                         eeg_data: np.ndarray,
                         processing_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get processed EEG from cache.
        
        Args:
            eeg_data: Original EEG data
            processing_params: Processing parameters used
            
        Returns:
            Cached processed EEG or None if not found
        """
        cache_key = self._compute_eeg_hash(eeg_data, processing_params)
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Try persistent cache
        if self.enable_persistence:
            result = self._get_from_db(cache_key)
            if result is not None:
                # Add back to memory cache
                self.memory_cache.put(cache_key, result)
                return result
        
        return None
    
    def put_processed_eeg(self,
                         eeg_data: np.ndarray,
                         processing_params: Dict[str, Any],
                         processed_eeg: np.ndarray):
        """Cache processed EEG result.
        
        Args:
            eeg_data: Original EEG data
            processing_params: Processing parameters used
            processed_eeg: Processed EEG result
        """
        cache_key = self._compute_eeg_hash(eeg_data, processing_params)
        
        # Store in memory cache
        self.memory_cache.put(cache_key, processed_eeg)
        
        # Store in persistent cache
        if self.enable_persistence:
            self._put_to_db(cache_key, processed_eeg)
    
    def _get_from_db(self, key: str) -> Optional[np.ndarray]:
        """Get item from persistent database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT value FROM eeg_cache WHERE key = ?', (key,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update access count
                cursor.execute(
                    'UPDATE eeg_cache SET access_count = access_count + 1 WHERE key = ?',
                    (key,)
                )
                conn.commit()
                conn.close()
                
                # Deserialize data
                return pickle.loads(row[0])
            
            conn.close()
            return None
            
        except Exception as e:
            warnings.warn(f"Database access failed: {e}")
            return None
    
    def _put_to_db(self, key: str, value: np.ndarray):
        """Put item to persistent database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize data
            value_blob = pickle.dumps(value)
            
            cursor.execute('''
                INSERT OR REPLACE INTO eeg_cache 
                (key, value, timestamp, access_count) 
                VALUES (?, ?, ?, 1)
            ''', (key, value_blob, time.time()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            warnings.warn(f"Database write failed: {e}")
    
    def cleanup_old_entries(self, max_age_hours: float = 24.0):
        """Clean up old cache entries."""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            cursor.execute(
                'DELETE FROM eeg_cache WHERE timestamp < ?',
                (cutoff_time,)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            warnings.warn(f"Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'memory_cache': self.memory_cache.get_stats(),
            'persistence_enabled': self.enable_persistence
        }
        
        if self.enable_persistence:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM eeg_cache')
                db_entries = cursor.fetchone()[0]
                
                cursor.execute('SELECT SUM(LENGTH(value)) FROM eeg_cache')
                db_size_bytes = cursor.fetchone()[0] or 0
                
                stats['db_entries'] = db_entries
                stats['db_size_mb'] = db_size_bytes / (1024 ** 2)
                
                conn.close()
                
            except Exception as e:
                warnings.warn(f"Failed to get DB stats: {e}")
        
        return stats


class InferenceCache:
    """Cache for model inference results."""
    
    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: float = 200.0,
                 ttl_seconds: float = 3600.0,
                 enable_distributed: bool = False,
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        """Initialize inference cache.
        
        Args:
            max_size: Maximum cache size
            max_memory_mb: Maximum memory usage
            ttl_seconds: Time to live for cache entries
            enable_distributed: Whether to use Redis for distributed caching
            redis_host: Redis host
            redis_port: Redis port
        """
        self.local_cache = LRUCache(max_size, max_memory_mb)
        self.ttl_seconds = ttl_seconds
        self.enable_distributed = enable_distributed and HAS_REDIS
        
        if self.enable_distributed:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
                self.redis_client.ping()  # Test connection
            except Exception as e:
                warnings.warn(f"Redis connection failed: {e}")
                self.enable_distributed = False
    
    def _compute_inference_hash(self,
                              input_data: Union[torch.Tensor, np.ndarray],
                              model_signature: str) -> str:
        """Compute hash for inference input and model."""
        if isinstance(input_data, torch.Tensor):
            input_bytes = input_data.cpu().numpy().tobytes()
        else:
            input_bytes = input_data.tobytes()
        
        input_hash = hashlib.md5(input_bytes).hexdigest()
        return f"inference_{model_signature}_{input_hash}"
    
    def get_inference_result(self,
                           input_data: Union[torch.Tensor, np.ndarray],
                           model_signature: str) -> Optional[Any]:
        """Get cached inference result.
        
        Args:
            input_data: Model input
            model_signature: Unique model identifier
            
        Returns:
            Cached result or None
        """
        cache_key = self._compute_inference_hash(input_data, model_signature)
        
        # Try local cache first
        result = self.local_cache.get(cache_key)
        if result is not None:
            # Check TTL
            cached_data, timestamp = result
            if time.time() - timestamp < self.ttl_seconds:
                return cached_data
            else:
                # Expired, remove from cache
                self.local_cache.cache.pop(cache_key, None)
        
        # Try distributed cache
        if self.enable_distributed:
            try:
                cached_bytes = self.redis_client.get(cache_key)
                if cached_bytes:
                    cached_data, timestamp = pickle.loads(cached_bytes)
                    if time.time() - timestamp < self.ttl_seconds:
                        # Add back to local cache
                        self.local_cache.put(cache_key, (cached_data, timestamp))
                        return cached_data
                    else:
                        # Expired, remove from Redis
                        self.redis_client.delete(cache_key)
            except Exception as e:
                warnings.warn(f"Redis get failed: {e}")
        
        return None
    
    def put_inference_result(self,
                           input_data: Union[torch.Tensor, np.ndarray],
                           model_signature: str,
                           result: Any):
        """Cache inference result.
        
        Args:
            input_data: Model input
            model_signature: Unique model identifier
            result: Inference result
        """
        cache_key = self._compute_inference_hash(input_data, model_signature)
        timestamp = time.time()
        cached_item = (result, timestamp)
        
        # Store in local cache
        self.local_cache.put(cache_key, cached_item)
        
        # Store in distributed cache
        if self.enable_distributed:
            try:
                cached_bytes = pickle.dumps(cached_item)
                self.redis_client.setex(
                    cache_key, 
                    int(self.ttl_seconds), 
                    cached_bytes
                )
            except Exception as e:
                warnings.warn(f"Redis put failed: {e}")
    
    def invalidate_model_cache(self, model_signature: str):
        """Invalidate all cache entries for a specific model."""
        # Local cache
        keys_to_remove = [
            key for key in self.local_cache.cache.keys()
            if key.startswith(f"inference_{model_signature}_")
        ]
        
        for key in keys_to_remove:
            self.local_cache.cache.pop(key, None)
        
        # Distributed cache
        if self.enable_distributed:
            try:
                pattern = f"inference_{model_signature}_*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                warnings.warn(f"Redis invalidation failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'local_cache': self.local_cache.get_stats(),
            'distributed_enabled': self.enable_distributed,
            'ttl_seconds': self.ttl_seconds
        }
        
        if self.enable_distributed:
            try:
                info = self.redis_client.info('memory')
                stats['redis_memory_mb'] = info.get('used_memory', 0) / (1024 ** 2)
                stats['redis_keys'] = self.redis_client.dbsize()
            except Exception as e:
                warnings.warn(f"Failed to get Redis stats: {e}")
        
        return stats


class AdaptiveCache:
    """Adaptive cache that learns from access patterns."""
    
    def __init__(self, 
                 initial_size: int = 100,
                 max_size: int = 1000,
                 adaptation_interval: int = 100):
        """Initialize adaptive cache.
        
        Args:
            initial_size: Initial cache size
            max_size: Maximum cache size
            adaptation_interval: How often to adapt cache parameters
        """
        self.cache = LRUCache(initial_size)
        self.max_size = max_size
        self.adaptation_interval = adaptation_interval
        self.access_count = 0
        self.access_patterns = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get item with pattern learning."""
        self.access_count += 1
        
        # Track access patterns
        if key not in self.access_patterns:
            self.access_patterns[key] = {'count': 0, 'last_access': time.time()}
        
        self.access_patterns[key]['count'] += 1
        self.access_patterns[key]['last_access'] = time.time()
        
        # Adapt cache if needed
        if self.access_count % self.adaptation_interval == 0:
            self._adapt_cache_size()
        
        return self.cache.get(key)
    
    def put(self, key: str, value: Any):
        """Put item in adaptive cache."""
        self.cache.put(key, value)
    
    def _adapt_cache_size(self):
        """Adapt cache size based on access patterns."""
        current_hit_rate = self.cache.stats.hit_rate
        
        # Analyze access patterns
        hot_keys = len([
            key for key, stats in self.access_patterns.items()
            if stats['count'] > 5 and time.time() - stats['last_access'] < 300
        ])
        
        # Adjust cache size
        if current_hit_rate < 0.7 and hot_keys > self.cache.max_size * 0.8:
            # Increase cache size if hit rate is low and we have many hot keys
            new_size = min(int(self.cache.max_size * 1.2), self.max_size)
            self.cache.max_size = new_size
        elif current_hit_rate > 0.9 and hot_keys < self.cache.max_size * 0.5:
            # Decrease cache size if hit rate is high and we have few hot keys
            new_size = max(int(self.cache.max_size * 0.8), 50)
            self.cache.max_size = new_size