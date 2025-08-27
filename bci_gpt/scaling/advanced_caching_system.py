#!/usr/bin/env python3
"""
Advanced Multi-Tier Caching System for BCI-GPT
Generation 3: High-performance caching with intelligent eviction
"""

import time
import json
import hashlib
import threading
import pickle
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio
from pathlib import Path

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[timedelta] = None,
                 cleanup_interval: int = 300):  # 5 minutes
        
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Background cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_active = True
        self.cleanup_thread.start()
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired():
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                self.misses += 1
                return None
            
            # Update access tracking
            entry.update_access()
            self.access_order.move_to_end(key)
            self.hits += 1
            
            return entry.value
    
    def set(self, 
            key: str, 
            value: Any, 
            ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = sys.getsizeof(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store entry
            self.cache[key] = entry
            self.access_order[key] = True
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.access_order:
            return
        
        # Remove oldest entry
        oldest_key = next(iter(self.access_order))
        del self.cache[oldest_key]
        del self.access_order[oldest_key]
        self.evictions += 1
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while self.cleanup_active:
            try:
                with self.lock:
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                        if key in self.access_order:
                            del self.access_order[key]
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)  # Back off on errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "total_size_bytes": total_size,
                "avg_entry_size": total_size / max(len(self.cache), 1)
            }

class ModelCache(MemoryCache):
    """Specialized cache for BCI-GPT model predictions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prediction_cache = {}
        self.model_weights_cache = {}
    
    def cache_prediction(self, 
                        eeg_hash: str, 
                        prediction: Dict[str, Any],
                        ttl: timedelta = timedelta(hours=1)) -> str:
        """Cache model prediction with EEG signal hash."""
        cache_key = f"pred_{eeg_hash}"
        
        # Add prediction metadata
        cached_prediction = {
            "prediction": prediction,
            "cached_at": datetime.now().isoformat(),
            "eeg_hash": eeg_hash
        }
        
        self.set(cache_key, cached_prediction, ttl)
        return cache_key
    
    def get_cached_prediction(self, eeg_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction by EEG hash."""
        cache_key = f"pred_{eeg_hash}"
        return self.get(cache_key)
    
    def cache_model_weights(self, 
                           model_id: str, 
                           weights: Dict[str, Any],
                           ttl: timedelta = timedelta(days=1)) -> str:
        """Cache model weights for quick loading."""
        cache_key = f"model_{model_id}"
        
        cached_weights = {
            "weights": weights,
            "model_id": model_id,
            "cached_at": datetime.now().isoformat()
        }
        
        self.set(cache_key, cached_weights, ttl)
        return cache_key
    
    def get_cached_weights(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get cached model weights."""
        cache_key = f"model_{model_id}"
        return self.get(cache_key)

class DistributedCache:
    """Distributed cache for multi-node deployments."""
    
    def __init__(self, 
                 nodes: List[str] = None,
                 replication_factor: int = 2):
        
        self.nodes = nodes or ["localhost:11211"]
        self.replication_factor = min(replication_factor, len(self.nodes))
        self.local_cache = MemoryCache()
        
        self.logger = logging.getLogger(__name__)
    
    def _get_node_for_key(self, key: str) -> List[str]:
        """Get nodes responsible for key using consistent hashing."""
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Simple consistent hashing
        node_index = key_hash % len(self.nodes)
        
        # Return replication_factor nodes
        selected_nodes = []
        for i in range(self.replication_factor):
            idx = (node_index + i) % len(self.nodes)
            selected_nodes.append(self.nodes[idx])
        
        return selected_nodes
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        local_value = self.local_cache.get(key)
        if local_value is not None:
            return local_value
        
        # Try remote nodes
        nodes = self._get_node_for_key(key)
        for node in nodes:
            try:
                value = self._get_from_node(node, key)
                if value is not None:
                    # Cache locally for faster future access
                    self.local_cache.set(key, value, timedelta(minutes=5))
                    return value
            except Exception as e:
                self.logger.warning(f"Failed to get from node {node}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in distributed cache."""
        # Set in local cache
        self.local_cache.set(key, value, ttl)
        
        # Set in remote nodes
        nodes = self._get_node_for_key(key)
        success_count = 0
        
        for node in nodes:
            try:
                if self._set_to_node(node, key, value, ttl):
                    success_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to set to node {node}: {e}")
        
        # Consider successful if at least one replica was stored
        return success_count > 0
    
    def _get_from_node(self, node: str, key: str) -> Optional[Any]:
        """Get value from specific node (mock implementation)."""
        # In production, would use actual distributed cache like Redis
        return None
    
    def _set_to_node(self, node: str, key: str, value: Any, ttl: Optional[timedelta]) -> bool:
        """Set value to specific node (mock implementation)."""
        # In production, would use actual distributed cache like Redis
        return True

class AdaptiveCacheManager:
    """Intelligent cache manager with adaptive sizing and warming."""
    
    def __init__(self):
        self.caches = {
            "predictions": ModelCache(max_size=10000, default_ttl=timedelta(hours=1)),
            "eeg_features": MemoryCache(max_size=50000, default_ttl=timedelta(minutes=30)),
            "model_weights": MemoryCache(max_size=100, default_ttl=timedelta(hours=24)),
            "user_sessions": MemoryCache(max_size=1000, default_ttl=timedelta(hours=2))
        }
        
        self.cache_stats = {}
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_active = True
        self.optimization_thread.start()
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get value from named cache."""
        if cache_name in self.caches:
            return self.caches[cache_name].get(key)
        return None
    
    def set(self, cache_name: str, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in named cache."""
        if cache_name in self.caches:
            return self.caches[cache_name].set(key, value, ttl)
        return False
    
    def warm_cache(self, predictions: List[Tuple[str, Dict[str, Any]]]):
        """Warm up prediction cache with common predictions."""
        self.logger.info(f"Warming cache with {len(predictions)} predictions")
        
        for eeg_hash, prediction in predictions:
            self.caches["predictions"].cache_prediction(eeg_hash, prediction)
    
    def _optimization_loop(self):
        """Background optimization of cache performance."""
        while self.optimization_active:
            try:
                # Collect cache statistics
                for name, cache in self.caches.items():
                    stats = cache.get_stats()
                    self.cache_stats[name] = stats
                    
                    # Adaptive sizing based on hit rate
                    if stats["hit_rate"] < 0.5 and stats["entries"] == stats["max_size"]:
                        # Low hit rate, increase cache size
                        new_size = min(stats["max_size"] * 2, 100000)
                        self.logger.info(f"Increasing {name} cache size to {new_size}")
                        cache.max_size = new_size
                    
                    elif stats["hit_rate"] > 0.9 and stats["entries"] < stats["max_size"] * 0.5:
                        # High hit rate but low utilization, decrease size
                        new_size = max(stats["max_size"] // 2, 100)
                        self.logger.info(f"Decreasing {name} cache size to {new_size}")
                        cache.max_size = new_size
                
                time.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache optimization error: {e}")
                time.sleep(60)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "cache_stats": self.cache_stats,
            "total_caches": len(self.caches),
            "optimization_active": self.optimization_active
        }

# Example usage and testing
if __name__ == "__main__":
    print("⚡ Testing Advanced Caching System...")
    
    # Test basic memory cache
    cache = MemoryCache(max_size=5)
    
    # Test basic operations
    cache.set("key1", "value1")
    cache.set("key2", {"data": "complex_value"})
    
    print(f"✅ Cache get key1: {cache.get('key1')}")
    print(f"✅ Cache get key2: {cache.get('key2')}")
    print(f"❌ Cache get missing: {cache.get('missing')}")
    
    # Test LRU eviction
    for i in range(10):
        cache.set(f"key{i}", f"value{i}")
    
    stats = cache.get_stats()
    print(f"✅ Cache stats: {stats['entries']} entries, {stats['hit_rate']:.2%} hit rate")
    
    # Test model cache
    model_cache = ModelCache()
    
    sample_prediction = {
        "text": "hello world",
        "confidence": 0.85,
        "latency_ms": 45
    }
    
    eeg_hash = hashlib.md5(b"sample_eeg_data").hexdigest()
    cache_key = model_cache.cache_prediction(eeg_hash, sample_prediction)
    
    cached_pred = model_cache.get_cached_prediction(eeg_hash)
    print(f"✅ Cached prediction: {cached_pred['prediction']['text']}")
    
    # Test adaptive cache manager
    cache_manager = AdaptiveCacheManager()
    
    # Test warming
    warm_predictions = [
        (hashlib.md5(f"eeg_{i}".encode()).hexdigest(), {"text": f"word_{i}", "confidence": 0.8})
        for i in range(5)
    ]
    
    cache_manager.warm_cache(warm_predictions)
    
    # Test retrieval
    test_hash = warm_predictions[0][0]
    warmed_pred = cache_manager.get("predictions", f"pred_{test_hash}")
    if warmed_pred:
        print(f"✅ Warmed cache retrieval: {warmed_pred['prediction']['text']}")
    
    # Get global stats
    global_stats = cache_manager.get_global_stats()
    print(f"✅ Global cache stats: {global_stats['total_caches']} caches active")
    
    print("\n🚀 Advanced Caching System Ready!")
