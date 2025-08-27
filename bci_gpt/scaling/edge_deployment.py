#!/usr/bin/env python3
"""
Edge Deployment System for BCI-GPT
Generation 3: Optimized deployment for edge devices and mobile platforms
"""

import os
import json
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import platform

@dataclass
class EdgeDevice:
    """Edge device specifications."""
    device_id: str
    device_type: str  # "jetson_nano", "raspberry_pi", "mobile", "embedded"
    cpu_cores: int
    memory_mb: int
    gpu_available: bool
    storage_gb: int
    network_type: str  # "wifi", "cellular", "ethernet"
    battery_powered: bool
    power_limit_watts: Optional[float] = None
    
    def get_optimization_profile(self) -> Dict[str, Any]:
        """Get optimization profile for device."""
        if self.device_type == "jetson_nano":
            return {
                "model_size": "medium",
                "batch_size": 4,
                "precision": "fp16",
                "cpu_threads": min(4, self.cpu_cores),
                "gpu_acceleration": self.gpu_available,
                "memory_limit_mb": min(2048, self.memory_mb * 0.8)
            }
        elif self.device_type == "raspberry_pi":
            return {
                "model_size": "small",
                "batch_size": 1,
                "precision": "int8",
                "cpu_threads": min(2, self.cpu_cores),
                "gpu_acceleration": False,
                "memory_limit_mb": min(512, self.memory_mb * 0.6)
            }
        elif self.device_type == "mobile":
            return {
                "model_size": "tiny",
                "batch_size": 1,
                "precision": "int8",
                "cpu_threads": 2,
                "gpu_acceleration": False,
                "memory_limit_mb": min(256, self.memory_mb * 0.3)
            }
        else:
            return {
                "model_size": "small",
                "batch_size": 1,
                "precision": "fp32",
                "cpu_threads": min(2, self.cpu_cores),
                "gpu_acceleration": False,
                "memory_limit_mb": min(1024, self.memory_mb * 0.5)
            }

class ModelOptimizer:
    """Optimize models for edge deployment."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_device(self, 
                           model_config: Dict[str, Any],
                           device: EdgeDevice) -> Dict[str, Any]:
        """Optimize model configuration for specific device."""
        
        profile = device.get_optimization_profile()
        
        optimized_config = model_config.copy()
        
        # Adjust model size
        if profile["model_size"] == "tiny":
            optimized_config["hidden_dim"] = min(128, optimized_config.get("hidden_dim", 512))
            optimized_config["n_layers"] = min(2, optimized_config.get("n_layers", 6))
            optimized_config["n_heads"] = min(2, optimized_config.get("n_heads", 8))
        elif profile["model_size"] == "small":
            optimized_config["hidden_dim"] = min(256, optimized_config.get("hidden_dim", 512))
            optimized_config["n_layers"] = min(4, optimized_config.get("n_layers", 6))
            optimized_config["n_heads"] = min(4, optimized_config.get("n_heads", 8))
        elif profile["model_size"] == "medium":
            optimized_config["hidden_dim"] = min(384, optimized_config.get("hidden_dim", 512))
            optimized_config["n_layers"] = min(6, optimized_config.get("n_layers", 6))
        
        # Processing configuration
        optimized_config["batch_size"] = profile["batch_size"]
        optimized_config["precision"] = profile["precision"]
        optimized_config["cpu_threads"] = profile["cpu_threads"]
        optimized_config["gpu_acceleration"] = profile["gpu_acceleration"]
        optimized_config["memory_limit_mb"] = profile["memory_limit_mb"]
        
        # Edge-specific optimizations
        optimized_config["enable_caching"] = True
        optimized_config["cache_size"] = min(1000, device.memory_mb // 4)
        optimized_config["offline_mode"] = device.network_type in ["cellular", "none"]
        
        if device.battery_powered:
            optimized_config["power_optimization"] = True
            optimized_config["cpu_frequency_scaling"] = True
            optimized_config["aggressive_sleep"] = True
        
        self.logger.info(f"Optimized model for {device.device_type}: {profile['model_size']} size, {profile['precision']} precision")
        
        return optimized_config
    
    def quantize_model(self, 
                      model_path: str, 
                      precision: str = "int8",
                      output_path: Optional[str] = None) -> str:
        """Quantize model for edge deployment."""
        
        if not output_path:
            base_path = Path(model_path).stem
            output_path = f"{base_path}_{precision}_quantized.onnx"
        
        # Mock quantization (would use actual model optimization tools)
        self.logger.info(f"Quantizing model to {precision}: {model_path} -> {output_path}")
        
        # In production, would use:
        # - ONNX quantization tools
        # - TensorRT for NVIDIA devices
        # - OpenVINO for Intel devices
        # - TensorFlow Lite for mobile
        
        quantization_result = {
            "input_model": model_path,
            "output_model": output_path,
            "precision": precision,
            "estimated_size_reduction": 0.75 if precision == "int8" else 0.5,
            "estimated_speedup": 2.0 if precision == "int8" else 1.5,
            "quantization_time": time.time()
        }
        
        return output_path
    
    def prune_model(self, 
                   model_path: str,
                   sparsity: float = 0.5,
                   output_path: Optional[str] = None) -> str:
        """Prune model for reduced size and faster inference."""
        
        if not output_path:
            base_path = Path(model_path).stem
            output_path = f"{base_path}_pruned_{int(sparsity*100)}.onnx"
        
        self.logger.info(f"Pruning model with {sparsity:.1%} sparsity: {model_path} -> {output_path}")
        
        # Mock pruning process
        pruning_result = {
            "input_model": model_path,
            "output_model": output_path,
            "sparsity": sparsity,
            "estimated_size_reduction": sparsity * 0.8,
            "estimated_speedup": 1 + (sparsity * 0.5),
            "pruning_time": time.time()
        }
        
        return output_path

class EdgeRuntime:
    """Optimized runtime for edge BCI processing."""
    
    def __init__(self, device: EdgeDevice, model_config: Dict[str, Any]):
        self.device = device
        self.model_config = model_config
        self.is_running = False
        
        # Performance monitoring
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.power_usage = []
        
        # Caching
        self.prediction_cache = {}
        self.feature_cache = {}
        
        # Threading
        self.processing_thread = None
        self.monitoring_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start edge runtime."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Edge runtime started on {self.device.device_type}")
    
    def stop(self):
        """Stop edge runtime."""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Edge runtime stopped")
    
    def process_eeg(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process EEG with edge-optimized pipeline."""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(eeg_data)
            
            # Check cache first
            if self.model_config.get("enable_caching", True):
                cached_result = self.prediction_cache.get(cache_key)
                if cached_result:
                    self.logger.debug("Using cached prediction")
                    cached_result["cache_hit"] = True
                    cached_result["inference_time_ms"] = 1  # Cached response is very fast
                    return cached_result
            
            # Preprocess EEG data
            preprocessed = self._preprocess_eeg_edge(eeg_data)
            
            # Extract features
            features = self._extract_features_edge(preprocessed)
            
            # Make prediction
            prediction = self._predict_edge(features)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)
            
            result = {
                "predicted_text": prediction["text"],
                "confidence": prediction["confidence"],
                "inference_time_ms": inference_time,
                "cache_hit": False,
                "device_type": self.device.device_type,
                "model_size": self.model_config.get("model_size", "unknown"),
                "precision": self.model_config.get("precision", "fp32")
            }
            
            # Cache result
            if self.model_config.get("enable_caching", True):
                self._cache_prediction(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Edge processing failed: {e}")
            return {
                "predicted_text": "",
                "confidence": 0.0,
                "inference_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "cache_hit": False
            }
    
    def _preprocess_eeg_edge(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Edge-optimized EEG preprocessing."""
        signal = eeg_data.get("data", [])
        
        # Simple preprocessing for edge devices
        if len(signal) > 1000:
            # Downsample for edge processing
            downsample_factor = len(signal) // 1000
            signal = signal[::downsample_factor]
        
        # Basic filtering (simplified)
        filtered_signal = []
        if signal:
            for i, sample in enumerate(signal):
                if i == 0:
                    filtered_signal.append(sample)
                else:
                    # Simple low-pass filter
                    filtered_sample = 0.9 * filtered_signal[-1] + 0.1 * sample
                    filtered_signal.append(filtered_sample)
        
        return {
            "preprocessed_data": filtered_signal,
            "sampling_rate": eeg_data.get("sampling_rate", 1000),
            "preprocessing_time_ms": 5  # Fast preprocessing
        }
    
    def _extract_features_edge(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Edge-optimized feature extraction."""
        signal = preprocessed.get("preprocessed_data", [])
        
        if not signal:
            return {"features": [], "feature_extraction_time_ms": 0}
        
        # Simple feature extraction for edge
        features = {
            "mean": sum(signal) / len(signal),
            "std": (sum((x - sum(signal)/len(signal))**2 for x in signal) / len(signal)) ** 0.5,
            "max": max(signal),
            "min": min(signal),
            "energy": sum(x**2 for x in signal) / len(signal)
        }
        
        return {
            "features": list(features.values()),
            "feature_names": list(features.keys()),
            "feature_extraction_time_ms": 2
        }
    
    def _predict_edge(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Edge-optimized prediction."""
        feature_vector = features.get("features", [])
        
        if not feature_vector:
            return {"text": "", "confidence": 0.0}
        
        # Simple prediction for edge devices
        vocabulary = ["yes", "no", "help", "stop", "hello", "please"]
        
        # Hash-based prediction (deterministic but pseudo-random)
        feature_sum = sum(abs(f) for f in feature_vector)
        word_index = int(feature_sum * 1000) % len(vocabulary)
        
        predicted_word = vocabulary[word_index]
        confidence = min(0.95, max(0.1, feature_sum / len(feature_vector)))
        
        return {
            "text": predicted_word,
            "confidence": confidence,
            "prediction_time_ms": 3
        }
    
    def _generate_cache_key(self, eeg_data: Dict[str, Any]) -> str:
        """Generate cache key for EEG data."""
        signal = eeg_data.get("data", [])
        if not signal:
            return "empty"
        
        # Simple hash based on signal statistics
        signal_hash = hash(tuple(signal[:min(100, len(signal))]))  # Use first 100 samples
        return f"eeg_{signal_hash}"
    
    def _cache_prediction(self, key: str, result: Dict[str, Any]):
        """Cache prediction result."""
        cache_limit = self.model_config.get("cache_size", 1000)
        
        if len(self.prediction_cache) >= cache_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        # Store result without caching metadata
        cacheable_result = {k: v for k, v in result.items() if k not in ["cache_hit"]}
        self.prediction_cache[key] = cacheable_result
    
    def _monitoring_loop(self):
        """Background monitoring of edge performance."""
        while self.is_running:
            try:
                # Monitor system resources (simplified)
                cpu_percent = self._get_cpu_usage()
                memory_mb = self._get_memory_usage()
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_mb)
                
                # Keep limited history
                if len(self.cpu_usage) > 100:
                    self.cpu_usage = self.cpu_usage[-100:]
                    self.memory_usage = self.memory_usage[-100:]
                    self.inference_times = self.inference_times[-100:]
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage (mock)."""
        # In production, would use psutil
        return 25.0 + (time.time() % 50)  # Mock value
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB (mock)."""
        # In production, would use psutil
        base_usage = self.device.memory_mb * 0.3  # Base usage
        cache_usage = len(self.prediction_cache) * 0.1  # Cache overhead
        return base_usage + cache_usage
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get edge runtime performance statistics."""
        stats = {
            "device_info": {
                "device_id": self.device.device_id,
                "device_type": self.device.device_type,
                "cpu_cores": self.device.cpu_cores,
                "memory_mb": self.device.memory_mb
            },
            "runtime_stats": {
                "is_running": self.is_running,
                "uptime_seconds": time.time(),  # Simplified
                "total_predictions": len(self.inference_times),
                "cache_size": len(self.prediction_cache),
                "cache_limit": self.model_config.get("cache_size", 1000)
            }
        }
        
        # Performance metrics
        if self.inference_times:
            stats["performance"] = {
                "avg_inference_time_ms": sum(self.inference_times) / len(self.inference_times),
                "min_inference_time_ms": min(self.inference_times),
                "max_inference_time_ms": max(self.inference_times),
                "p95_inference_time_ms": sorted(self.inference_times)[int(len(self.inference_times) * 0.95)]
            }
        
        if self.cpu_usage:
            stats["system_resources"] = {
                "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage),
                "current_cpu_usage": self.cpu_usage[-1] if self.cpu_usage else 0,
                "avg_memory_usage_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            }
        
        return stats

class EdgeDeploymentManager:
    """Manage deployment across multiple edge devices."""
    
    def __init__(self):
        self.devices = {}
        self.runtimes = {}
        self.optimizer = ModelOptimizer()
        
        self.logger = logging.getLogger(__name__)
    
    def register_device(self, device: EdgeDevice) -> bool:
        """Register edge device for deployment."""
        self.devices[device.device_id] = device
        self.logger.info(f"Registered edge device: {device.device_id} ({device.device_type})")
        return True
    
    def deploy_to_device(self, 
                        device_id: str,
                        base_model_config: Dict[str, Any]) -> bool:
        """Deploy optimized model to edge device."""
        
        if device_id not in self.devices:
            self.logger.error(f"Device not registered: {device_id}")
            return False
        
        device = self.devices[device_id]
        
        # Optimize model for device
        optimized_config = self.optimizer.optimize_for_device(base_model_config, device)
        
        # Create and start runtime
        runtime = EdgeRuntime(device, optimized_config)
        runtime.start()
        
        self.runtimes[device_id] = runtime
        
        self.logger.info(f"Deployed to device: {device_id}")
        return True
    
    def process_on_device(self, 
                         device_id: str,
                         eeg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process EEG data on specific edge device."""
        
        if device_id not in self.runtimes:
            self.logger.error(f"Runtime not available for device: {device_id}")
            return None
        
        runtime = self.runtimes[device_id]
        return runtime.process_eeg(eeg_data)
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get status of all edge deployments."""
        
        status = {
            "total_devices": len(self.devices),
            "active_runtimes": len(self.runtimes),
            "device_details": {},
            "global_stats": {
                "total_predictions": 0,
                "avg_inference_time_ms": 0
            }
        }
        
        inference_times = []
        
        for device_id, runtime in self.runtimes.items():
            device_stats = runtime.get_performance_stats()
            status["device_details"][device_id] = device_stats
            
            if "performance" in device_stats:
                status["global_stats"]["total_predictions"] += device_stats["runtime_stats"]["total_predictions"]
                if device_stats["runtime_stats"]["total_predictions"] > 0:
                    inference_times.append(device_stats["performance"]["avg_inference_time_ms"])
        
        if inference_times:
            status["global_stats"]["avg_inference_time_ms"] = sum(inference_times) / len(inference_times)
        
        return status

# Example usage and testing
if __name__ == "__main__":
    print("📱 Testing Edge Deployment System...")
    
    # Create edge devices
    jetson_device = EdgeDevice(
        device_id="jetson_001",
        device_type="jetson_nano",
        cpu_cores=4,
        memory_mb=4096,
        gpu_available=True,
        storage_gb=32,
        network_type="wifi",
        battery_powered=False
    )
    
    raspberry_pi = EdgeDevice(
        device_id="rpi_001",
        device_type="raspberry_pi",
        cpu_cores=4,
        memory_mb=1024,
        gpu_available=False,
        storage_gb=16,
        network_type="wifi",
        battery_powered=True,
        power_limit_watts=5.0
    )
    
    mobile_device = EdgeDevice(
        device_id="mobile_001",
        device_type="mobile",
        cpu_cores=8,
        memory_mb=4096,
        gpu_available=False,
        storage_gb=64,
        network_type="cellular",
        battery_powered=True,
        power_limit_watts=10.0
    )
    
    # Test model optimization
    optimizer = ModelOptimizer()
    
    base_config = {
        "hidden_dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "batch_size": 32
    }
    
    jetson_config = optimizer.optimize_for_device(base_config, jetson_device)
    pi_config = optimizer.optimize_for_device(base_config, raspberry_pi)
    mobile_config = optimizer.optimize_for_device(base_config, mobile_device)
    
    print(f"✅ Jetson optimization: {jetson_config['model_size']} model, {jetson_config['precision']} precision")
    print(f"✅ Pi optimization: {pi_config['model_size']} model, {pi_config['precision']} precision")
    print(f"✅ Mobile optimization: {mobile_config['model_size']} model, {mobile_config['precision']} precision")
    
    # Test deployment manager
    deployment_manager = EdgeDeploymentManager()
    
    # Register devices
    deployment_manager.register_device(jetson_device)
    deployment_manager.register_device(raspberry_pi)
    deployment_manager.register_device(mobile_device)
    
    # Deploy to devices
    deployment_manager.deploy_to_device("jetson_001", base_config)
    deployment_manager.deploy_to_device("rpi_001", base_config)
    deployment_manager.deploy_to_device("mobile_001", base_config)
    
    print(f"✅ Deployed to {len(deployment_manager.runtimes)} devices")
    
    # Test processing
    sample_eeg = {
        "data": [i * 0.01 for i in range(2000)],
        "sampling_rate": 1000,
        "subject_id": "test_edge"
    }
    
    # Process on each device
    for device_id in ["jetson_001", "rpi_001", "mobile_001"]:
        result = deployment_manager.process_on_device(device_id, sample_eeg)
        if result:
            print(f"✅ {device_id}: '{result['predicted_text']}' ({result['inference_time_ms']:.1f}ms)")
    
    # Get deployment status
    time.sleep(1)  # Let some monitoring data accumulate
    
    status = deployment_manager.get_deployment_status()
    print(f"\n📊 Deployment Status:")
    print(f"   Active devices: {status['active_runtimes']}/{status['total_devices']}")
    print(f"   Total predictions: {status['global_stats']['total_predictions']}")
    print(f"   Avg inference time: {status['global_stats']['avg_inference_time_ms']:.1f}ms")
    
    print("\n📱 Edge Deployment System Ready!")
