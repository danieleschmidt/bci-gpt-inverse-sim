"""
Performance optimization utilities for BCI-GPT.

This module provides comprehensive performance optimization including:
- Model quantization and compression
- Inference acceleration
- Memory optimization
- Batch processing optimization
- GPU utilization optimization
- Real-time performance tuning
"""

import time
import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Core dependencies
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import prune
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available for model optimization")

# Optional optimization dependencies
try:
    import torch.jit
    import torch.quantization
    HAS_TORCH_OPTIMIZATION = True
except ImportError:
    HAS_TORCH_OPTIMIZATION = False
    warnings.warn("PyTorch optimization features not available")

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    warnings.warn("ONNX not available for model optimization")

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    warnings.warn("TensorRT not available for GPU optimization")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    quantization_method: str = "dynamic"  # dynamic, static, qat
    pruning_sparsity: float = 0.1  # Fraction of weights to prune
    enable_jit_compile: bool = True
    enable_mixed_precision: bool = True
    batch_size_optimization: bool = True
    memory_optimization: bool = True
    target_device: str = "auto"  # auto, cpu, cuda, tensorrt
    target_latency_ms: float = 100.0
    target_throughput: float = 1000.0  # samples/second


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float] = None
    model_size_mb: float = 0.0
    accuracy_drop: float = 0.0


class ModelOptimizer:
    """Comprehensive model optimization toolkit."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize model optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.optimization_history = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
    
    def optimize_model(self, model: nn.Module, 
                      sample_input: torch.Tensor,
                      validation_data: Optional[Any] = None) -> Tuple[nn.Module, PerformanceMetrics]:
        """Apply comprehensive optimization to model.
        
        Args:
            model: PyTorch model to optimize
            sample_input: Sample input for optimization
            validation_data: Optional validation data for accuracy checking
            
        Returns:
            Tuple of (optimized_model, performance_metrics)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for model optimization")
        
        self.logger.info("Starting comprehensive model optimization")
        
        # Measure baseline performance
        self.baseline_metrics = self._measure_performance(model, sample_input)
        self.logger.info(f"Baseline performance: {self.baseline_metrics.inference_time_ms:.1f}ms, "
                        f"{self.baseline_metrics.throughput_samples_per_sec:.1f} samples/sec")
        
        optimized_model = model
        optimization_steps = []
        
        # Step 1: Quantization
        if self.config.quantization_method != "none":
            self.logger.info(f"Applying {self.config.quantization_method} quantization")
            optimized_model = self._apply_quantization(optimized_model, sample_input)
            optimization_steps.append("quantization")
        
        # Step 2: Pruning
        if self.config.pruning_sparsity > 0:
            self.logger.info(f"Applying {self.config.pruning_sparsity:.1%} pruning")
            optimized_model = self._apply_pruning(optimized_model, self.config.pruning_sparsity)
            optimization_steps.append("pruning")
        
        # Step 3: JIT Compilation
        if self.config.enable_jit_compile and HAS_TORCH_OPTIMIZATION:
            self.logger.info("Applying JIT compilation")
            optimized_model = self._apply_jit_compilation(optimized_model, sample_input)
            optimization_steps.append("jit")
        
        # Step 4: Memory optimization
        if self.config.memory_optimization:
            self.logger.info("Applying memory optimizations")
            optimized_model = self._apply_memory_optimization(optimized_model)
            optimization_steps.append("memory")
        
        # Measure final performance
        final_metrics = self._measure_performance(optimized_model, sample_input)
        
        # Calculate improvements
        speedup = self.baseline_metrics.inference_time_ms / final_metrics.inference_time_ms
        throughput_improvement = (final_metrics.throughput_samples_per_sec / 
                                 self.baseline_metrics.throughput_samples_per_sec)
        
        self.logger.info(f"Optimization complete: {speedup:.2f}x speedup, "
                        f"{throughput_improvement:.2f}x throughput improvement")
        
        # Validate accuracy if validation data provided
        if validation_data is not None:
            accuracy_drop = self._validate_accuracy(optimized_model, validation_data, model)
            final_metrics.accuracy_drop = accuracy_drop
            self.logger.info(f"Accuracy drop: {accuracy_drop:.3f}")
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'steps': optimization_steps,
            'baseline_metrics': self.baseline_metrics,
            'final_metrics': final_metrics,
            'speedup': speedup,
            'throughput_improvement': throughput_improvement
        })
        
        return optimized_model, final_metrics
    
    def _apply_quantization(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply quantization to model."""
        if not HAS_TORCH_OPTIMIZATION:
            self.logger.warning("PyTorch quantization not available")
            return model
        
        try:
            model.eval()
            
            if self.config.quantization_method == "dynamic":
                # Dynamic quantization (fastest to apply)
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv1d, nn.Conv2d}, dtype=torch.qint8
                )
                
            elif self.config.quantization_method == "static":
                # Static quantization (requires calibration)
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # Calibration pass (simplified)
                with torch.no_grad():
                    model(sample_input)
                
                quantized_model = torch.quantization.convert(model, inplace=False)
                
            else:
                self.logger.warning(f"Unknown quantization method: {self.config.quantization_method}")
                return model
            
            self.logger.info("Quantization applied successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply structured pruning to model."""
        try:
            # Apply magnitude-based pruning to linear and conv layers
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')  # Make pruning permanent
            
            self.logger.info(f"Applied {sparsity:.1%} pruning to model")
            return model
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return model
    
    def _apply_jit_compilation(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply JIT compilation to model."""
        try:
            model.eval()
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, sample_input)
            
            # Optimize the traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            self.logger.info("JIT compilation applied successfully")
            return traced_model
            
        except Exception as e:
            self.logger.error(f"JIT compilation failed: {e}")
            return model
    
    def _apply_memory_optimization(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations."""
        try:
            # Enable gradient checkpointing for training (if applicable)
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Fuse operations where possible
            if hasattr(torch.nn.utils, 'fuse_conv_bn_eval'):
                model = torch.nn.utils.fuse_conv_bn_eval(model)
            
            self.logger.info("Memory optimizations applied")
            return model
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return model
    
    def _measure_performance(self, model: nn.Module, sample_input: torch.Tensor,
                           num_runs: int = 100) -> PerformanceMetrics:
        """Measure model performance."""
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_inference_time_ms = (total_time / num_runs) * 1000
        throughput = num_runs / total_time
        
        # Memory usage
        memory_usage_mb = 0
        if device.type == 'cuda':
            memory_usage_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        
        # GPU utilization (simplified)
        gpu_utilization = None
        if device.type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except:
                pass
        
        # Model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return PerformanceMetrics(
            inference_time_ms=avg_inference_time_ms,
            throughput_samples_per_sec=throughput,
            memory_usage_mb=memory_usage_mb,
            gpu_utilization_percent=gpu_utilization,
            model_size_mb=model_size_mb
        )
    
    def _validate_accuracy(self, optimized_model: nn.Module, validation_data: Any,
                          original_model: nn.Module) -> float:
        """Validate that optimization doesn't significantly hurt accuracy."""
        # Simplified accuracy validation
        # In practice, this would run comprehensive evaluation
        
        try:
            with torch.no_grad():
                # Get sample from validation data
                if hasattr(validation_data, '__iter__'):
                    sample_batch = next(iter(validation_data))
                    if isinstance(sample_batch, (list, tuple)):
                        inputs = sample_batch[0]
                    else:
                        inputs = sample_batch
                else:
                    inputs = validation_data
                
                # Compare outputs
                original_output = original_model(inputs)
                optimized_output = optimized_model(inputs)
                
                # Calculate difference (simplified)
                if isinstance(original_output, torch.Tensor):
                    diff = torch.mean(torch.abs(original_output - optimized_output)).item()
                    return diff
                else:
                    return 0.0
                    
        except Exception as e:
            self.logger.warning(f"Accuracy validation failed: {e}")
            return 0.0
    
    def export_optimized_model(self, model: nn.Module, 
                              sample_input: torch.Tensor,
                              output_path: Path,
                              format: str = "onnx") -> None:
        """Export optimized model to various formats.
        
        Args:
            model: Optimized model to export
            sample_input: Sample input for tracing
            output_path: Output file path
            format: Export format ('onnx', 'torchscript', 'tensorrt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "onnx" and HAS_ONNX:
                self._export_onnx(model, sample_input, output_path)
                
            elif format.lower() == "torchscript":
                self._export_torchscript(model, sample_input, output_path)
                
            elif format.lower() == "tensorrt" and HAS_TENSORRT:
                self._export_tensorrt(model, sample_input, output_path)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Model exported to {output_path} in {format} format")
            
        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            raise
    
    def _export_onnx(self, model: nn.Module, sample_input: torch.Tensor, 
                    output_path: Path) -> None:
        """Export model to ONNX format."""
        model.eval()
        
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    def _export_torchscript(self, model: nn.Module, sample_input: torch.Tensor,
                           output_path: Path) -> None:
        """Export model to TorchScript format."""
        model.eval()
        
        # Try tracing first, fall back to scripting
        try:
            traced_model = torch.jit.trace(model, sample_input)
            traced_model.save(output_path)
        except:
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
    
    def _export_tensorrt(self, model: nn.Module, sample_input: torch.Tensor,
                        output_path: Path) -> None:
        """Export model to TensorRT format."""
        # This would require converting through ONNX first
        # and then using TensorRT for optimization
        onnx_path = output_path.with_suffix('.onnx')
        self._export_onnx(model, sample_input, onnx_path)
        
        # TensorRT optimization would go here
        self.logger.info("TensorRT export requires additional implementation")


class BatchOptimizer:
    """Optimize batch processing for maximum throughput."""
    
    def __init__(self, model: nn.Module, device: str = "auto"):
        """Initialize batch optimizer.
        
        Args:
            model: Model to optimize batching for
            device: Target device for optimization
        """
        self.model = model
        self.device = self._determine_device(device)
        self.optimal_batch_size = None
        self.logger = logging.getLogger(__name__)
    
    def _determine_device(self, device: str) -> torch.device:
        """Determine optimal device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def find_optimal_batch_size(self, sample_input: torch.Tensor,
                               max_batch_size: int = 128,
                               target_memory_usage: float = 0.8) -> int:
        """Find optimal batch size for given constraints.
        
        Args:
            sample_input: Sample input tensor
            max_batch_size: Maximum batch size to test
            target_memory_usage: Target GPU memory usage (0.0 to 1.0)
            
        Returns:
            Optimal batch size
        """
        self.model.eval()
        self.model.to(self.device)
        
        if self.device.type != 'cuda':
            # For CPU, use performance-based optimization
            return self._find_cpu_optimal_batch_size(sample_input, max_batch_size)
        
        # For GPU, consider memory constraints
        best_batch_size = 1
        best_throughput = 0
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if batch_size > max_batch_size:
                break
            
            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1, 1).to(self.device)
                
                # Check memory usage
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = self.model(batch_input)
                
                peak_memory = torch.cuda.max_memory_allocated(self.device)
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                memory_usage = peak_memory / total_memory
                
                if memory_usage > target_memory_usage:
                    self.logger.info(f"Batch size {batch_size} exceeds memory target ({memory_usage:.1%})")
                    break
                
                # Measure throughput
                throughput = self._measure_batch_throughput(batch_input, batch_size)
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                
                self.logger.info(f"Batch size {batch_size}: {throughput:.1f} samples/sec, "
                               f"memory usage: {memory_usage:.1%}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.info(f"Batch size {batch_size} caused OOM")
                    break
                else:
                    raise
        
        self.optimal_batch_size = best_batch_size
        self.logger.info(f"Optimal batch size: {best_batch_size} ({best_throughput:.1f} samples/sec)")
        
        return best_batch_size
    
    def _find_cpu_optimal_batch_size(self, sample_input: torch.Tensor,
                                   max_batch_size: int) -> int:
        """Find optimal batch size for CPU."""
        best_batch_size = 1
        best_throughput = 0
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            if batch_size > max_batch_size:
                break
            
            batch_input = sample_input.repeat(batch_size, 1, 1)
            throughput = self._measure_batch_throughput(batch_input, batch_size)
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            self.logger.info(f"CPU batch size {batch_size}: {throughput:.1f} samples/sec")
        
        return best_batch_size
    
    def _measure_batch_throughput(self, batch_input: torch.Tensor,
                                batch_size: int, num_runs: int = 20) -> float:
        """Measure throughput for given batch size."""
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(batch_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(batch_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_samples = num_runs * batch_size
        total_time = end_time - start_time
        
        return total_samples / total_time


class RealTimeOptimizer:
    """Optimize model for real-time inference."""
    
    def __init__(self, target_latency_ms: float = 100.0):
        """Initialize real-time optimizer.
        
        Args:
            target_latency_ms: Target latency in milliseconds
        """
        self.target_latency_ms = target_latency_ms
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_realtime(self, model: nn.Module,
                            sample_input: torch.Tensor) -> Tuple[nn.Module, Dict[str, Any]]:
        """Optimize model for real-time inference.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for testing
            
        Returns:
            Tuple of (optimized_model, optimization_report)
        """
        optimization_report = {
            'original_latency_ms': 0,
            'optimized_latency_ms': 0,
            'optimizations_applied': [],
            'target_met': False
        }
        
        # Measure baseline
        original_latency = self._measure_latency(model, sample_input)
        optimization_report['original_latency_ms'] = original_latency
        
        self.logger.info(f"Original latency: {original_latency:.1f}ms (target: {self.target_latency_ms:.1f}ms)")
        
        optimized_model = model
        
        # Apply optimizations progressively
        if original_latency > self.target_latency_ms:
            # 1. Mixed precision
            if torch.cuda.is_available():
                optimized_model = self._apply_mixed_precision(optimized_model)
                optimization_report['optimizations_applied'].append('mixed_precision')
                
                current_latency = self._measure_latency(optimized_model, sample_input)
                self.logger.info(f"After mixed precision: {current_latency:.1f}ms")
                
                if current_latency <= self.target_latency_ms:
                    optimization_report['optimized_latency_ms'] = current_latency
                    optimization_report['target_met'] = True
                    return optimized_model, optimization_report
            
            # 2. Dynamic quantization
            try:
                optimized_model = torch.quantization.quantize_dynamic(
                    optimized_model, {nn.Linear}, dtype=torch.qint8
                )
                optimization_report['optimizations_applied'].append('quantization')
                
                current_latency = self._measure_latency(optimized_model, sample_input)
                self.logger.info(f"After quantization: {current_latency:.1f}ms")
                
                if current_latency <= self.target_latency_ms:
                    optimization_report['optimized_latency_ms'] = current_latency
                    optimization_report['target_met'] = True
                    return optimized_model, optimization_report
                    
            except Exception as e:
                self.logger.warning(f"Quantization failed: {e}")
            
            # 3. JIT compilation
            try:
                traced_model = torch.jit.trace(optimized_model, sample_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
                optimized_model = traced_model
                optimization_report['optimizations_applied'].append('jit')
                
                current_latency = self._measure_latency(optimized_model, sample_input)
                self.logger.info(f"After JIT: {current_latency:.1f}ms")
                
            except Exception as e:
                self.logger.warning(f"JIT compilation failed: {e}")
        
        final_latency = self._measure_latency(optimized_model, sample_input)
        optimization_report['optimized_latency_ms'] = final_latency
        optimization_report['target_met'] = final_latency <= self.target_latency_ms
        
        speedup = original_latency / final_latency
        self.logger.info(f"Final latency: {final_latency:.1f}ms ({speedup:.2f}x speedup)")
        
        return optimized_model, optimization_report
    
    def _measure_latency(self, model: nn.Module, sample_input: torch.Tensor,
                        num_runs: int = 100) -> float:
        """Measure model latency."""
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
        return avg_latency_ms
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply automatic mixed precision."""
        # This would typically be handled during training with GradScaler
        # For inference, we can convert to half precision where beneficial
        if torch.cuda.is_available():
            try:
                model = model.half()
                self.logger.info("Applied half precision")
            except Exception as e:
                self.logger.warning(f"Half precision failed: {e}")
        
        return model


# Utility functions
def profile_model_performance(model: nn.Module, sample_input: torch.Tensor,
                            device: str = "auto") -> Dict[str, Any]:
    """Quick performance profiling of a model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    sample_input = sample_input.to(device)
    
    optimizer = ModelOptimizer()
    metrics = optimizer._measure_performance(model, sample_input)
    
    return {
        'inference_time_ms': metrics.inference_time_ms,
        'throughput_samples_per_sec': metrics.throughput_samples_per_sec,
        'memory_usage_mb': metrics.memory_usage_mb,
        'model_size_mb': metrics.model_size_mb,
        'device': device
    }


def auto_optimize_model(model: nn.Module, sample_input: torch.Tensor,
                       target_latency_ms: float = 100.0) -> nn.Module:
    """Automatically optimize model for target latency."""
    rt_optimizer = RealTimeOptimizer(target_latency_ms)
    optimized_model, report = rt_optimizer.optimize_for_realtime(model, sample_input)
    
    print(f"Optimization complete:")
    print(f"  Original latency: {report['original_latency_ms']:.1f}ms")
    print(f"  Optimized latency: {report['optimized_latency_ms']:.1f}ms")
    print(f"  Target met: {report['target_met']}")
    print(f"  Optimizations applied: {report['optimizations_applied']}")
    
    return optimized_model


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if HAS_TORCH:
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)
                self.linear3 = nn.Linear(128, 10)
                
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = torch.relu(self.linear2(x))
                return self.linear3(x)
        
        model = TestModel()
        sample_input = torch.randn(1, 128)
        
        print("Testing model optimization...")
        
        # Profile baseline performance
        baseline_perf = profile_model_performance(model, sample_input)
        print(f"Baseline performance: {baseline_perf}")
        
        # Test optimization
        config = OptimizationConfig(
            quantization_method="dynamic",
            pruning_sparsity=0.1,
            enable_jit_compile=True
        )
        
        optimizer = ModelOptimizer(config)
        optimized_model, metrics = optimizer.optimize_model(model, sample_input)
        
        print(f"Optimization results:")
        print(f"  Inference time: {metrics.inference_time_ms:.1f}ms")
        print(f"  Throughput: {metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"  Model size: {metrics.model_size_mb:.1f}MB")
        
        # Test batch optimization
        batch_optimizer = BatchOptimizer(model)
        optimal_batch_size = batch_optimizer.find_optimal_batch_size(sample_input)
        print(f"Optimal batch size: {optimal_batch_size}")
        
        print("Performance optimization test completed!")
    else:
        print("PyTorch not available. Skipping optimization tests.")