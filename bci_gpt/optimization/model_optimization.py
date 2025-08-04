"""Model optimization techniques for deployment and inference."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import warnings
import os
from pathlib import Path

try:
    import torch.quantization as quantization
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    warnings.warn("PyTorch quantization not available")

try:
    import tensorrt as trt
    import torch2trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    warnings.warn("TensorRT not available for optimization")


class ModelOptimizer:
    """Comprehensive model optimization for deployment."""
    
    def __init__(self):
        """Initialize model optimizer."""
        self.optimized_models = {}
        
    def quantize_model(self,
                      model: nn.Module,
                      method: str = "dynamic",
                      calibration_data: Optional[torch.utils.data.DataLoader] = None,
                      backend: str = "fbgemm") -> nn.Module:
        """Quantize model for reduced memory and faster inference.
        
        Args:
            model: PyTorch model to quantize
            method: Quantization method ("dynamic", "static", "qat")
            calibration_data: Data for calibration (required for static quantization)
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        if not HAS_QUANTIZATION:
            warnings.warn("Quantization not available, returning original model")
            return model
        
        model.eval()
        
        if method == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
            
        elif method == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            # Static quantization
            model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, (list, tuple)):
                        model(*batch)
                    else:
                        model(batch)
                    break  # Only need one batch for calibration
            
            quantized_model = torch.quantization.convert(model, inplace=False)
            
        elif method == "qat":
            # Quantization-aware training
            model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            quantized_model = torch.quantization.prepare_qat(model)
            
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return quantized_model
    
    def prune_model(self,
                   model: nn.Module,
                   sparsity: float = 0.5,
                   structured: bool = False) -> nn.Module:
        """Prune model weights for reduced size and faster inference.
        
        Args:
            model: Model to prune
            sparsity: Fraction of weights to prune (0-1)
            structured: Whether to use structured pruning
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            warnings.warn("Pruning not available, returning original model")
            return model
        
        pruned_model = model
        
        # Get all parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning (prune entire channels/filters)
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Conv1d):
                    prune.ln_structured(module, param_name, amount=sparsity, n=2, dim=0)
                elif isinstance(module, nn.Linear):
                    prune.ln_structured(module, param_name, amount=sparsity, n=2, dim=0)
        else:
            # Unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return pruned_model
    
    def optimize_for_inference(self,
                             model: nn.Module,
                             example_input: torch.Tensor,
                             optimization_level: str = "default") -> nn.Module:
        """Optimize model for inference with TorchScript.
        
        Args:
            model: Model to optimize
            example_input: Example input for tracing
            optimization_level: Optimization level ("conservative", "default", "aggressive")
            
        Returns:
            Optimized TorchScript model
        """
        model.eval()
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Optimize the traced model
            if optimization_level == "conservative":
                # Minimal optimizations
                optimized_model = torch.jit.optimize_for_inference(traced_model)
            elif optimization_level == "default":
                # Standard optimizations
                traced_model = torch.jit.freeze(traced_model)
                optimized_model = torch.jit.optimize_for_inference(traced_model)
            elif optimization_level == "aggressive":
                # Maximum optimizations
                traced_model = torch.jit.freeze(traced_model)
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                # Additional graph optimizations
                torch._C._jit_pass_optimize_graph(optimized_model.graph)
            else:
                raise ValueError(f"Unknown optimization level: {optimization_level}")
            
            return optimized_model
            
        except Exception as e:
            warnings.warn(f"TorchScript optimization failed: {e}")
            return model
    
    def convert_to_tensorrt(self,
                          model: nn.Module,
                          example_input: torch.Tensor,
                          fp16_mode: bool = True,
                          max_workspace_size: int = 1 << 25) -> Optional[Any]:
        """Convert model to TensorRT for NVIDIA GPU optimization.
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            fp16_mode: Whether to use FP16 precision
            max_workspace_size: Maximum workspace size in bytes
            
        Returns:
            TensorRT model or None if conversion fails
        """
        if not HAS_TENSORRT:
            warnings.warn("TensorRT not available")
            return None
        
        try:
            model.eval()
            
            # Convert to TensorRT
            model_trt = torch2trt.torch2trt(
                model,
                [example_input],
                fp16_mode=fp16_mode,
                max_workspace_size=max_workspace_size
            )
            
            return model_trt
            
        except Exception as e:
            warnings.warn(f"TensorRT conversion failed: {e}")
            return None
    
    def export_onnx(self,
                   model: nn.Module,
                   example_input: torch.Tensor,
                   output_path: str,
                   optimize_for: str = "general") -> None:
        """Export model to ONNX format.
        
        Args:
            model: PyTorch model
            example_input: Example input for tracing
            output_path: Path to save ONNX model
            optimize_for: Target optimization ("general", "jetson_nano", "cpu", "gpu")
        """
        model.eval()
        
        # Set dynamic axes for flexible input sizes
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        # Optimize ONNX model if possible
        try:
            import onnx
            import onnxoptimizer
            
            # Load and optimize
            onnx_model = onnx.load(output_path)
            
            if optimize_for == "jetson_nano":
                # Optimizations for Jetson Nano
                optimizations = [
                    'eliminate_nop_transpose',
                    'eliminate_nop_pad',
                    'eliminate_unused_initializer',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_transposes',
                    'fuse_transpose_into_gemm'
                ]
            elif optimize_for == "cpu":
                # CPU-specific optimizations
                optimizations = [
                    'eliminate_deadend',
                    'eliminate_identity',
                    'eliminate_nop_dropout',
                    'eliminate_nop_monotone_argmax',
                    'eliminate_nop_pad',
                    'eliminate_nop_transpose',
                    'eliminate_unused_initializer',
                    'extract_constant_to_initializer',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_log_softmax',
                    'fuse_consecutive_reduce_unsqueeze',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm',
                    'lift_lexical_references'
                ]
            else:
                # General optimizations
                optimizations = onnxoptimizer.get_available_passes()
            
            optimized_model = onnxoptimizer.optimize(onnx_model, optimizations)
            onnx.save(optimized_model, output_path)
            
        except ImportError:
            warnings.warn("ONNX optimization packages not available")
    
    def benchmark_model(self,
                       model: nn.Module,
                       example_input: torch.Tensor,
                       num_runs: int = 100,
                       warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            example_input: Example input
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with performance metrics
        """
        model.eval()
        device = next(model.parameters()).device
        example_input = example_input.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(example_input)
        
        # Benchmark runs
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(example_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = 1.0 / avg_time  # inferences per second
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved(device) / 1024**2  # MB
        else:
            memory_allocated = 0
            memory_cached = 0
        
        return {
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'total_time': total_time,
            'memory_allocated_mb': memory_allocated,
            'memory_cached_mb': memory_cached,
            'device': str(device)
        }
    
    def optimize_batch_size(self,
                          model: nn.Module,
                          example_input: torch.Tensor,
                          max_batch_size: int = 64,
                          memory_limit_mb: Optional[float] = None) -> int:
        """Find optimal batch size for inference.
        
        Args:
            model: Model to optimize
            example_input: Example input (batch size 1)
            max_batch_size: Maximum batch size to test
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Optimal batch size
        """
        model.eval()
        device = next(model.parameters()).device
        
        optimal_batch_size = 1
        best_throughput = 0
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break
            
            try:
                # Create batch
                batch_input = example_input.repeat(batch_size, *([1] * (example_input.dim() - 1)))
                batch_input = batch_input.to(device)
                
                # Check memory usage before benchmark
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                
                # Benchmark
                metrics = self.benchmark_model(model, batch_input, num_runs=20, warmup_runs=5)
                
                # Check memory limit
                if memory_limit_mb and metrics['memory_allocated_mb'] > memory_limit_mb:
                    break
                
                # Calculate per-sample throughput
                per_sample_throughput = metrics['throughput'] / batch_size
                
                if per_sample_throughput > best_throughput:
                    best_throughput = per_sample_throughput
                    optimal_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e
        
        return optimal_batch_size
    
    def create_optimized_pipeline(self,
                                model: nn.Module,
                                example_input: torch.Tensor,
                                target_device: str = "cpu",
                                optimization_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create complete optimized inference pipeline.
        
        Args:
            model: Model to optimize
            example_input: Example input
            target_device: Target deployment device
            optimization_config: Optimization configuration
            
        Returns:
            Dictionary with optimized components
        """
        if optimization_config is None:
            optimization_config = {}
        
        pipeline = {}
        
        # 1. Base model optimization
        if optimization_config.get('use_torchscript', True):
            optimized_model = self.optimize_for_inference(
                model, example_input, 
                optimization_level=optimization_config.get('optimization_level', 'default')
            )
        else:
            optimized_model = model
        
        # 2. Quantization
        if optimization_config.get('quantize', False):
            quantized_model = self.quantize_model(
                optimized_model,
                method=optimization_config.get('quantization_method', 'dynamic')
            )
            pipeline['quantized_model'] = quantized_model
        
        # 3. Pruning
        if optimization_config.get('prune', False):
            pruned_model = self.prune_model(
                optimized_model,
                sparsity=optimization_config.get('pruning_sparsity', 0.5)
            )
            pipeline['pruned_model'] = pruned_model
        
        # 4. TensorRT (for NVIDIA GPUs)
        if target_device == "gpu" and optimization_config.get('use_tensorrt', False):
            trt_model = self.convert_to_tensorrt(optimized_model, example_input)
            if trt_model:
                pipeline['tensorrt_model'] = trt_model
        
        # 5. Optimal batch size
        optimal_batch_size = self.optimize_batch_size(
            optimized_model, example_input,
            max_batch_size=optimization_config.get('max_batch_size', 32)
        )
        pipeline['optimal_batch_size'] = optimal_batch_size
        
        # 6. Performance benchmarks
        benchmark_results = self.benchmark_model(optimized_model, example_input)
        pipeline['benchmark_results'] = benchmark_results
        
        # 7. Export formats
        if optimization_config.get('export_onnx', False):
            onnx_path = optimization_config.get('onnx_path', 'model_optimized.onnx')
            self.export_onnx(optimized_model, example_input, onnx_path, target_device)
            pipeline['onnx_path'] = onnx_path
        
        pipeline['optimized_model'] = optimized_model
        
        return pipeline
    
    def get_optimization_recommendations(self,
                                       model: nn.Module,
                                       target_device: str,
                                       target_latency_ms: Optional[float] = None,
                                       memory_constraint_mb: Optional[float] = None) -> Dict[str, Any]:
        """Get optimization recommendations based on constraints.
        
        Args:
            model: Model to analyze
            target_device: Target deployment device
            target_latency_ms: Target latency in milliseconds
            memory_constraint_mb: Memory constraint in MB
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'quantization': False,
            'pruning': False,
            'torchscript': True,
            'tensorrt': False,
            'batch_optimization': True,
            'onnx_export': False
        }
        
        # Device-specific recommendations
        if target_device.lower() in ['cpu', 'edge']:
            recommendations.update({
                'quantization': True,
                'quantization_method': 'dynamic',
                'pruning': True,
                'pruning_sparsity': 0.3,
                'onnx_export': True
            })
        
        elif target_device.lower() in ['gpu', 'cuda']:
            recommendations.update({
                'tensorrt': True,
                'quantization': True,
                'quantization_method': 'static'
            })
        
        elif target_device.lower() in ['jetson', 'jetson_nano']:
            recommendations.update({
                'quantization': True,
                'quantization_method': 'dynamic',
                'tensorrt': True,
                'onnx_export': True,
                'pruning': True,
                'pruning_sparsity': 0.5
            })
        
        # Constraint-based adjustments
        if target_latency_ms and target_latency_ms < 10:
            recommendations.update({
                'quantization': True,
                'pruning': True,
                'tensorrt': True if target_device == 'gpu' else False
            })
        
        if memory_constraint_mb and memory_constraint_mb < 100:
            recommendations.update({
                'quantization': True,
                'pruning': True,
                'pruning_sparsity': 0.7
            })
        
        return recommendations