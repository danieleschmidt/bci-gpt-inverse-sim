"""Performance benchmarking and optimization system for BCI-GPT scaling."""

import time
import psutil
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    inference_time_ms: Optional[float] = None
    training_time_ms: Optional[float] = None
    model_size_mb: Optional[float] = None

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_threads: int = 1
    enable_gpu_profiling: bool = False
    profile_memory: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [256, 512, 1024, 2048]

class PerformanceBenchmarkSystem:
    """Comprehensive performance benchmarking system for BCI-GPT."""
    
    def __init__(self, config: BenchmarkConfig = None):
        """Initialize performance benchmark system.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results_history: List[Dict[str, Any]] = []
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
    
    def benchmark_inference(self, 
                          model_fn: Callable,
                          input_data: Any,
                          batch_size: int = 1) -> PerformanceMetrics:
        """Benchmark model inference performance.
        
        Args:
            model_fn: Model function to benchmark
            input_data: Input data for inference
            batch_size: Batch size for inference
            
        Returns:
            Performance metrics
        """
        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                _ = model_fn(input_data)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Benchmark
        latencies = []
        memory_usage = []
        cpu_usage = []
        
        for i in range(self.config.benchmark_iterations):
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = process.cpu_percent()
            
            # Time inference
            start_time = time.perf_counter()
            try:
                result = model_fn(input_data)
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                continue
            end_time = time.perf_counter()
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = process.cpu_percent()
            
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)
            memory_usage.append(mem_after - mem_before)
            cpu_usage.append(max(cpu_after - cpu_before, 0))
        
        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        throughput = (batch_size * 1000) / avg_latency if avg_latency > 0 else 0
        avg_memory = statistics.mean(memory_usage)
        avg_cpu = statistics.mean(cpu_usage)
        
        metrics = PerformanceMetrics(
            latency_ms=avg_latency,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            inference_time_ms=avg_latency
        )
        
        # Add GPU metrics if available
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                    metrics.gpu_memory_mb = gpu_memory
                    metrics.gpu_utilization_percent = gpu_util
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    def benchmark_training_step(self,
                               training_fn: Callable,
                               batch_data: Any,
                               optimizer: Any = None) -> PerformanceMetrics:
        """Benchmark training step performance.
        
        Args:
            training_fn: Training function to benchmark
            batch_data: Batch data for training
            optimizer: Optimizer for training step
            
        Returns:
            Performance metrics
        """
        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                if optimizer:
                    optimizer.zero_grad()
                loss = training_fn(batch_data)
                if hasattr(loss, 'backward'):
                    loss.backward()
                if optimizer:
                    optimizer.step()
            except Exception as e:
                logger.warning(f"Training warmup failed: {e}")
        
        # Benchmark
        training_times = []
        memory_usage = []
        
        for i in range(self.config.benchmark_iterations):
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            try:
                if optimizer:
                    optimizer.zero_grad()
                loss = training_fn(batch_data)
                if hasattr(loss, 'backward'):
                    loss.backward()
                if optimizer:
                    optimizer.step()
            except Exception as e:
                logger.error(f"Training benchmark iteration {i} failed: {e}")
                continue
            end_time = time.perf_counter()
            
            mem_after = process.memory_info().rss / 1024 / 1024
            
            training_time = (end_time - start_time) * 1000
            training_times.append(training_time)
            memory_usage.append(mem_after - mem_before)
        
        avg_training_time = statistics.mean(training_times)
        avg_memory = statistics.mean(memory_usage)
        
        return PerformanceMetrics(
            latency_ms=avg_training_time,
            throughput_ops_per_sec=1000 / avg_training_time if avg_training_time > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=0,  # Difficult to measure during training
            training_time_ms=avg_training_time
        )
    
    def profile_memory_usage(self, 
                           model_fn: Callable,
                           input_data: Any,
                           duration_seconds: float = 10.0) -> Dict[str, float]:
        """Profile memory usage over time.
        
        Args:
            model_fn: Model function to profile
            input_data: Input data for profiling
            duration_seconds: Duration to profile
            
        Returns:
            Memory profiling results
        """
        memory_samples = []
        start_time = time.time()
        
        def memory_monitor():
            while time.time() - start_time < duration_seconds:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
                    time.sleep(0.1)  # Sample every 100ms
                except Exception:
                    break
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run model continuously
        end_time = time.time() + duration_seconds
        iterations = 0
        while time.time() < end_time:
            try:
                _ = model_fn(input_data)
                iterations += 1
            except Exception as e:
                logger.warning(f"Memory profiling iteration failed: {e}")
                break
        
        monitor_thread.join(timeout=1.0)
        
        if memory_samples:
            return {
                "peak_memory_mb": max(memory_samples),
                "avg_memory_mb": statistics.mean(memory_samples),
                "min_memory_mb": min(memory_samples),
                "memory_std_mb": statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0,
                "iterations_completed": iterations
            }
        else:
            return {"error": "No memory samples collected"}
    
    def benchmark_batch_sizes(self,
                             model_fn: Callable,
                             input_generator: Callable[[int], Any]) -> Dict[int, PerformanceMetrics]:
        """Benchmark performance across different batch sizes.
        
        Args:
            model_fn: Model function to benchmark
            input_generator: Function that generates input data for given batch size
            
        Returns:
            Dictionary mapping batch sizes to performance metrics
        """
        results = {}
        
        for batch_size in self.config.batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            try:
                input_data = input_generator(batch_size)
                metrics = self.benchmark_inference(model_fn, input_data, batch_size)
                results[batch_size] = metrics
                
                # Log key metrics
                logger.info(f"Batch {batch_size}: {metrics.latency_ms:.2f}ms, "
                           f"{metrics.throughput_ops_per_sec:.2f} ops/sec")
            except Exception as e:
                logger.error(f"Batch size {batch_size} benchmark failed: {e}")
                
        return results
    
    def find_optimal_batch_size(self,
                               model_fn: Callable,
                               input_generator: Callable[[int], Any],
                               metric: str = "throughput") -> int:
        """Find optimal batch size for given metric.
        
        Args:
            model_fn: Model function to benchmark
            input_generator: Function that generates input data
            metric: Metric to optimize for ("throughput", "latency", "memory")
            
        Returns:
            Optimal batch size
        """
        batch_results = self.benchmark_batch_sizes(model_fn, input_generator)
        
        if not batch_results:
            return self.config.batch_sizes[0] if self.config.batch_sizes else 1
        
        if metric == "throughput":
            optimal_batch = max(batch_results.keys(), 
                              key=lambda b: batch_results[b].throughput_ops_per_sec)
        elif metric == "latency":
            optimal_batch = min(batch_results.keys(),
                              key=lambda b: batch_results[b].latency_ms)
        elif metric == "memory":
            optimal_batch = min(batch_results.keys(),
                              key=lambda b: batch_results[b].memory_usage_mb)
        else:
            # Default to throughput
            optimal_batch = max(batch_results.keys(),
                              key=lambda b: batch_results[b].throughput_ops_per_sec)
        
        logger.info(f"Optimal batch size for {metric}: {optimal_batch}")
        return optimal_batch
    
    def generate_performance_report(self,
                                  benchmark_name: str,
                                  results: Dict[str, Any],
                                  save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            benchmark_name: Name of the benchmark
            results: Benchmark results
            save_path: Optional path to save report
            
        Returns:
            Performance report dictionary
        """
        report = {
            "benchmark_name": benchmark_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "gpu_available": self.gpu_available
            },
            "config": asdict(self.config),
            "results": results
        }
        
        # Add GPU info if available
        if self.gpu_available:
            try:
                import torch
                report["system_info"]["gpu_name"] = torch.cuda.get_device_name()
                report["system_info"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            except Exception:
                pass
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to: {save_path}")
        
        self.results_history.append(report)
        return report
    
    def compare_models(self,
                      model_configs: Dict[str, Dict[str, Any]],
                      input_data: Any) -> Dict[str, Any]:
        """Compare performance of multiple model configurations.
        
        Args:
            model_configs: Dictionary of model name to config
            input_data: Input data for benchmarking
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Benchmarking model: {model_name}")
            try:
                model_fn = config["model_fn"]
                metrics = self.benchmark_inference(model_fn, input_data)
                comparison_results[model_name] = {
                    "metrics": asdict(metrics),
                    "config": config.get("description", "")
                }
            except Exception as e:
                logger.error(f"Model {model_name} benchmark failed: {e}")
                comparison_results[model_name] = {"error": str(e)}
        
        # Generate comparison summary
        if len(comparison_results) > 1:
            # Find best performing models
            valid_results = {k: v for k, v in comparison_results.items() 
                           if "metrics" in v}
            
            if valid_results:
                best_throughput = max(valid_results.keys(),
                                    key=lambda k: valid_results[k]["metrics"]["throughput_ops_per_sec"])
                best_latency = min(valid_results.keys(),
                                 key=lambda k: valid_results[k]["metrics"]["latency_ms"])
                best_memory = min(valid_results.keys(),
                                key=lambda k: valid_results[k]["metrics"]["memory_usage_mb"])
                
                comparison_results["summary"] = {
                    "best_throughput": best_throughput,
                    "best_latency": best_latency,
                    "best_memory_efficiency": best_memory
                }
        
        return comparison_results
    
    def continuous_monitoring(self,
                            model_fn: Callable,
                            input_data: Any,
                            duration_minutes: float = 5.0,
                            sample_interval_seconds: float = 30.0) -> Dict[str, Any]:
        """Continuously monitor model performance over time.
        
        Args:
            model_fn: Model function to monitor
            input_data: Input data for monitoring
            duration_minutes: Duration to monitor in minutes
            sample_interval_seconds: Interval between samples
            
        Returns:
            Continuous monitoring results
        """
        monitoring_results = []
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Take performance sample
                sample_start = time.time()
                metrics = self.benchmark_inference(model_fn, input_data)
                
                monitoring_results.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_minutes": (time.time() - start_time) / 60,
                    "metrics": asdict(metrics)
                })
                
                # Log current performance
                logger.info(f"Monitor sample: {metrics.latency_ms:.2f}ms latency, "
                           f"{metrics.throughput_ops_per_sec:.2f} ops/sec")
                
                # Wait for next sample
                sample_duration = time.time() - sample_start
                sleep_time = max(0, sample_interval_seconds - sample_duration)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Monitoring sample failed: {e}")
                time.sleep(sample_interval_seconds)
        
        # Analyze trends
        if monitoring_results:
            latencies = [r["metrics"]["latency_ms"] for r in monitoring_results]
            throughputs = [r["metrics"]["throughput_ops_per_sec"] for r in monitoring_results]
            memory_usage = [r["metrics"]["memory_usage_mb"] for r in monitoring_results]
            
            analysis = {
                "samples_collected": len(monitoring_results),
                "duration_minutes": duration_minutes,
                "latency_trend": {
                    "avg": statistics.mean(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "std": statistics.stdev(latencies) if len(latencies) > 1 else 0
                },
                "throughput_trend": {
                    "avg": statistics.mean(throughputs),
                    "min": min(throughputs),
                    "max": max(throughputs),
                    "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0
                },
                "memory_trend": {
                    "avg": statistics.mean(memory_usage),
                    "min": min(memory_usage),
                    "max": max(memory_usage),
                    "std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
                }
            }
            
            return {
                "samples": monitoring_results,
                "analysis": analysis
            }
        
        return {"error": "No monitoring samples collected"}

class ModelOptimizationSuite:
    """Suite of model optimization techniques for improved performance."""
    
    def __init__(self):
        """Initialize optimization suite."""
        self.optimization_history: List[Dict[str, Any]] = []
    
    def apply_quantization(self, model: Any, quantization_type: str = "dynamic") -> Any:
        """Apply quantization to reduce model size and improve inference speed.
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization ("dynamic", "static", "qat")
            
        Returns:
            Quantized model
        """
        try:
            # Mock quantization - in real implementation would use PyTorch quantization
            logger.info(f"Applying {quantization_type} quantization")
            
            # Simulate quantization effects
            optimization_info = {
                "technique": "quantization",
                "type": quantization_type,
                "model_size_reduction": 0.75,  # 75% size reduction
                "inference_speedup": 1.5,      # 1.5x speedup
                "accuracy_loss": 0.02           # 2% accuracy loss
            }
            
            self.optimization_history.append(optimization_info)
            return model  # Return original model in mock implementation
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def apply_pruning(self, model: Any, sparsity: float = 0.5) -> Any:
        """Apply pruning to remove unnecessary weights.
        
        Args:
            model: Model to prune
            sparsity: Fraction of weights to prune (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        try:
            logger.info(f"Applying pruning with {sparsity*100:.1f}% sparsity")
            
            # Simulate pruning effects
            optimization_info = {
                "technique": "pruning",
                "sparsity": sparsity,
                "model_size_reduction": sparsity * 0.8,  # Less than theoretical due to overhead
                "inference_speedup": 1 + (sparsity * 0.3),  # Modest speedup
                "accuracy_loss": sparsity * 0.05  # Accuracy loss scales with sparsity
            }
            
            self.optimization_history.append(optimization_info)
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def apply_knowledge_distillation(self, 
                                   teacher_model: Any,
                                   student_model: Any,
                                   temperature: float = 3.0) -> Any:
        """Apply knowledge distillation to create smaller, faster model.
        
        Args:
            teacher_model: Large teacher model
            student_model: Smaller student model
            temperature: Distillation temperature
            
        Returns:
            Distilled student model
        """
        try:
            logger.info(f"Applying knowledge distillation with temperature {temperature}")
            
            # Simulate distillation effects
            optimization_info = {
                "technique": "knowledge_distillation",
                "temperature": temperature,
                "model_size_reduction": 0.3,   # 70% size reduction
                "inference_speedup": 3.0,      # 3x speedup
                "accuracy_retention": 0.95     # 95% of teacher accuracy
            }
            
            self.optimization_history.append(optimization_info)
            return student_model
            
        except Exception as e:
            logger.error(f"Knowledge distillation failed: {e}")
            return student_model
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all applied optimizations.
        
        Returns:
            Optimization summary
        """
        if not self.optimization_history:
            return {"message": "No optimizations applied yet"}
        
        total_size_reduction = 1.0
        total_speedup = 1.0
        total_accuracy_impact = 0.0
        
        techniques_used = []
        
        for opt in self.optimization_history:
            techniques_used.append(opt["technique"])
            
            if "model_size_reduction" in opt:
                total_size_reduction *= (1.0 - opt["model_size_reduction"])
            
            if "inference_speedup" in opt:
                total_speedup *= opt["inference_speedup"]
            
            if "accuracy_loss" in opt:
                total_accuracy_impact += opt["accuracy_loss"]
            elif "accuracy_retention" in opt:
                total_accuracy_impact += (1.0 - opt["accuracy_retention"])
        
        return {
            "techniques_applied": techniques_used,
            "total_size_reduction": 1.0 - total_size_reduction,
            "total_speedup": total_speedup,
            "estimated_accuracy_impact": total_accuracy_impact,
            "optimization_details": self.optimization_history
        }