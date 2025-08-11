"""Advanced performance profiler for BCI-GPT system optimization."""

import time
import functools
import threading
import traceback
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    function_name: str
    execution_time: float
    memory_usage: float
    gpu_memory_usage: float
    cpu_percent: float
    thread_id: int
    timestamp: float
    args_hash: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProfileReport:
    """Comprehensive performance profile report."""
    total_calls: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    memory_peak: float
    gpu_memory_peak: float
    success_rate: float
    bottlenecks: List[str]
    recommendations: List[str]


class AdvancedProfiler:
    """Advanced performance profiler with intelligent analysis."""
    
    def __init__(self, 
                 max_records: int = 10000,
                 sampling_interval: float = 0.1,
                 enable_memory_tracking: bool = True,
                 enable_gpu_tracking: bool = True):
        """Initialize advanced profiler.
        
        Args:
            max_records: Maximum number of performance records to keep
            sampling_interval: Sampling interval for continuous monitoring
            enable_memory_tracking: Enable memory usage tracking
            enable_gpu_tracking: Enable GPU memory tracking
        """
        self.max_records = max_records
        self.sampling_interval = sampling_interval
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_gpu_tracking = enable_gpu_tracking
        
        # Performance data storage
        self.metrics: deque = deque(maxlen=max_records)
        self.function_stats: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.bottlenecks: Dict[str, float] = {}
        
        # System monitoring
        self.system_metrics: deque = deque(maxlen=1000)
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Profiling state
        self.active_profiles: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        
        # Analysis thresholds
        self.slow_function_threshold = 1.0  # seconds
        self.memory_leak_threshold = 0.1    # 10% increase per call
        self.bottleneck_threshold = 0.05    # 5% of total time
    
    def start_monitoring(self) -> None:
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_system(self) -> None:
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                system_metric = self._collect_system_metrics()
                with self._lock:
                    self.system_metrics.append(system_metric)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_mb': 0.0,
            'gpu_utilization': 0.0,
            'gpu_memory_mb': 0.0
        }
        
        if HAS_PSUTIL:
            process = psutil.Process()
            metrics.update({
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024
            })
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                metrics.update({
                    'gpu_utilization': torch.cuda.utilization(),
                    'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024
                })
            except:
                pass
        
        return metrics
    
    def profile_function(self, 
                        include_memory: bool = True,
                        include_gpu: bool = True,
                        track_args: bool = False) -> Callable:
        """Decorator for profiling function performance.
        
        Args:
            include_memory: Track memory usage
            include_gpu: Track GPU memory usage
            track_args: Track function arguments for analysis
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(
                    func, args, kwargs,
                    include_memory, include_gpu, track_args
                )
            return wrapper
        return decorator
    
    def _profile_execution(self,
                          func: Callable,
                          args: tuple,
                          kwargs: dict,
                          include_memory: bool,
                          include_gpu: bool,
                          track_args: bool) -> Any:
        """Profile single function execution."""
        func_name = f"{func.__module__}.{func.__name__}"
        thread_id = threading.get_ident()
        start_time = time.time()
        
        # Pre-execution metrics
        pre_metrics = self._collect_system_metrics() if (include_memory or include_gpu) else {}
        
        # Arguments hash for caching analysis
        args_hash = ""
        if track_args:
            try:
                args_hash = str(hash(str(args) + str(sorted(kwargs.items()))))
            except:
                args_hash = "unhashable"
        
        success = True
        error_message = None
        result = None
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Function {func_name} failed: {e}")
            raise
        
        finally:
            # Post-execution metrics
            execution_time = time.time() - start_time
            post_metrics = self._collect_system_metrics() if (include_memory or include_gpu) else {}
            
            # Create performance metric
            metric = PerformanceMetric(
                function_name=func_name,
                execution_time=execution_time,
                memory_usage=post_metrics.get('memory_mb', 0) - pre_metrics.get('memory_mb', 0),
                gpu_memory_usage=post_metrics.get('gpu_memory_mb', 0) - pre_metrics.get('gpu_memory_mb', 0),
                cpu_percent=post_metrics.get('cpu_percent', 0),
                thread_id=thread_id,
                timestamp=start_time,
                args_hash=args_hash,
                success=success,
                error_message=error_message
            )
            
            # Store metric
            with self._lock:
                self.metrics.append(metric)
                self.function_stats[func_name].append(metric)
                
                # Keep function stats bounded
                if len(self.function_stats[func_name]) > 1000:
                    self.function_stats[func_name] = self.function_stats[func_name][-500:]
        
        return result
    
    def profile_code_block(self, name: str):
        """Context manager for profiling code blocks."""
        return CodeBlockProfiler(self, name)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze collected performance data and generate insights."""
        with self._lock:
            if not self.metrics:
                return {'error': 'No performance data available'}
            
            analysis = {
                'summary': self._generate_summary(),
                'function_analysis': self._analyze_functions(),
                'bottlenecks': self._identify_bottlenecks(),
                'memory_analysis': self._analyze_memory_usage(),
                'recommendations': self._generate_recommendations(),
                'trends': self._analyze_trends()
            }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        total_metrics = len(self.metrics)
        if total_metrics == 0:
            return {}
        
        execution_times = [m.execution_time for m in self.metrics]
        memory_usage = [m.memory_usage for m in self.metrics if m.memory_usage > 0]
        success_count = sum(1 for m in self.metrics if m.success)
        
        return {
            'total_function_calls': total_metrics,
            'success_rate': success_count / total_metrics,
            'total_execution_time': sum(execution_times),
            'average_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'memory_usage_mb': sum(memory_usage),
            'unique_functions': len(self.function_stats),
            'monitoring_duration': time.time() - self.metrics[0].timestamp if self.metrics else 0
        }
    
    def _analyze_functions(self) -> Dict[str, ProfileReport]:
        """Analyze performance of individual functions."""
        function_reports = {}
        
        for func_name, metrics_list in self.function_stats.items():
            if not metrics_list:
                continue
                
            execution_times = [m.execution_time for m in metrics_list]
            memory_usage = [m.memory_usage for m in metrics_list if m.memory_usage > 0]
            success_count = sum(1 for m in metrics_list if m.success)
            
            # Identify bottlenecks for this function
            bottlenecks = []
            if max(execution_times) > self.slow_function_threshold:
                bottlenecks.append(f"Slow execution (max: {max(execution_times):.2f}s)")
            
            if memory_usage and max(memory_usage) > 100:  # MB
                bottlenecks.append(f"High memory usage (max: {max(memory_usage):.1f} MB)")
            
            # Generate recommendations
            recommendations = []
            if len(execution_times) > 10:
                avg_time = sum(execution_times) / len(execution_times)
                if avg_time > 0.5:
                    recommendations.append("Consider optimization or caching")
                if max(execution_times) / avg_time > 10:
                    recommendations.append("Investigate performance variance")
            
            function_reports[func_name] = ProfileReport(
                total_calls=len(metrics_list),
                total_time=sum(execution_times),
                average_time=sum(execution_times) / len(execution_times),
                min_time=min(execution_times),
                max_time=max(execution_times),
                memory_peak=max(memory_usage) if memory_usage else 0,
                gpu_memory_peak=max(m.gpu_memory_usage for m in metrics_list),
                success_rate=success_count / len(metrics_list),
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
        
        return function_reports
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify system bottlenecks."""
        if not self.function_stats:
            return []
        
        bottlenecks = []
        total_execution_time = sum(
            sum(m.execution_time for m in metrics)
            for metrics in self.function_stats.values()
        )
        
        for func_name, metrics_list in self.function_stats.items():
            func_total_time = sum(m.execution_time for m in metrics_list)
            percentage = func_total_time / total_execution_time if total_execution_time > 0 else 0
            
            if percentage > self.bottleneck_threshold:
                bottlenecks.append({
                    'function': func_name,
                    'total_time': func_total_time,
                    'percentage': percentage * 100,
                    'call_count': len(metrics_list),
                    'avg_time': func_total_time / len(metrics_list)
                })
        
        return sorted(bottlenecks, key=lambda x: x['percentage'], reverse=True)
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.enable_memory_tracking:
            return {'error': 'Memory tracking disabled'}
        
        memory_metrics = [m for m in self.metrics if m.memory_usage != 0]
        if not memory_metrics:
            return {'error': 'No memory data available'}
        
        memory_values = [m.memory_usage for m in memory_metrics]
        
        # Detect potential memory leaks
        memory_leaks = []
        for func_name, metrics_list in self.function_stats.items():
            func_memory = [m.memory_usage for m in metrics_list if m.memory_usage > 0]
            if len(func_memory) > 10:
                # Check for consistent memory increase
                trend = self._calculate_trend(func_memory)
                if trend > self.memory_leak_threshold:
                    memory_leaks.append({
                        'function': func_name,
                        'trend': trend,
                        'avg_usage': sum(func_memory) / len(func_memory)
                    })
        
        return {
            'total_memory_allocated': sum(m for m in memory_values if m > 0),
            'total_memory_freed': abs(sum(m for m in memory_values if m < 0)),
            'peak_allocation': max(memory_values),
            'potential_leaks': memory_leaks,
            'memory_efficiency': len([m for m in memory_values if m < 0]) / len(memory_values)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (val - y_mean) for i, val in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.metrics) < 10:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Group metrics by time windows
        window_size = 60  # 60 second windows
        windows = defaultdict(list)
        
        for metric in self.metrics:
            window = int(metric.timestamp // window_size)
            windows[window].append(metric)
        
        if len(windows) < 3:
            return {'error': 'Need more time windows for trend analysis'}
        
        # Calculate trends
        window_times = []
        window_avg_times = []
        
        for window_id in sorted(windows.keys()):
            window_metrics = windows[window_id]
            avg_time = sum(m.execution_time for m in window_metrics) / len(window_metrics)
            window_times.append(window_id)
            window_avg_times.append(avg_time)
        
        performance_trend = self._calculate_trend(window_avg_times)
        
        return {
            'performance_trend': performance_trend,
            'trend_description': (
                'Performance degrading' if performance_trend > 0.01 else
                'Performance improving' if performance_trend < -0.01 else
                'Performance stable'
            ),
            'time_windows': len(windows),
            'data_points': len(self.metrics)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze bottlenecks
        bottlenecks = self._identify_bottlenecks()
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            if top_bottleneck['percentage'] > 20:
                recommendations.append(
                    f"Optimize {top_bottleneck['function']} - consumes {top_bottleneck['percentage']:.1f}% of execution time"
                )
        
        # Memory recommendations
        memory_analysis = self._analyze_memory_usage()
        if 'potential_leaks' in memory_analysis and memory_analysis['potential_leaks']:
            recommendations.append("Investigate potential memory leaks in identified functions")
        
        # Performance trends
        trends = self._analyze_trends()
        if 'performance_trend' in trends and trends['performance_trend'] > 0.01:
            recommendations.append("Performance is degrading over time - investigate causes")
        
        # Success rate recommendations
        summary = self._generate_summary()
        if 'success_rate' in summary and summary['success_rate'] < 0.95:
            recommendations.append("Low success rate detected - improve error handling")
        
        return recommendations
    
    def generate_report(self, format: str = 'text') -> str:
        """Generate comprehensive performance report."""
        analysis = self.analyze_performance()
        
        if format == 'text':
            return self._generate_text_report(analysis)
        elif format == 'json':
            import json
            return json.dumps(analysis, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_report(self, analysis: Dict[str, Any]) -> str:
        """Generate text format performance report."""
        report = []
        report.append("=" * 60)
        report.append("BCI-GPT PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Summary
        if 'summary' in analysis:
            summary = analysis['summary']
            report.append("\n📊 SUMMARY:")
            report.append(f"  Total Function Calls: {summary.get('total_function_calls', 0)}")
            report.append(f"  Success Rate: {summary.get('success_rate', 0):.2%}")
            report.append(f"  Total Execution Time: {summary.get('total_execution_time', 0):.2f}s")
            report.append(f"  Average Execution Time: {summary.get('average_execution_time', 0)*1000:.1f}ms")
            report.append(f"  Memory Usage: {summary.get('memory_usage_mb', 0):.1f} MB")
        
        # Bottlenecks
        if 'bottlenecks' in analysis and analysis['bottlenecks']:
            report.append("\n🚨 PERFORMANCE BOTTLENECKS:")
            for bottleneck in analysis['bottlenecks'][:5]:
                report.append(f"  • {bottleneck['function']}: {bottleneck['percentage']:.1f}% ({bottleneck['total_time']:.2f}s)")
        
        # Recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            report.append("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                report.append(f"  {i}. {rec}")
        
        # Trends
        if 'trends' in analysis and 'trend_description' in analysis['trends']:
            report.append(f"\n📈 PERFORMANCE TREND: {analysis['trends']['trend_description']}")
        
        return '\n'.join(report)
    
    def clear_data(self) -> None:
        """Clear all collected performance data."""
        with self._lock:
            self.metrics.clear()
            self.function_stats.clear()
            self.bottlenecks.clear()
            self.system_metrics.clear()
        logger.info("Performance data cleared")
    
    def export_data(self, filepath: str) -> None:
        """Export performance data to file."""
        import json
        
        with self._lock:
            data = {
                'metrics': [asdict(m) for m in self.metrics],
                'system_metrics': list(self.system_metrics),
                'export_timestamp': time.time()
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Performance data exported to {filepath}")


class CodeBlockProfiler:
    """Context manager for profiling code blocks."""
    
    def __init__(self, profiler: AdvancedProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.pre_metrics = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.profiler.enable_memory_tracking or self.profiler.enable_gpu_tracking:
            self.pre_metrics = self.profiler._collect_system_metrics()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        post_metrics = self.profiler._collect_system_metrics() if self.pre_metrics else {}
        
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        metric = PerformanceMetric(
            function_name=f"code_block.{self.name}",
            execution_time=execution_time,
            memory_usage=post_metrics.get('memory_mb', 0) - self.pre_metrics.get('memory_mb', 0) if self.pre_metrics else 0,
            gpu_memory_usage=post_metrics.get('gpu_memory_mb', 0) - self.pre_metrics.get('gpu_memory_mb', 0) if self.pre_metrics else 0,
            cpu_percent=post_metrics.get('cpu_percent', 0),
            thread_id=threading.get_ident(),
            timestamp=self.start_time,
            args_hash="",
            success=success,
            error_message=error_message
        )
        
        with self.profiler._lock:
            self.profiler.metrics.append(metric)
            self.profiler.function_stats[f"code_block.{self.name}"].append(metric)


# Global profiler instance
global_profiler = AdvancedProfiler()


def profile(include_memory: bool = True, include_gpu: bool = True, track_args: bool = False):
    """Convenient profiling decorator using global profiler."""
    return global_profiler.profile_function(include_memory, include_gpu, track_args)


def start_profiling():
    """Start global profiling."""
    global_profiler.start_monitoring()


def stop_profiling():
    """Stop global profiling."""
    global_profiler.stop_monitoring()


def get_performance_report(format: str = 'text') -> str:
    """Get performance report from global profiler."""
    return global_profiler.generate_report(format)


def clear_profiling_data():
    """Clear global profiling data."""
    global_profiler.clear_data()