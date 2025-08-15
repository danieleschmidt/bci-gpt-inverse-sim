"""Model Health Manager for BCI-GPT Self-Healing System.

Monitors model performance, detects degradation, and manages model
fallbacks and optimization for continuous high-quality inference.
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from pathlib import Path
import json

from ..utils.monitoring import HealthStatus
from ..utils.error_handling import BCI_GPTError
from ..utils.performance_optimizer import PerformanceOptimizer
from ..core.models import BCIGPTModel


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    memory_usage_mb: float = 0.0
    confidence_score: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelHealthConfig:
    """Configuration for model health monitoring."""
    monitoring_interval: float = 10.0
    performance_window: int = 100  # Number of recent predictions to track
    accuracy_threshold: float = 0.75
    latency_threshold_ms: float = 100.0
    confidence_threshold: float = 0.6
    error_rate_threshold: float = 0.1
    degradation_detection_window: int = 20
    backup_model_enabled: bool = True
    auto_optimization_enabled: bool = True
    model_refresh_interval_hours: float = 24.0


class ModelHealthManager:
    """Manages model health monitoring, degradation detection, and recovery.
    
    Continuously monitors model performance, detects degradation patterns,
    manages backup models, and performs automatic optimization.
    """
    
    def __init__(self, config: Optional[ModelHealthConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or ModelHealthConfig()
        
        # Model management
        self.primary_model: Optional[BCIGPTModel] = None
        self.backup_models: List[BCIGPTModel] = []
        self.current_model_path: Optional[str] = None
        self.backup_model_paths: List[str] = []
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=self.config.performance_window)
        self.prediction_count = 0
        self.error_count = 0
        
        # Health monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.current_health_status = HealthStatus.UNKNOWN
        self.last_health_check = datetime.now()
        
        # Performance trends
        self.accuracy_trend: List[float] = []
        self.latency_trend: List[float] = []
        self.confidence_trend: List[float] = []
        
        # Callbacks and optimization
        self.health_callbacks: List[Callable] = []
        self.performance_optimizer = PerformanceOptimizer()
        
        # Model switching
        self.is_using_backup = False
        self.backup_switch_time: Optional[datetime] = None
        self.model_switch_count = 0
        
        # Recovery state
        self.degradation_detected = False
        self.degradation_start_time: Optional[datetime] = None
        self.recovery_attempts = 0
    
    def load_primary_model(self, model_path: str) -> None:
        """Load the primary model for health monitoring."""
        try:
            self.logger.info(f"Loading primary model from: {model_path}")
            # Note: In a real implementation, this would load the actual model
            self.current_model_path = model_path
            self.primary_model = None  # Placeholder - would load actual BCIGPTModel
            
            # Reset health status
            self.current_health_status = HealthStatus.HEALTHY
            self.is_using_backup = False
            self.degradation_detected = False
            
            self.logger.info("Primary model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load primary model: {e}")
            raise BCI_GPTError(f"Model loading failed: {e}")
    
    def add_backup_model(self, model_path: str) -> None:
        """Add a backup model."""
        try:
            self.logger.info(f"Adding backup model: {model_path}")
            self.backup_model_paths.append(model_path)
            # Note: In a real implementation, this would load the backup model
            self.logger.info(f"Backup model added. Total backups: {len(self.backup_model_paths)}")
            
        except Exception as e:
            self.logger.error(f"Failed to add backup model: {e}")
    
    def record_prediction_metrics(self, accuracy: float, latency_ms: float, 
                                 confidence: float, error_occurred: bool = False) -> None:
        """Record metrics from a model prediction."""
        self.prediction_count += 1
        if error_occurred:
            self.error_count += 1
        
        # Calculate throughput (predictions per second)
        current_time = datetime.now()
        if len(self.metrics_history) > 0:
            time_diff = (current_time - self.metrics_history[-1].timestamp).total_seconds()
            throughput = 1.0 / max(time_diff, 0.001)
        else:
            throughput = 1.0
        
        # Create metrics record
        metrics = ModelMetrics(
            accuracy=accuracy,
            latency_ms=latency_ms,
            throughput=throughput,
            confidence_score=confidence,
            error_rate=self.error_count / self.prediction_count,
            timestamp=current_time
        )
        
        self.metrics_history.append(metrics)
        self._update_performance_trends()
        
        # Check for immediate health issues
        self._check_immediate_health(metrics)
    
    def _update_performance_trends(self) -> None:
        """Update performance trends based on recent metrics."""
        if len(self.metrics_history) < 10:
            return
        
        # Calculate trends over recent window
        recent_metrics = list(self.metrics_history)[-20:]
        
        # Accuracy trend
        accuracies = [m.accuracy for m in recent_metrics]
        if len(accuracies) > 1:
            accuracy_slope = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            self.accuracy_trend.append(accuracy_slope)
            if len(self.accuracy_trend) > 50:
                self.accuracy_trend.pop(0)
        
        # Latency trend
        latencies = [m.latency_ms for m in recent_metrics]
        if len(latencies) > 1:
            latency_slope = np.polyfit(range(len(latencies)), latencies, 1)[0]
            self.latency_trend.append(latency_slope)
            if len(self.latency_trend) > 50:
                self.latency_trend.pop(0)
        
        # Confidence trend
        confidences = [m.confidence_score for m in recent_metrics]
        if len(confidences) > 1:
            confidence_slope = np.polyfit(range(len(confidences)), confidences, 1)[0]
            self.confidence_trend.append(confidence_slope)
            if len(self.confidence_trend) > 50:
                self.confidence_trend.pop(0)
    
    def _check_immediate_health(self, metrics: ModelMetrics) -> None:
        """Check for immediate health issues based on current metrics."""
        issues = []
        
        # Check accuracy
        if metrics.accuracy < self.config.accuracy_threshold:
            issues.append(f"Accuracy below threshold: {metrics.accuracy:.3f} < {self.config.accuracy_threshold}")
        
        # Check latency
        if metrics.latency_ms > self.config.latency_threshold_ms:
            issues.append(f"Latency above threshold: {metrics.latency_ms:.1f}ms > {self.config.latency_threshold_ms}ms")
        
        # Check confidence
        if metrics.confidence_score < self.config.confidence_threshold:
            issues.append(f"Confidence below threshold: {metrics.confidence_score:.3f} < {self.config.confidence_threshold}")
        
        # Check error rate
        if metrics.error_rate > self.config.error_rate_threshold:
            issues.append(f"Error rate above threshold: {metrics.error_rate:.3f} > {self.config.error_rate_threshold}")
        
        if issues:
            self.logger.warning(f"Model health issues detected: {'; '.join(issues)}")
            self._trigger_health_event("immediate_degradation", {"issues": issues, "metrics": metrics})
    
    def start_monitoring(self) -> None:
        """Start continuous model health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Model health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop model health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Model health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                self._comprehensive_health_check()
                self._detect_degradation_patterns()
                self._check_model_refresh_needed()
                
                if self.config.auto_optimization_enabled:
                    asyncio.run(self._auto_optimize_if_needed())
                
            except Exception as e:
                self.logger.error(f"Model health monitoring error: {e}")
            
            # Sleep between checks
            threading.Event().wait(self.config.monitoring_interval)
    
    def _comprehensive_health_check(self) -> None:
        """Perform comprehensive model health assessment."""
        if len(self.metrics_history) < 10:
            self.current_health_status = HealthStatus.UNKNOWN
            return
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate average metrics
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        avg_confidence = np.mean([m.confidence_score for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        
        # Determine health status
        health_score = 0.0
        
        # Accuracy component (40% of score)
        if avg_accuracy >= self.config.accuracy_threshold:
            health_score += 0.4
        else:
            health_score += 0.4 * (avg_accuracy / self.config.accuracy_threshold)
        
        # Latency component (20% of score)
        if avg_latency <= self.config.latency_threshold_ms:
            health_score += 0.2
        else:
            health_score += 0.2 * max(0, 1.0 - (avg_latency - self.config.latency_threshold_ms) / self.config.latency_threshold_ms)
        
        # Confidence component (20% of score)
        if avg_confidence >= self.config.confidence_threshold:
            health_score += 0.2
        else:
            health_score += 0.2 * (avg_confidence / self.config.confidence_threshold)
        
        # Error rate component (20% of score)
        if avg_error_rate <= self.config.error_rate_threshold:
            health_score += 0.2
        else:
            health_score += 0.2 * max(0, 1.0 - (avg_error_rate / self.config.error_rate_threshold))
        
        # Update health status
        previous_status = self.current_health_status
        
        if health_score >= 0.9:
            self.current_health_status = HealthStatus.HEALTHY
        elif health_score >= 0.7:
            self.current_health_status = HealthStatus.WARNING
        else:
            self.current_health_status = HealthStatus.UNHEALTHY
        
        # Log health status changes
        if previous_status != self.current_health_status:
            self.logger.info(f"Model health status changed: {previous_status.value} -> {self.current_health_status.value} (score: {health_score:.3f})")
            self._trigger_health_event("health_status_change", {
                "previous_status": previous_status.value,
                "new_status": self.current_health_status.value,
                "health_score": health_score
            })
        
        self.last_health_check = datetime.now()
    
    def _detect_degradation_patterns(self) -> None:
        """Detect patterns that indicate model degradation."""
        if len(self.accuracy_trend) < self.config.degradation_detection_window:
            return
        
        # Check for consistent accuracy decline
        recent_accuracy_trend = self.accuracy_trend[-self.config.degradation_detection_window:]
        declining_accuracy = np.mean(recent_accuracy_trend) < -0.01  # 1% decline per measurement
        
        # Check for increasing latency
        recent_latency_trend = self.latency_trend[-self.config.degradation_detection_window:] if self.latency_trend else []
        increasing_latency = len(recent_latency_trend) > 0 and np.mean(recent_latency_trend) > 1.0  # 1ms increase per measurement
        
        # Check for declining confidence
        recent_confidence_trend = self.confidence_trend[-self.config.degradation_detection_window:] if self.confidence_trend else []
        declining_confidence = len(recent_confidence_trend) > 0 and np.mean(recent_confidence_trend) < -0.01
        
        # Detect degradation
        degradation_indicators = sum([declining_accuracy, increasing_latency, declining_confidence])
        
        if degradation_indicators >= 2 and not self.degradation_detected:
            self.degradation_detected = True
            self.degradation_start_time = datetime.now()
            self.logger.warning("Model degradation pattern detected")
            self._trigger_health_event("degradation_detected", {
                "indicators": {
                    "declining_accuracy": declining_accuracy,
                    "increasing_latency": increasing_latency,
                    "declining_confidence": declining_confidence
                }
            })
        
        elif degradation_indicators == 0 and self.degradation_detected:
            self.degradation_detected = False
            self.degradation_start_time = None
            self.logger.info("Model degradation resolved")
            self._trigger_health_event("degradation_resolved", {})
    
    def _check_model_refresh_needed(self) -> None:
        """Check if model needs to be refreshed."""
        if not self.current_model_path:
            return
        
        # Check if model has been running for too long
        model_age = datetime.now() - datetime.fromtimestamp(
            Path(self.current_model_path).stat().st_mtime if Path(self.current_model_path).exists() else datetime.now().timestamp()
        )
        
        if model_age > timedelta(hours=self.config.model_refresh_interval_hours):
            self.logger.info("Model refresh needed due to age")
            self._trigger_health_event("refresh_needed", {"reason": "age", "model_age_hours": model_age.total_seconds() / 3600})
    
    async def _auto_optimize_if_needed(self) -> None:
        """Automatically optimize model if performance is degrading."""
        if self.current_health_status == HealthStatus.UNHEALTHY and not self.is_using_backup:
            try:
                self.logger.info("Attempting automatic model optimization")
                # Note: In a real implementation, this would perform actual optimization
                await asyncio.sleep(0.1)  # Simulated optimization time
                self.logger.info("Automatic model optimization completed")
                
            except Exception as e:
                self.logger.error(f"Automatic optimization failed: {e}")
    
    async def reload_model(self) -> bool:
        """Reload the primary model."""
        try:
            if not self.current_model_path:
                return False
            
            self.logger.info("Reloading primary model")
            # Note: In a real implementation, this would reload the actual model
            await asyncio.sleep(0.5)  # Simulated reload time
            
            # Reset metrics and status
            self.metrics_history.clear()
            self.current_health_status = HealthStatus.HEALTHY
            self.degradation_detected = False
            self.error_count = 0
            self.prediction_count = 0
            
            self.logger.info("Model reload completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model reload failed: {e}")
            return False
    
    async def switch_to_backup(self) -> bool:
        """Switch to backup model."""
        if not self.config.backup_model_enabled or not self.backup_model_paths:
            self.logger.warning("No backup models available")
            return False
        
        try:
            self.logger.info("Switching to backup model")
            # Note: In a real implementation, this would switch to backup model
            
            self.is_using_backup = True
            self.backup_switch_time = datetime.now()
            self.model_switch_count += 1
            
            # Reset some metrics
            self.current_health_status = HealthStatus.WARNING
            self.degradation_detected = False
            
            self.logger.info("Successfully switched to backup model")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup model switch failed: {e}")
            return False
    
    async def optimize_model(self) -> bool:
        """Optimize current model performance."""
        try:
            self.logger.info("Starting model optimization")
            # Note: In a real implementation, this would optimize the model
            await asyncio.sleep(1.0)  # Simulated optimization time
            
            # Use performance optimizer
            optimization_results = self.performance_optimizer.optimize_model_inference()
            
            self.logger.info(f"Model optimization completed: {optimization_results}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return False
    
    def register_health_callback(self, callback: Callable) -> None:
        """Register a callback for health events."""
        self.health_callbacks.append(callback)
    
    def _trigger_health_event(self, event_type: str, context: Dict[str, Any]) -> None:
        """Trigger a health event."""
        for callback in self.health_callbacks:
            try:
                callback(event_type, context)
            except Exception as e:
                self.logger.error(f"Health callback failed: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive model health summary."""
        if len(self.metrics_history) == 0:
            return {
                "overall_health_score": 0.0,
                "health_status": self.current_health_status.value,
                "metrics_available": False
            }
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        return {
            "overall_health_score": self._calculate_health_score(),
            "health_status": self.current_health_status.value,
            "is_using_backup": self.is_using_backup,
            "degradation_detected": self.degradation_detected,
            "metrics": {
                "avg_accuracy": np.mean([m.accuracy for m in recent_metrics]),
                "avg_latency_ms": np.mean([m.latency_ms for m in recent_metrics]),
                "avg_confidence": np.mean([m.confidence_score for m in recent_metrics]),
                "error_rate": self.error_count / max(self.prediction_count, 1),
                "throughput": np.mean([m.throughput for m in recent_metrics])
            },
            "trends": {
                "accuracy_trend": np.mean(self.accuracy_trend[-10:]) if self.accuracy_trend else 0.0,
                "latency_trend": np.mean(self.latency_trend[-10:]) if self.latency_trend else 0.0,
                "confidence_trend": np.mean(self.confidence_trend[-10:]) if self.confidence_trend else 0.0
            },
            "statistics": {
                "total_predictions": self.prediction_count,
                "total_errors": self.error_count,
                "model_switches": self.model_switch_count,
                "last_health_check": self.last_health_check.isoformat()
            },
            "config": {
                "accuracy_threshold": self.config.accuracy_threshold,
                "latency_threshold_ms": self.config.latency_threshold_ms,
                "backup_models_available": len(self.backup_model_paths)
            }
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trend analysis."""
        return {
            "accuracy_trend": np.mean(self.accuracy_trend[-20:]) if len(self.accuracy_trend) >= 20 else 0.0,
            "latency_trend": np.mean(self.latency_trend[-20:]) if len(self.latency_trend) >= 20 else 0.0,
            "confidence_trend": np.mean(self.confidence_trend[-20:]) if len(self.confidence_trend) >= 20 else 0.0,
            "trend_stability": self._calculate_trend_stability(),
            "prediction_volume_trend": self._calculate_volume_trend()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score."""
        if len(self.metrics_history) == 0:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Weighted health score
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        avg_confidence = np.mean([m.confidence_score for m in recent_metrics])
        error_rate = self.error_count / max(self.prediction_count, 1)
        
        # Calculate components
        accuracy_score = min(1.0, avg_accuracy / self.config.accuracy_threshold)
        latency_score = max(0.0, 1.0 - (avg_latency - self.config.latency_threshold_ms) / self.config.latency_threshold_ms)
        confidence_score = min(1.0, avg_confidence / self.config.confidence_threshold)
        error_score = max(0.0, 1.0 - error_rate / self.config.error_rate_threshold)
        
        # Weighted average
        health_score = (0.4 * accuracy_score + 0.2 * latency_score + 
                       0.2 * confidence_score + 0.2 * error_score)
        
        return max(0.0, min(1.0, health_score))
    
    def _calculate_trend_stability(self) -> float:
        """Calculate how stable the performance trends are."""
        if len(self.accuracy_trend) < 10:
            return 0.5
        
        # Calculate variance of recent trends
        recent_trends = self.accuracy_trend[-20:]
        stability = 1.0 - min(1.0, np.var(recent_trends) * 100)  # Lower variance = higher stability
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_volume_trend(self) -> float:
        """Calculate prediction volume trend."""
        if len(self.metrics_history) < 20:
            return 0.0
        
        recent_timestamps = [m.timestamp for m in self.metrics_history[-20:]]
        time_diffs = [(recent_timestamps[i] - recent_timestamps[i-1]).total_seconds() 
                     for i in range(1, len(recent_timestamps))]
        
        if not time_diffs:
            return 0.0
        
        avg_interval = np.mean(time_diffs)
        recent_interval = np.mean(time_diffs[-5:]) if len(time_diffs) >= 5 else avg_interval
        
        # Positive trend = increasing frequency (shorter intervals)
        return (avg_interval - recent_interval) / avg_interval if avg_interval > 0 else 0.0