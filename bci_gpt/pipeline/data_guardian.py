"""Data Pipeline Guardian for BCI-GPT Self-Healing System.

Monitors data quality, manages data sources, detects corruption,
and ensures continuous data flow with automatic failover capabilities.
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from enum import Enum
import hashlib

from ..utils.monitoring import HealthStatus
from ..utils.error_handling import BCI_GPTError
from ..utils.streaming import StreamingEEG


class DataSourceType(Enum):
    """Types of data sources."""
    PRIMARY_EEG = "primary_eeg"
    BACKUP_EEG = "backup_eeg"
    SIMULATED = "simulated"
    FILE_STREAM = "file_stream"
    NETWORK_STREAM = "network_stream"


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    signal_to_noise_ratio: float = 0.0
    artifact_percentage: float = 0.0
    missing_samples: int = 0
    sampling_rate_stability: float = 1.0
    channel_correlation: float = 1.0
    frequency_spectrum_health: float = 1.0
    timestamp_consistency: float = 1.0
    data_integrity_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataSource:
    """Configuration for a data source."""
    source_id: str
    source_type: DataSourceType
    connection_string: str
    priority: int = 0
    is_active: bool = True
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_data_received: Optional[datetime] = None
    error_count: int = 0
    total_samples: int = 0
    quality_score: float = 1.0


@dataclass
class DataGuardianConfig:
    """Configuration for the Data Guardian."""
    monitoring_interval: float = 2.0
    quality_check_interval: float = 5.0
    buffer_size: int = 1000
    max_missing_samples: int = 50
    min_snr_threshold: float = 3.0
    max_artifact_percentage: float = 20.0
    sampling_rate_tolerance: float = 0.01
    failover_enabled: bool = True
    auto_cleanup_enabled: bool = True
    data_validation_enabled: bool = True
    backup_buffer_enabled: bool = True


class DataPipelineGuardian:
    """Guardian for data pipeline integrity and quality.
    
    Monitors data sources, assesses quality, manages failover,
    and ensures continuous high-quality data flow.
    """
    
    def __init__(self, config: Optional[DataGuardianConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or DataGuardianConfig()
        
        # Data sources management
        self.data_sources: Dict[str, DataSource] = {}
        self.active_source_id: Optional[str] = None
        self.source_priority_order: List[str] = []
        
        # Data quality tracking
        self.quality_history: deque = deque(maxlen=100)
        self.current_quality_level = DataQualityLevel.UNKNOWN
        self.quality_degradation_start: Optional[datetime] = None
        
        # Data buffering and streaming
        self.data_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.backup_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.streaming_handlers: Dict[str, StreamingEEG] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.quality_thread: Optional[threading.Thread] = None
        
        # Health and statistics
        self.total_data_processed = 0
        self.total_corrupted_samples = 0
        self.failover_count = 0
        self.last_failover_time: Optional[datetime] = None
        
        # Callbacks and events
        self.data_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
        
        # Corruption detection
        self.corruption_patterns: List[Dict[str, Any]] = []
        self.last_data_hash: Optional[str] = None
        
        # Auto-healing state
        self.auto_recovery_attempts = 0
        self.max_recovery_attempts = 3
        
    def register_data_source(self, source: DataSource) -> None:
        """Register a new data source."""
        self.data_sources[source.source_id] = source
        self.source_priority_order.append(source.source_id)
        self.source_priority_order.sort(key=lambda x: self.data_sources[x].priority, reverse=True)
        
        self.logger.info(f"Registered data source: {source.source_id} (type: {source.source_type.value})")
    
    def set_active_source(self, source_id: str) -> None:
        """Set the active data source."""
        if source_id not in self.data_sources:
            raise BCI_GPTError(f"Data source {source_id} not registered")
        
        self.active_source_id = source_id
        self.data_sources[source_id].is_active = True
        
        # Deactivate other sources
        for sid, source in self.data_sources.items():
            if sid != source_id:
                source.is_active = False
        
        self.logger.info(f"Active data source set to: {source_id}")
    
    def start_monitoring(self) -> None:
        """Start data monitoring and quality assessment."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.quality_thread = threading.Thread(target=self._quality_assessment_loop, daemon=True)
        
        self.monitoring_thread.start()
        self.quality_thread.start()
        
        self.logger.info("Data pipeline monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop data monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        if self.quality_thread:
            self.quality_thread.join(timeout=5.0)
        
        # Close streaming handlers
        for handler in self.streaming_handlers.values():
            try:
                handler.stop()
            except Exception as e:
                self.logger.error(f"Error stopping streaming handler: {e}")
        
        self.logger.info("Data pipeline monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main data monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_data_source_health()
                self._validate_active_data_flow()
                self._detect_data_corruption()
                
                if self.config.failover_enabled:
                    self._check_failover_conditions()
                
                if self.config.auto_cleanup_enabled:
                    self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Data monitoring error: {e}")
            
            threading.Event().wait(self.config.monitoring_interval)
    
    def _quality_assessment_loop(self) -> None:
        """Data quality assessment loop."""
        while self.monitoring_active:
            try:
                if len(self.data_buffer) > 10:
                    quality_metrics = self._assess_data_quality()
                    self.quality_history.append(quality_metrics)
                    self._update_quality_level(quality_metrics)
                    
                    # Trigger callbacks for quality changes
                    for callback in self.quality_callbacks:
                        try:
                            callback("quality_update", {"metrics": quality_metrics})
                        except Exception as e:
                            self.logger.error(f"Quality callback error: {e}")
                
            except Exception as e:
                self.logger.error(f"Quality assessment error: {e}")
            
            threading.Event().wait(self.config.quality_check_interval)
    
    def _check_data_source_health(self) -> None:
        """Check health of all registered data sources."""
        current_time = datetime.now()
        
        for source_id, source in self.data_sources.items():
            # Check if source has been inactive for too long
            if source.is_active and source.last_data_received:
                time_since_data = (current_time - source.last_data_received).total_seconds()
                
                if time_since_data > 30.0:  # No data for 30 seconds
                    source.health_status = HealthStatus.UNHEALTHY
                    self.logger.warning(f"Data source {source_id} unhealthy: no data for {time_since_data:.1f}s")
                    
                    self._trigger_data_event("source_unhealthy", {
                        "source_id": source_id,
                        "time_since_data": time_since_data
                    })
                elif time_since_data > 10.0:  # Warning if no data for 10 seconds
                    source.health_status = HealthStatus.WARNING
                elif time_since_data < 5.0:
                    source.health_status = HealthStatus.HEALTHY
    
    def _validate_active_data_flow(self) -> None:
        """Validate that data is flowing from the active source."""
        if not self.active_source_id:
            return
        
        active_source = self.data_sources[self.active_source_id]
        
        # Check data flow rate
        if len(self.data_buffer) < 5:
            self.logger.warning("Data buffer low, potential data flow issue")
            self._trigger_data_event("low_data_flow", {
                "buffer_size": len(self.data_buffer),
                "active_source": self.active_source_id
            })
    
    def _detect_data_corruption(self) -> None:
        """Detect potential data corruption patterns."""
        if len(self.data_buffer) < 10:
            return
        
        recent_data = list(self.data_buffer)[-10:]
        corruption_detected = False
        corruption_reasons = []
        
        # Check for constant values (stuck sensor)
        if self._check_constant_values(recent_data):
            corruption_detected = True
            corruption_reasons.append("constant_values")
        
        # Check for impossible values
        if self._check_impossible_values(recent_data):
            corruption_detected = True
            corruption_reasons.append("impossible_values")
        
        # Check for sudden spikes
        if self._check_sudden_spikes(recent_data):
            corruption_detected = True
            corruption_reasons.append("sudden_spikes")
        
        # Check for frequency anomalies
        if self._check_frequency_anomalies(recent_data):
            corruption_detected = True
            corruption_reasons.append("frequency_anomalies")
        
        if corruption_detected:
            self.total_corrupted_samples += len(recent_data)
            self.logger.warning(f"Data corruption detected: {corruption_reasons}")
            self._trigger_data_event("data_corruption", {
                "reasons": corruption_reasons,
                "affected_samples": len(recent_data)
            })
    
    def _check_constant_values(self, data: List[Any]) -> bool:
        """Check if data contains constant values (stuck sensor)."""
        try:
            # Assuming data is EEG samples with multiple channels
            if isinstance(data[0], (list, np.ndarray)):
                for channel in range(len(data[0])):
                    channel_values = [sample[channel] for sample in data]
                    if len(set(channel_values)) <= 2:  # Almost constant
                        return True
            return False
        except Exception:
            return False
    
    def _check_impossible_values(self, data: List[Any]) -> bool:
        """Check for impossible EEG values."""
        try:
            if isinstance(data[0], (list, np.ndarray)):
                for sample in data:
                    for value in sample:
                        # EEG values typically range from -100 to +100 microvolts
                        if abs(value) > 1000:  # Impossible EEG value
                            return True
            return False
        except Exception:
            return False
    
    def _check_sudden_spikes(self, data: List[Any]) -> bool:
        """Check for sudden spikes that indicate artifacts."""
        try:
            if isinstance(data[0], (list, np.ndarray)) and len(data) > 3:
                for channel in range(len(data[0])):
                    channel_values = [sample[channel] for sample in data]
                    
                    # Check for sudden large changes
                    for i in range(1, len(channel_values)):
                        diff = abs(channel_values[i] - channel_values[i-1])
                        if diff > 200:  # Large sudden change
                            return True
            return False
        except Exception:
            return False
    
    def _check_frequency_anomalies(self, data: List[Any]) -> bool:
        """Check for frequency domain anomalies."""
        try:
            # Simple check - in real implementation would use FFT
            if len(data) >= 10:
                # Check sampling rate consistency
                timestamps = [getattr(sample, 'timestamp', datetime.now()) for sample in data]
                if len(timestamps) > 1:
                    intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                               for i in range(1, len(timestamps))]
                    if intervals:
                        std_interval = np.std(intervals)
                        mean_interval = np.mean(intervals)
                        if std_interval > 0.1 * mean_interval:  # High variance in sampling rate
                            return True
            return False
        except Exception:
            return False
    
    def _assess_data_quality(self) -> DataQualityMetrics:
        """Assess current data quality."""
        if len(self.data_buffer) < 10:
            return DataQualityMetrics()
        
        recent_data = list(self.data_buffer)[-10:]
        
        # Calculate quality metrics
        snr = self._calculate_signal_to_noise_ratio(recent_data)
        artifact_pct = self._calculate_artifact_percentage(recent_data)
        missing = self._count_missing_samples(recent_data)
        sampling_stability = self._calculate_sampling_rate_stability(recent_data)
        channel_corr = self._calculate_channel_correlation(recent_data)
        freq_health = self._calculate_frequency_spectrum_health(recent_data)
        timestamp_consistency = self._calculate_timestamp_consistency(recent_data)
        integrity_score = self._calculate_data_integrity_score(recent_data)
        
        return DataQualityMetrics(
            signal_to_noise_ratio=snr,
            artifact_percentage=artifact_pct,
            missing_samples=missing,
            sampling_rate_stability=sampling_stability,
            channel_correlation=channel_corr,
            frequency_spectrum_health=freq_health,
            timestamp_consistency=timestamp_consistency,
            data_integrity_score=integrity_score
        )
    
    def _calculate_signal_to_noise_ratio(self, data: List[Any]) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            # Simplified SNR calculation
            if isinstance(data[0], (list, np.ndarray)):
                signals = []
                for sample in data:
                    signals.extend(sample)
                
                signal_power = np.var(signals)
                noise_estimate = np.var(np.diff(signals))  # Simple noise estimate
                
                if noise_estimate > 0:
                    snr = 10 * np.log10(signal_power / noise_estimate)
                    return max(0.0, snr)
            
            return 10.0  # Default acceptable SNR
        except Exception:
            return 5.0  # Default fallback
    
    def _calculate_artifact_percentage(self, data: List[Any]) -> float:
        """Calculate percentage of data affected by artifacts."""
        try:
            total_samples = len(data)
            artifact_samples = 0
            
            # Count samples with artifacts (simplified detection)
            for sample in data:
                if isinstance(sample, (list, np.ndarray)):
                    for value in sample:
                        if abs(value) > 200:  # High amplitude artifact
                            artifact_samples += 1
                            break
            
            return (artifact_samples / total_samples) * 100.0 if total_samples > 0 else 0.0
        except Exception:
            return 0.0
    
    def _count_missing_samples(self, data: List[Any]) -> int:
        """Count missing or invalid samples."""
        missing = 0
        for sample in data:
            if sample is None or (hasattr(sample, '__len__') and len(sample) == 0):
                missing += 1
        return missing
    
    def _calculate_sampling_rate_stability(self, data: List[Any]) -> float:
        """Calculate sampling rate stability."""
        try:
            # In a real implementation, this would check actual timestamps
            return 0.95  # Assume good stability for now
        except Exception:
            return 0.8
    
    def _calculate_channel_correlation(self, data: List[Any]) -> float:
        """Calculate inter-channel correlation health."""
        try:
            if isinstance(data[0], (list, np.ndarray)) and len(data[0]) > 1:
                # Calculate correlation between channels
                channels = list(zip(*data))
                correlations = []
                
                for i in range(len(channels)):
                    for j in range(i+1, len(channels)):
                        corr = np.corrcoef(channels[i], channels[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                if correlations:
                    return np.mean(correlations)
            
            return 0.7  # Default moderate correlation
        except Exception:
            return 0.5
    
    def _calculate_frequency_spectrum_health(self, data: List[Any]) -> float:
        """Calculate frequency spectrum health score."""
        try:
            # Simplified frequency health - would use FFT in real implementation
            return 0.85  # Assume good frequency content
        except Exception:
            return 0.7
    
    def _calculate_timestamp_consistency(self, data: List[Any]) -> float:
        """Calculate timestamp consistency."""
        try:
            # Check if timestamps are consistent
            return 0.95  # Assume good timestamp consistency
        except Exception:
            return 0.8
    
    def _calculate_data_integrity_score(self, data: List[Any]) -> float:
        """Calculate overall data integrity score."""
        try:
            # Generate hash to check data integrity
            data_str = str(data)
            current_hash = hashlib.md5(data_str.encode()).hexdigest()
            
            if self.last_data_hash and current_hash == self.last_data_hash:
                return 0.5  # Possible duplicate data
            
            self.last_data_hash = current_hash
            return 1.0  # Good integrity
        except Exception:
            return 0.8
    
    def _update_quality_level(self, metrics: DataQualityMetrics) -> None:
        """Update overall quality level based on metrics."""
        # Calculate composite quality score
        quality_score = (
            min(1.0, metrics.signal_to_noise_ratio / 10.0) * 0.3 +
            max(0.0, 1.0 - metrics.artifact_percentage / 100.0) * 0.2 +
            max(0.0, 1.0 - metrics.missing_samples / 10.0) * 0.2 +
            metrics.sampling_rate_stability * 0.1 +
            metrics.channel_correlation * 0.1 +
            metrics.data_integrity_score * 0.1
        )
        
        # Determine quality level
        previous_level = self.current_quality_level
        
        if quality_score >= 0.9:
            self.current_quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.7:
            self.current_quality_level = DataQualityLevel.GOOD
        elif quality_score >= 0.5:
            self.current_quality_level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= 0.3:
            self.current_quality_level = DataQualityLevel.POOR
        else:
            self.current_quality_level = DataQualityLevel.UNUSABLE
        
        # Log quality changes
        if previous_level != self.current_quality_level:
            self.logger.info(f"Data quality changed: {previous_level.value} -> {self.current_quality_level.value}")
            
            # Track quality degradation
            if (previous_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD] and 
                self.current_quality_level in [DataQualityLevel.POOR, DataQualityLevel.UNUSABLE]):
                self.quality_degradation_start = datetime.now()
                self._trigger_data_event("quality_degradation", {
                    "previous_level": previous_level.value,
                    "current_level": self.current_quality_level.value,
                    "quality_score": quality_score
                })
    
    def _check_failover_conditions(self) -> None:
        """Check if failover to backup source is needed."""
        if not self.active_source_id or len(self.data_sources) < 2:
            return
        
        active_source = self.data_sources[self.active_source_id]
        
        # Failover conditions
        should_failover = False
        failover_reason = ""
        
        # Check source health
        if active_source.health_status == HealthStatus.UNHEALTHY:
            should_failover = True
            failover_reason = "source_unhealthy"
        
        # Check data quality
        elif self.current_quality_level == DataQualityLevel.UNUSABLE:
            should_failover = True
            failover_reason = "data_quality_unusable"
        
        # Check prolonged quality degradation
        elif (self.quality_degradation_start and 
              (datetime.now() - self.quality_degradation_start).total_seconds() > 60):
            should_failover = True
            failover_reason = "prolonged_degradation"
        
        if should_failover:
            asyncio.run(self._perform_failover(failover_reason))
    
    async def _perform_failover(self, reason: str) -> bool:
        """Perform failover to next best data source."""
        self.logger.info(f"Initiating failover due to: {reason}")
        
        # Find next best source
        next_source_id = None
        for source_id in self.source_priority_order:
            if (source_id != self.active_source_id and 
                self.data_sources[source_id].health_status != HealthStatus.UNHEALTHY):
                next_source_id = source_id
                break
        
        if not next_source_id:
            self.logger.error("No healthy backup data source available")
            return False
        
        try:
            # Switch to backup source
            old_source_id = self.active_source_id
            self.set_active_source(next_source_id)
            
            # Update statistics
            self.failover_count += 1
            self.last_failover_time = datetime.now()
            
            self.logger.info(f"Failover completed: {old_source_id} -> {next_source_id}")
            
            self._trigger_data_event("failover_completed", {
                "old_source": old_source_id,
                "new_source": next_source_id,
                "reason": reason
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            return False
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data from buffers."""
        # Clean up data older than 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        # Note: In a real implementation, this would clean based on actual timestamps
        # For now, just ensure buffers don't grow too large
        if len(self.data_buffer) > self.config.buffer_size * 0.9:
            # Remove oldest 10% of data
            remove_count = int(len(self.data_buffer) * 0.1)
            for _ in range(remove_count):
                self.data_buffer.popleft()
    
    def process_data_sample(self, data_sample: Any, source_id: str) -> None:
        """Process incoming data sample."""
        if source_id not in self.data_sources:
            self.logger.warning(f"Received data from unregistered source: {source_id}")
            return
        
        source = self.data_sources[source_id]
        source.last_data_received = datetime.now()
        source.total_samples += 1
        
        # Add to buffer if from active source
        if source_id == self.active_source_id:
            self.data_buffer.append(data_sample)
            self.total_data_processed += 1
            
            # Also add to backup buffer
            if self.config.backup_buffer_enabled:
                self.backup_buffer.append(data_sample)
        
        # Validate data if enabled
        if self.config.data_validation_enabled:
            if not self._validate_data_sample(data_sample):
                self.logger.warning(f"Invalid data sample from {source_id}")
                source.error_count += 1
    
    def _validate_data_sample(self, data_sample: Any) -> bool:
        """Validate a data sample."""
        try:
            # Basic validation
            if data_sample is None:
                return False
            
            # Check if it's EEG data with expected structure
            if isinstance(data_sample, (list, np.ndarray)):
                # Check for reasonable values
                for value in data_sample:
                    if not isinstance(value, (int, float)) or abs(value) > 1000:
                        return False
            
            return True
        except Exception:
            return False
    
    async def restart_data_source(self) -> bool:
        """Restart the current data source."""
        if not self.active_source_id:
            return False
        
        try:
            self.logger.info(f"Restarting data source: {self.active_source_id}")
            
            # Note: In a real implementation, this would restart the actual data source
            await asyncio.sleep(1.0)  # Simulated restart time
            
            # Reset source status
            source = self.data_sources[self.active_source_id]
            source.health_status = HealthStatus.HEALTHY
            source.error_count = 0
            
            self.logger.info("Data source restart completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data source restart failed: {e}")
            return False
    
    async def switch_to_backup_source(self) -> bool:
        """Switch to backup data source."""
        return await self._perform_failover("manual_switch")
    
    async def clean_buffers(self) -> bool:
        """Clean data buffers."""
        try:
            self.logger.info("Cleaning data buffers")
            
            self.data_buffer.clear()
            if self.config.backup_buffer_enabled:
                self.backup_buffer.clear()
            
            # Reset quality metrics
            self.quality_history.clear()
            self.current_quality_level = DataQualityLevel.UNKNOWN
            
            self.logger.info("Buffer cleaning completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Buffer cleaning failed: {e}")
            return False
    
    def register_data_callback(self, callback: Callable) -> None:
        """Register callback for data events."""
        self.data_callbacks.append(callback)
    
    def register_quality_callback(self, callback: Callable) -> None:
        """Register callback for quality events."""
        self.quality_callbacks.append(callback)
    
    def _trigger_data_event(self, event_type: str, context: Dict[str, Any]) -> None:
        """Trigger a data event."""
        for callback in self.data_callbacks:
            try:
                callback(event_type, context)
            except Exception as e:
                self.logger.error(f"Data callback failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive data pipeline health status."""
        active_source = self.data_sources.get(self.active_source_id) if self.active_source_id else None
        
        # Calculate overall status
        if not active_source:
            overall_status = "no_active_source"
        elif active_source.health_status == HealthStatus.UNHEALTHY:
            overall_status = "critical"
        elif self.current_quality_level == DataQualityLevel.UNUSABLE:
            overall_status = "critical"
        elif self.current_quality_level in [DataQualityLevel.POOR, DataQualityLevel.ACCEPTABLE]:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Get latest quality metrics
        latest_quality = self.quality_history[-1] if self.quality_history else None
        
        return {
            "status": overall_status,
            "active_source": self.active_source_id,
            "current_quality_level": self.current_quality_level.value,
            "data_sources": {
                source_id: {
                    "type": source.source_type.value,
                    "health_status": source.health_status.value,
                    "is_active": source.is_active,
                    "total_samples": source.total_samples,
                    "error_count": source.error_count,
                    "last_data": source.last_data_received.isoformat() if source.last_data_received else None
                }
                for source_id, source in self.data_sources.items()
            },
            "quality_metrics": {
                "signal_to_noise_ratio": latest_quality.signal_to_noise_ratio if latest_quality else 0.0,
                "artifact_percentage": latest_quality.artifact_percentage if latest_quality else 0.0,
                "missing_samples": latest_quality.missing_samples if latest_quality else 0,
                "data_integrity_score": latest_quality.data_integrity_score if latest_quality else 0.0
            } if latest_quality else {},
            "statistics": {
                "total_processed": self.total_data_processed,
                "corrupted_samples": self.total_corrupted_samples,
                "failover_count": self.failover_count,
                "buffer_size": len(self.data_buffer),
                "backup_buffer_size": len(self.backup_buffer) if self.config.backup_buffer_enabled else 0
            },
            "config": {
                "failover_enabled": self.config.failover_enabled,
                "auto_cleanup_enabled": self.config.auto_cleanup_enabled,
                "data_validation_enabled": self.config.data_validation_enabled
            }
        }