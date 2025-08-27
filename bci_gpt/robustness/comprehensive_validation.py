#!/usr/bin/env python3
"""
Comprehensive Validation Framework for BCI-GPT System
Generation 2: Robust data validation with clinical-grade quality assurance
"""

import numpy as np
import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of data validation with detailed metrics."""
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[Dict[str, Any]]
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime
    
    def add_issue(self, severity: ValidationSeverity, message: str, details: Dict[str, Any] = None):
        """Add validation issue."""
        self.issues.append({
            "severity": severity.value,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[Dict[str, Any]]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue["severity"] == severity.value]

class EEGSignalValidator:
    """Comprehensive EEG signal validation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Clinical-grade validation thresholds
        self.quality_thresholds = {
            "min_sampling_rate": 250,  # Hz
            "max_sampling_rate": 4000,  # Hz
            "min_amplitude": -500,      # microvolts
            "max_amplitude": 500,       # microvolts
            "max_artifact_ratio": 0.3,  # 30% artifacts maximum
            "min_signal_duration": 0.5, # seconds
            "max_signal_duration": 300,  # seconds
            "impedance_threshold": 50000,  # ohms
        }
    
    def validate_signal_quality(self, eeg_data: Dict[str, Any]) -> ValidationResult:
        """Comprehensive EEG signal quality validation."""
        result = ValidationResult(
            passed=True,
            score=1.0,
            issues=[],
            metrics={},
            recommendations=[],
            timestamp=datetime.now()
        )
        
        try:
            # Extract signal data
            signal = np.array(eeg_data.get("data", []))
            sampling_rate = eeg_data.get("sampling_rate", 0)
            n_channels = eeg_data.get("n_channels", 0)
            
            # Basic structure validation
            self._validate_basic_structure(signal, sampling_rate, n_channels, result)
            
            # Signal quality metrics
            self._validate_signal_metrics(signal, sampling_rate, result)
            
            # Artifact detection
            self._validate_artifacts(signal, result)
            
            # Channel quality assessment
            if signal.ndim == 2:
                self._validate_channel_quality(signal, result)
            
            # Impedance check (if available)
            if "impedances" in eeg_data:
                self._validate_impedances(eeg_data["impedances"], result)
            
            # Calculate overall score
            result.score = self._calculate_quality_score(result)
            result.passed = result.score >= 0.7 and len(result.get_issues_by_severity(ValidationSeverity.CRITICAL)) == 0
            
        except Exception as e:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Validation failed: {str(e)}",
                {"exception_type": type(e).__name__}
            )
            result.passed = False
            result.score = 0.0
        
        return result
    
    def _validate_basic_structure(self, signal: np.ndarray, sampling_rate: int, n_channels: int, result: ValidationResult):
        """Validate basic signal structure."""
        # Check sampling rate
        if sampling_rate < self.quality_thresholds["min_sampling_rate"]:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Sampling rate {sampling_rate} Hz below minimum {self.quality_thresholds['min_sampling_rate']} Hz"
            )
        elif sampling_rate > self.quality_thresholds["max_sampling_rate"]:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Sampling rate {sampling_rate} Hz above typical maximum {self.quality_thresholds['max_sampling_rate']} Hz"
            )
        
        result.metrics["sampling_rate"] = sampling_rate
        
        # Check signal dimensions
        if signal.size == 0:
            result.add_issue(ValidationSeverity.CRITICAL, "Signal is empty")
            return
        
        # Check signal duration
        if signal.ndim == 1:
            duration = len(signal) / max(sampling_rate, 1)
        elif signal.ndim == 2:
            duration = signal.shape[1] / max(sampling_rate, 1)
        else:
            result.add_issue(ValidationSeverity.ERROR, f"Invalid signal dimensions: {signal.shape}")
            return
        
        result.metrics["duration"] = duration
        
        if duration < self.quality_thresholds["min_signal_duration"]:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Signal duration {duration:.2f}s below minimum {self.quality_thresholds['min_signal_duration']}s"
            )
        elif duration > self.quality_thresholds["max_signal_duration"]:
            result.add_issue(
                ValidationSeverity.INFO,
                f"Signal duration {duration:.2f}s exceeds typical maximum {self.quality_thresholds['max_signal_duration']}s"
            )
    
    def _validate_signal_metrics(self, signal: np.ndarray, sampling_rate: int, result: ValidationResult):
        """Validate signal amplitude and statistical metrics."""
        if signal.size == 0:
            return
        
        # Amplitude validation
        min_amp = np.min(signal)
        max_amp = np.max(signal)
        
        result.metrics["min_amplitude"] = min_amp
        result.metrics["max_amplitude"] = max_amp
        
        if min_amp < self.quality_thresholds["min_amplitude"]:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Minimum amplitude {min_amp:.2f}µV below typical range"
            )
        
        if max_amp > self.quality_thresholds["max_amplitude"]:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Maximum amplitude {max_amp:.2f}µV above typical range"
            )
        
        # Statistical metrics
        result.metrics["mean_amplitude"] = np.mean(signal)
        result.metrics["std_amplitude"] = np.std(signal)
        result.metrics["signal_variance"] = np.var(signal)
        
        # Signal-to-noise ratio estimation
        if signal.ndim == 1:
            snr_estimate = self._estimate_snr(signal)
            result.metrics["snr_estimate"] = snr_estimate
            
            if snr_estimate < 10:  # dB
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Low signal-to-noise ratio: {snr_estimate:.1f} dB"
                )
    
    def _validate_artifacts(self, signal: np.ndarray, result: ValidationResult):
        """Detect and validate artifacts in EEG signal."""
        if signal.size == 0:
            return
        
        # Simple artifact detection (would use more sophisticated methods in production)
        artifact_count = 0
        total_samples = signal.size
        
        # Detect amplitude artifacts (values outside normal range)
        amplitude_artifacts = np.sum((signal < -200) | (signal > 200))
        artifact_count += amplitude_artifacts
        
        # Detect gradient artifacts (sudden jumps)
        if signal.ndim == 1:
            gradients = np.abs(np.diff(signal))
            gradient_artifacts = np.sum(gradients > 50)  # Arbitrary threshold
            artifact_count += gradient_artifacts
        
        artifact_ratio = artifact_count / total_samples
        result.metrics["artifact_ratio"] = artifact_ratio
        
        if artifact_ratio > self.quality_thresholds["max_artifact_ratio"]:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Artifact ratio {artifact_ratio:.2%} exceeds maximum {self.quality_thresholds['max_artifact_ratio']:.2%}",
                {"artifact_count": artifact_count, "total_samples": total_samples}
            )
        elif artifact_ratio > 0.1:  # 10%
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Elevated artifact ratio: {artifact_ratio:.2%}"
            )
    
    def _validate_channel_quality(self, signal: np.ndarray, result: ValidationResult):
        """Validate individual channel quality."""
        n_channels = signal.shape[0]
        channel_quality = []
        
        for ch in range(n_channels):
            ch_signal = signal[ch, :]
            
            # Channel-specific metrics
            ch_std = np.std(ch_signal)
            ch_mean = np.mean(ch_signal)
            
            # Detect flat channels (potential electrode issues)
            if ch_std < 0.1:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Channel {ch} appears flat (std: {ch_std:.3f})",
                    {"channel": ch, "std": ch_std}
                )
                channel_quality.append(0.0)
            else:
                # Simple quality score based on signal characteristics
                quality_score = min(1.0, ch_std / 10.0)  # Normalize to reasonable EEG std
                channel_quality.append(quality_score)
        
        result.metrics["channel_quality_scores"] = channel_quality
        result.metrics["mean_channel_quality"] = np.mean(channel_quality)
        
        # Flag channels with poor quality
        poor_channels = [i for i, q in enumerate(channel_quality) if q < 0.3]
        if poor_channels:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Poor quality channels detected: {poor_channels}"
            )
    
    def _validate_impedances(self, impedances: List[float], result: ValidationResult):
        """Validate electrode impedances."""
        result.metrics["impedances"] = impedances
        
        high_impedance_channels = []
        for i, impedance in enumerate(impedances):
            if impedance > self.quality_thresholds["impedance_threshold"]:
                high_impedance_channels.append(i)
        
        if high_impedance_channels:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"High impedance channels: {high_impedance_channels}",
                {"threshold": self.quality_thresholds["impedance_threshold"]}
            )
        
        result.metrics["mean_impedance"] = np.mean(impedances)
        result.metrics["max_impedance"] = np.max(impedances)
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Simple SNR estimation (would use more sophisticated methods in production)
        signal_power = np.mean(signal ** 2)
        
        # Estimate noise from high-frequency components
        if len(signal) > 100:
            diff_signal = np.diff(signal)
            noise_power = np.mean(diff_signal ** 2)
        else:
            noise_power = np.var(signal) * 0.1  # Rough estimate
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-10))
            return snr_db
        
        return float('inf')
    
    def _calculate_quality_score(self, result: ValidationResult) -> float:
        """Calculate overall quality score."""
        base_score = 1.0
        
        # Penalize based on issue severity
        for issue in result.issues:
            severity = issue["severity"]
            if severity == ValidationSeverity.CRITICAL.value:
                base_score -= 0.5
            elif severity == ValidationSeverity.ERROR.value:
                base_score -= 0.2
            elif severity == ValidationSeverity.WARNING.value:
                base_score -= 0.1
        
        # Bonus for good metrics
        if "snr_estimate" in result.metrics and result.metrics["snr_estimate"] > 20:
            base_score += 0.1
        
        if "artifact_ratio" in result.metrics and result.metrics["artifact_ratio"] < 0.05:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))

class ModelOutputValidator:
    """Validate model predictions and outputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> ValidationResult:
        """Validate model prediction output."""
        result = ValidationResult(
            passed=True,
            score=1.0,
            issues=[],
            metrics={},
            recommendations=[],
            timestamp=datetime.now()
        )
        
        # Required fields validation
        required_fields = ["predicted_text", "confidence"]
        for field in required_fields:
            if field not in prediction:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Missing required field: {field}"
                )
        
        # Confidence validation
        if "confidence" in prediction:
            confidence = prediction["confidence"]
            result.metrics["confidence"] = confidence
            
            if not (0.0 <= confidence <= 1.0):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Confidence {confidence} outside valid range [0.0, 1.0]"
                )
            elif confidence < 0.1:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Very low confidence: {confidence:.2%}"
                )
        
        # Text validation
        if "predicted_text" in prediction:
            text = prediction["predicted_text"]
            result.metrics["text_length"] = len(text)
            
            if not text or not text.strip():
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Empty or whitespace-only prediction"
                )
            elif len(text) > 1000:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Unusually long prediction: {len(text)} characters"
                )
        
        # Latency validation
        if "latency_ms" in prediction:
            latency = prediction["latency_ms"]
            result.metrics["latency_ms"] = latency
            
            if latency > 200:  # ms
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"High latency: {latency}ms exceeds target <200ms"
                )
        
        result.score = self._calculate_prediction_score(result)
        result.passed = len(result.get_issues_by_severity(ValidationSeverity.ERROR)) == 0
        
        return result
    
    def _calculate_prediction_score(self, result: ValidationResult) -> float:
        """Calculate prediction quality score."""
        base_score = 1.0
        
        # Penalize based on issues
        for issue in result.issues:
            severity = issue["severity"]
            if severity == ValidationSeverity.ERROR.value:
                base_score -= 0.3
            elif severity == ValidationSeverity.WARNING.value:
                base_score -= 0.1
        
        # Bonus for high confidence
        if "confidence" in result.metrics and result.metrics["confidence"] > 0.8:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))

# Example usage and testing
if __name__ == "__main__":
    print("🔍 Testing Comprehensive Validation Framework...")
    
    # Test EEG validation
    eeg_validator = EEGSignalValidator()
    
    # Good signal
    good_eeg = {
        "data": np.random.normal(0, 10, (8, 1000)).tolist(),  # 8 channels, 1000 samples
        "sampling_rate": 1000,
        "n_channels": 8,
        "impedances": [5000, 6000, 4000, 5500, 6200, 4800, 5300, 5800]
    }
    
    result = eeg_validator.validate_signal_quality(good_eeg)
    print(f"✅ Good EEG Signal - Passed: {result.passed}, Score: {result.score:.2f}")
    print(f"   Issues: {len(result.issues)}, Metrics: {len(result.metrics)}")
    
    # Bad signal (artifacts)
    bad_eeg = {
        "data": np.concatenate([
            np.random.normal(0, 10, 500),  # Normal signal
            np.random.normal(0, 1000, 500)  # High artifacts
        ]).tolist(),
        "sampling_rate": 100,  # Too low
        "n_channels": 1
    }
    
    result = eeg_validator.validate_signal_quality(bad_eeg)
    print(f"❌ Bad EEG Signal - Passed: {result.passed}, Score: {result.score:.2f}")
    print(f"   Issues: {len(result.issues)}")
    
    # Test prediction validation
    model_validator = ModelOutputValidator()
    
    good_prediction = {
        "predicted_text": "hello world",
        "confidence": 0.85,
        "latency_ms": 45,
        "token_probabilities": {"hello": 0.9, "world": 0.8}
    }
    
    result = model_validator.validate_prediction(good_prediction)
    print(f"✅ Good Prediction - Passed: {result.passed}, Score: {result.score:.2f}")
    
    bad_prediction = {
        "predicted_text": "",
        "confidence": -0.5,  # Invalid range
        "latency_ms": 500    # Too high
    }
    
    result = model_validator.validate_prediction(bad_prediction)
    print(f"❌ Bad Prediction - Passed: {result.passed}, Score: {result.score:.2f}")
    print(f"   Issues: {[issue['message'] for issue in result.issues]}")
    
    print("\n🎯 Validation Framework Ready!")
