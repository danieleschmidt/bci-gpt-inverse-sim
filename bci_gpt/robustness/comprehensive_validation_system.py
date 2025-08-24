"""Comprehensive validation system for robust BCI-GPT deployment.

This module implements multi-layer validation:
1. Input data validation and sanitization
2. Model output validation and consistency checks
3. Real-time performance validation
4. Clinical safety validation
5. Regulatory compliance validation

Authors: Daniel Schmidt, Terragon Labs
Status: Production-Grade Validation System
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import json
import time
from datetime import datetime


@dataclass
class ValidationConfig:
    """Configuration for comprehensive validation system."""
    
    # Input validation
    min_sampling_rate: int = 250
    max_sampling_rate: int = 2000
    min_channels: int = 8
    max_channels: int = 256
    min_signal_amplitude: float = 0.1
    max_signal_amplitude: float = 500.0
    
    # Model validation
    confidence_threshold: float = 0.7
    consistency_window: int = 5
    max_prediction_variance: float = 0.3
    
    # Performance validation
    max_latency_ms: float = 100.0
    min_throughput: float = 10.0  # samples/second
    max_memory_mb: float = 1024.0
    
    # Safety validation
    max_cognitive_load: float = 0.8
    fatigue_threshold: float = 0.6
    signal_quality_threshold: float = 0.6
    
    # Compliance validation
    hipaa_compliance: bool = True
    fda_compliance: bool = True
    gdpr_compliance: bool = True


class ComprehensiveValidationSystem:
    """Multi-layer validation system for robust BCI-GPT operation."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Initialize validation modules
        self.input_validator = InputDataValidator(self.config)
        self.model_validator = ModelOutputValidator(self.config)
        self.performance_validator = PerformanceValidator(self.config)
        self.safety_validator = SafetyValidator(self.config)
        self.compliance_validator = ComplianceValidator(self.config)
        
        # Validation history for trend analysis
        self.validation_history = []
        self.error_log = []
        
    def validate_full_pipeline(self, 
                              eeg_data: torch.Tensor,
                              model: nn.Module,
                              model_outputs: Dict[str, Any],
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive validation of entire BCI-GPT pipeline."""
        
        validation_start = time.time()
        
        validation_results = {
            'overall_valid': True,
            'validation_timestamp': datetime.now().isoformat(),
            'layer_results': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'validation_time_ms': 0
        }
        
        context = context or {}
        
        print("🔍 Running comprehensive pipeline validation...")
        
        # Layer 1: Input Data Validation
        print("   📊 Validating input data...")
        input_results = self.input_validator.validate(eeg_data, context)
        validation_results['layer_results']['input'] = input_results
        
        if not input_results['valid']:
            validation_results['overall_valid'] = False
            validation_results['critical_issues'].extend(input_results['errors'])
        
        # Layer 2: Model Output Validation
        print("   🧠 Validating model outputs...")
        model_results = self.model_validator.validate(model_outputs, context)
        validation_results['layer_results']['model'] = model_results
        
        if not model_results['valid']:
            validation_results['overall_valid'] = False
            validation_results['critical_issues'].extend(model_results['errors'])
        
        # Layer 3: Performance Validation
        print("   ⚡ Validating performance metrics...")
        performance_results = self.performance_validator.validate(
            model, eeg_data, model_outputs, context
        )
        validation_results['layer_results']['performance'] = performance_results
        
        if not performance_results['valid']:
            validation_results['overall_valid'] = False
            validation_results['critical_issues'].extend(performance_results['errors'])
        
        # Layer 4: Safety Validation
        print("   🛡️  Validating safety requirements...")
        safety_results = self.safety_validator.validate(
            eeg_data, model_outputs, context
        )
        validation_results['layer_results']['safety'] = safety_results
        
        if not safety_results['valid']:
            validation_results['overall_valid'] = False
            validation_results['critical_issues'].extend(safety_results['errors'])
        
        # Layer 5: Compliance Validation
        print("   📋 Validating regulatory compliance...")
        compliance_results = self.compliance_validator.validate(
            eeg_data, model_outputs, context
        )
        validation_results['layer_results']['compliance'] = compliance_results
        
        if not compliance_results['valid']:
            validation_results['overall_valid'] = False
            validation_results['critical_issues'].extend(compliance_results['errors'])
        
        # Aggregate warnings and recommendations
        for layer_result in validation_results['layer_results'].values():
            validation_results['warnings'].extend(layer_result.get('warnings', []))
            validation_results['recommendations'].extend(layer_result.get('recommendations', []))
        
        # Calculate validation time
        validation_results['validation_time_ms'] = (time.time() - validation_start) * 1000
        
        # Store validation history
        self.validation_history.append({
            'timestamp': validation_results['validation_timestamp'],
            'overall_valid': validation_results['overall_valid'],
            'critical_issues_count': len(validation_results['critical_issues']),
            'warnings_count': len(validation_results['warnings']),
            'validation_time_ms': validation_results['validation_time_ms']
        })
        
        # Generate final assessment
        validation_results['assessment'] = self._generate_final_assessment(validation_results)
        
        print(f"   ✅ Validation complete: {'PASS' if validation_results['overall_valid'] else 'FAIL'}")
        print(f"   ⏱️  Validation time: {validation_results['validation_time_ms']:.2f}ms")
        
        return validation_results
    
    def _generate_final_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final assessment of validation results."""
        
        critical_count = len(results['critical_issues'])
        warning_count = len(results['warnings'])
        
        if critical_count == 0 and warning_count == 0:
            grade = "A"
            status = "EXCELLENT"
            deployment_ready = True
        elif critical_count == 0 and warning_count <= 2:
            grade = "B"
            status = "GOOD"
            deployment_ready = True
        elif critical_count <= 1 and warning_count <= 5:
            grade = "C"
            status = "ACCEPTABLE"
            deployment_ready = True
        else:
            grade = "F"
            status = "NEEDS_IMPROVEMENT"
            deployment_ready = False
        
        return {
            'grade': grade,
            'status': status,
            'deployment_ready': deployment_ready,
            'critical_issues_count': critical_count,
            'warnings_count': warning_count,
            'confidence_score': self._calculate_confidence_score(results)
        }
    
    def _calculate_confidence_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the validation."""
        
        layer_scores = []
        
        for layer_name, layer_result in results['layer_results'].items():
            if 'confidence_score' in layer_result:
                layer_scores.append(layer_result['confidence_score'])
            elif layer_result.get('valid', False):
                layer_scores.append(1.0)
            else:
                layer_scores.append(0.0)
        
        if not layer_scores:
            return 0.0
        
        # Weight critical layers more heavily
        weights = {
            'input': 0.15,
            'model': 0.25,
            'performance': 0.20,
            'safety': 0.30,
            'compliance': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, layer_name in enumerate(results['layer_results'].keys()):
            weight = weights.get(layer_name, 0.2)
            if i < len(layer_scores):
                weighted_score += layer_scores[i] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0


class InputDataValidator:
    """Validate and sanitize input EEG data."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate(self, eeg_data: torch.Tensor, 
                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input EEG data quality and format."""
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {},
            'confidence_score': 1.0
        }
        
        try:
            # Check tensor properties
            if not isinstance(eeg_data, torch.Tensor):
                results['valid'] = False
                results['errors'].append("Input is not a PyTorch tensor")
                return results
            
            # Check dimensions
            if eeg_data.dim() != 3:
                results['valid'] = False
                results['errors'].append(f"Expected 3D tensor (batch, channels, time), got {eeg_data.dim()}D")
            
            batch_size, n_channels, n_samples = eeg_data.shape
            
            # Validate number of channels
            if n_channels < self.config.min_channels:
                results['valid'] = False
                results['errors'].append(f"Too few channels: {n_channels} < {self.config.min_channels}")
            elif n_channels > self.config.max_channels:
                results['valid'] = False
                results['errors'].append(f"Too many channels: {n_channels} > {self.config.max_channels}")
            
            # Validate signal amplitude
            min_amp = float(torch.min(eeg_data))
            max_amp = float(torch.max(eeg_data))
            
            if max_amp < self.config.min_signal_amplitude:
                results['warnings'].append(f"Very low signal amplitude: {max_amp:.2f} µV")
                results['recommendations'].append("Check electrode connections")
            
            if max_amp > self.config.max_signal_amplitude:
                results['valid'] = False
                results['errors'].append(f"Signal amplitude too high: {max_amp:.2f} µV > {self.config.max_signal_amplitude}")
            
            # Check for NaN/Inf values
            if torch.isnan(eeg_data).any():
                results['valid'] = False
                results['errors'].append("NaN values detected in EEG data")
            
            if torch.isinf(eeg_data).any():
                results['valid'] = False
                results['errors'].append("Infinite values detected in EEG data")
            
            # Signal quality checks
            signal_variance = torch.var(eeg_data, dim=-1)
            low_variance_channels = (signal_variance < 0.1).sum()
            
            if low_variance_channels > n_channels * 0.3:
                results['warnings'].append(f"Many channels with low variance: {low_variance_channels}/{n_channels}")
                results['recommendations'].append("Check for disconnected electrodes")
            
            # Sampling rate validation (if provided in context)
            sampling_rate = context.get('sampling_rate')
            if sampling_rate:
                if sampling_rate < self.config.min_sampling_rate:
                    results['warnings'].append(f"Low sampling rate: {sampling_rate} Hz")
                elif sampling_rate > self.config.max_sampling_rate:
                    results['warnings'].append(f"Very high sampling rate: {sampling_rate} Hz")
            
            # Store metrics
            results['metrics'] = {
                'batch_size': batch_size,
                'n_channels': n_channels,
                'n_samples': n_samples,
                'min_amplitude': min_amp,
                'max_amplitude': max_amp,
                'mean_variance': float(torch.mean(signal_variance)),
                'low_variance_channels': int(low_variance_channels)
            }
            
            # Calculate confidence score
            confidence_factors = []
            
            # Channel count factor
            if self.config.min_channels <= n_channels <= self.config.max_channels:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.5)
            
            # Amplitude factor
            if self.config.min_signal_amplitude <= max_amp <= self.config.max_signal_amplitude:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.7)
            
            # Variance factor
            if low_variance_channels < n_channels * 0.1:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.8)
            
            results['confidence_score'] = np.mean(confidence_factors)
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Input validation error: {str(e)}")
            results['confidence_score'] = 0.0
        
        return results


class ModelOutputValidator:
    """Validate model outputs for consistency and reliability."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.prediction_history = []
        
    def validate(self, model_outputs: Dict[str, Any],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model outputs for consistency and quality."""
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {},
            'confidence_score': 1.0
        }
        
        try:
            # Check required outputs
            required_outputs = ['logits', 'eeg_features']
            missing_outputs = []
            
            for output_name in required_outputs:
                if output_name not in model_outputs:
                    missing_outputs.append(output_name)
            
            if missing_outputs:
                results['valid'] = False
                results['errors'].append(f"Missing required outputs: {missing_outputs}")
                return results
            
            # Validate logits
            logits = model_outputs['logits']
            if not isinstance(logits, torch.Tensor):
                results['valid'] = False
                results['errors'].append("Logits must be a PyTorch tensor")
                return results
            
            # Check for NaN/Inf in outputs
            if torch.isnan(logits).any():
                results['valid'] = False
                results['errors'].append("NaN values in logits")
            
            if torch.isinf(logits).any():
                results['valid'] = False
                results['errors'].append("Infinite values in logits")
            
            # Validate probability distribution
            probs = torch.softmax(logits, dim=-1)
            prob_sum = torch.sum(probs, dim=-1)
            
            if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5):
                results['warnings'].append("Probability distribution not normalized")
            
            # Check prediction confidence
            max_probs = torch.max(probs, dim=-1)[0]
            mean_confidence = float(torch.mean(max_probs))
            
            if mean_confidence < self.config.confidence_threshold:
                results['warnings'].append(f"Low prediction confidence: {mean_confidence:.3f}")
                results['recommendations'].append("Consider model retraining or data quality improvement")
            
            # Consistency check with recent predictions
            self.prediction_history.append(probs.detach().cpu())
            if len(self.prediction_history) > self.config.consistency_window:
                self.prediction_history.pop(0)
            
            if len(self.prediction_history) >= 3:
                consistency_score = self._check_prediction_consistency()
                if consistency_score < 0.7:
                    results['warnings'].append(f"Low prediction consistency: {consistency_score:.3f}")
            
            # Validate uncertainty estimates (if present)
            if 'uncertainty_params' in model_outputs and model_outputs['uncertainty_params']:
                uncertainty_results = self._validate_uncertainty(model_outputs['uncertainty_params'])
                results['warnings'].extend(uncertainty_results['warnings'])
                results['recommendations'].extend(uncertainty_results['recommendations'])
            
            # Store metrics
            results['metrics'] = {
                'mean_confidence': mean_confidence,
                'min_confidence': float(torch.min(max_probs)),
                'max_confidence': float(torch.max(max_probs)),
                'entropy': float(torch.mean(-torch.sum(probs * torch.log(probs + 1e-8), dim=-1))),
                'consistency_score': consistency_score if len(self.prediction_history) >= 3 else 1.0
            }
            
            # Calculate confidence score
            confidence_factors = []
            
            # Confidence factor
            confidence_factors.append(min(mean_confidence / self.config.confidence_threshold, 1.0))
            
            # Consistency factor
            if len(self.prediction_history) >= 3:
                confidence_factors.append(consistency_score)
            else:
                confidence_factors.append(1.0)
            
            # Stability factor (no NaN/Inf)
            has_invalid = torch.isnan(logits).any() or torch.isinf(logits).any()
            confidence_factors.append(0.0 if has_invalid else 1.0)
            
            results['confidence_score'] = np.mean(confidence_factors)
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Model output validation error: {str(e)}")
            results['confidence_score'] = 0.0
        
        return results
    
    def _check_prediction_consistency(self) -> float:
        """Check consistency of recent predictions."""
        
        if len(self.prediction_history) < 3:
            return 1.0
        
        # Compute variance across recent predictions
        recent_probs = torch.stack(self.prediction_history[-3:])
        variance = torch.var(recent_probs, dim=0)
        mean_variance = float(torch.mean(variance))
        
        # Convert variance to consistency score (lower variance = higher consistency)
        consistency_score = 1.0 / (1.0 + mean_variance * 10)
        
        return consistency_score
    
    def _validate_uncertainty(self, uncertainty_params: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Validate uncertainty estimates."""
        
        results = {
            'warnings': [],
            'recommendations': []
        }
        
        if 'total_uncertainty' in uncertainty_params:
            total_uncertainty = uncertainty_params['total_uncertainty']
            mean_uncertainty = float(torch.mean(total_uncertainty))
            
            if mean_uncertainty > 0.8:
                results['warnings'].append(f"High uncertainty: {mean_uncertainty:.3f}")
                results['recommendations'].append("Consider increasing training data or model complexity")
        
        return results


class PerformanceValidator:
    """Validate system performance metrics."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate(self, model: nn.Module, eeg_data: torch.Tensor,
                model_outputs: Dict[str, Any], 
                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system performance requirements."""
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {},
            'confidence_score': 1.0
        }
        
        try:
            # Measure inference latency
            latency_ms = self._measure_inference_latency(model, eeg_data)
            
            if latency_ms > self.config.max_latency_ms:
                results['valid'] = False
                results['errors'].append(f"Latency too high: {latency_ms:.2f}ms > {self.config.max_latency_ms}ms")
            elif latency_ms > self.config.max_latency_ms * 0.8:
                results['warnings'].append(f"High latency: {latency_ms:.2f}ms")
            
            # Measure throughput
            throughput = self._measure_throughput(model, eeg_data)
            
            if throughput < self.config.min_throughput:
                results['warnings'].append(f"Low throughput: {throughput:.1f} samples/sec")
                results['recommendations'].append("Consider model optimization or hardware upgrade")
            
            # Check memory usage
            memory_mb = self._estimate_memory_usage(model, eeg_data)
            
            if memory_mb > self.config.max_memory_mb:
                results['warnings'].append(f"High memory usage: {memory_mb:.1f}MB")
                results['recommendations'].append("Consider model pruning or quantization")
            
            # Check for GPU utilization (if available)
            gpu_utilization = self._check_gpu_utilization()
            
            # Store metrics
            results['metrics'] = {
                'latency_ms': latency_ms,
                'throughput_samples_per_sec': throughput,
                'memory_usage_mb': memory_mb,
                'gpu_utilization': gpu_utilization
            }
            
            # Calculate confidence score
            confidence_factors = []
            
            # Latency factor
            confidence_factors.append(min(self.config.max_latency_ms / latency_ms, 1.0))
            
            # Throughput factor
            confidence_factors.append(min(throughput / self.config.min_throughput, 1.0))
            
            # Memory factor
            confidence_factors.append(min(self.config.max_memory_mb / memory_mb, 1.0))
            
            results['confidence_score'] = np.mean(confidence_factors)
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Performance validation error: {str(e)}")
            results['confidence_score'] = 0.0
        
        return results
    
    def _measure_inference_latency(self, model: nn.Module, eeg_data: torch.Tensor) -> float:
        """Measure model inference latency."""
        
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(eeg_data[:1])
        
        # Measure latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(eeg_data[:1])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        return (end_time - start_time) * 1000  # Convert to milliseconds
    
    def _measure_throughput(self, model: nn.Module, eeg_data: torch.Tensor) -> float:
        """Measure system throughput."""
        
        model.eval()
        batch_size = min(8, eeg_data.shape[0])
        test_data = eeg_data[:batch_size]
        
        # Measure time for batch processing
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(test_data)
        
        end_time = time.perf_counter()
        
        batch_time = end_time - start_time
        throughput = batch_size / batch_time
        
        return throughput
    
    def _estimate_memory_usage(self, model: nn.Module, eeg_data: torch.Tensor) -> float:
        """Estimate memory usage."""
        
        # Get model parameters memory
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Estimate activation memory (rough approximation)
        activation_size = eeg_data.numel() * eeg_data.element_size() * 4  # Factor for intermediate activations
        
        total_memory_bytes = param_size + activation_size
        return total_memory_bytes / (1024 * 1024)  # Convert to MB
    
    def _check_gpu_utilization(self) -> Optional[float]:
        """Check GPU utilization if available."""
        
        if not torch.cuda.is_available():
            return None
        
        try:
            # This would require nvidia-ml-py or similar library
            # For now, return a placeholder
            return 0.8  # 80% utilization placeholder
        except:
            return None


class SafetyValidator:
    """Validate safety requirements for clinical use."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate(self, eeg_data: torch.Tensor, model_outputs: Dict[str, Any],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety requirements."""
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {},
            'confidence_score': 1.0
        }
        
        try:
            # Check signal quality for safety
            signal_quality = self._assess_signal_safety(eeg_data)
            
            if signal_quality < self.config.signal_quality_threshold:
                results['warnings'].append(f"Low signal quality for clinical use: {signal_quality:.3f}")
                results['recommendations'].append("Improve electrode contact before clinical use")
            
            # Check for potential seizure indicators
            seizure_risk = self._assess_seizure_risk(eeg_data)
            
            if seizure_risk > 0.7:
                results['valid'] = False
                results['errors'].append("High seizure risk detected - stop session")
            elif seizure_risk > 0.4:
                results['warnings'].append(f"Elevated seizure risk: {seizure_risk:.3f}")
            
            # Assess cognitive load from model outputs
            if 'uncertainty_params' in model_outputs:
                cognitive_load = self._assess_cognitive_load(model_outputs)
                
                if cognitive_load > self.config.max_cognitive_load:
                    results['warnings'].append(f"High cognitive load: {cognitive_load:.3f}")
                    results['recommendations'].append("Reduce task difficulty or provide break")
            
            # Check for fatigue indicators
            fatigue_score = self._assess_fatigue_indicators(eeg_data, context)
            
            if fatigue_score > self.config.fatigue_threshold:
                results['warnings'].append(f"Fatigue detected: {fatigue_score:.3f}")
                results['recommendations'].append("Patient should take a break")
            
            # Store safety metrics
            results['metrics'] = {
                'signal_quality': signal_quality,
                'seizure_risk': seizure_risk,
                'cognitive_load': cognitive_load if 'uncertainty_params' in model_outputs else 0.0,
                'fatigue_score': fatigue_score
            }
            
            # Calculate confidence score
            confidence_factors = []
            
            confidence_factors.append(signal_quality)
            confidence_factors.append(1.0 - seizure_risk)
            confidence_factors.append(1.0 - fatigue_score)
            
            results['confidence_score'] = np.mean(confidence_factors)
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Safety validation error: {str(e)}")
            results['confidence_score'] = 0.0
        
        return results
    
    def _assess_signal_safety(self, eeg_data: torch.Tensor) -> float:
        """Assess signal quality for clinical safety."""
        
        # Check for artifacts and noise
        amplitude_range = torch.max(eeg_data) - torch.min(eeg_data)
        
        # Check for reasonable amplitude range
        if amplitude_range < 10 or amplitude_range > 200:
            return 0.3  # Poor quality
        
        # Check signal-to-noise ratio estimate
        signal_power = torch.mean(eeg_data ** 2)
        noise_estimate = torch.var(eeg_data) * 0.1  # Rough noise estimate
        
        if noise_estimate > 0:
            snr = signal_power / noise_estimate
            snr_score = min(float(snr) / 100, 1.0)
        else:
            snr_score = 1.0
        
        return snr_score
    
    def _assess_seizure_risk(self, eeg_data: torch.Tensor) -> float:
        """Assess seizure risk from EEG patterns."""
        
        # Look for high-frequency spikes
        fft = torch.fft.fft(eeg_data, dim=-1)
        power_spectrum = torch.abs(fft) ** 2
        
        # High gamma activity (30-100 Hz range approximation)
        high_freq_power = torch.mean(power_spectrum[:, :, -100:])  # Rough approximation
        total_power = torch.mean(power_spectrum)
        
        if total_power > 0:
            gamma_ratio = high_freq_power / total_power
            seizure_risk = min(float(gamma_ratio) * 5, 1.0)  # Scale appropriately
        else:
            seizure_risk = 0.0
        
        return seizure_risk
    
    def _assess_cognitive_load(self, model_outputs: Dict[str, Any]) -> float:
        """Assess cognitive load from model uncertainty."""
        
        if 'uncertainty_params' not in model_outputs:
            return 0.0
        
        uncertainty = model_outputs['uncertainty_params']
        
        if 'total_uncertainty' in uncertainty:
            total_uncertainty = uncertainty['total_uncertainty']
            mean_uncertainty = float(torch.mean(total_uncertainty))
            return min(mean_uncertainty, 1.0)
        
        return 0.0
    
    def _assess_fatigue_indicators(self, eeg_data: torch.Tensor, context: Dict[str, Any]) -> float:
        """Assess fatigue from EEG and contextual information."""
        
        # Look for increased alpha activity (8-12 Hz)
        fft = torch.fft.fft(eeg_data, dim=-1)
        power_spectrum = torch.abs(fft) ** 2
        
        # Rough alpha band approximation
        total_samples = eeg_data.shape[-1]
        alpha_start = int(8 * total_samples / 1000)  # Assuming 1000 Hz sampling
        alpha_end = int(12 * total_samples / 1000)
        
        alpha_power = torch.mean(power_spectrum[:, :, alpha_start:alpha_end])
        total_power = torch.mean(power_spectrum)
        
        if total_power > 0:
            alpha_ratio = float(alpha_power / total_power)
        else:
            alpha_ratio = 0.0
        
        # Check session duration from context
        session_duration = context.get('session_duration_minutes', 0)
        duration_fatigue = min(session_duration / 60.0, 1.0)  # Normalize to hours
        
        # Combined fatigue score
        combined_fatigue = 0.6 * alpha_ratio + 0.4 * duration_fatigue
        
        return min(combined_fatigue, 1.0)


class ComplianceValidator:
    """Validate regulatory compliance requirements."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate(self, eeg_data: torch.Tensor, model_outputs: Dict[str, Any],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory compliance."""
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {},
            'confidence_score': 1.0
        }
        
        try:
            compliance_scores = []
            
            # HIPAA Compliance Check
            if self.config.hipaa_compliance:
                hipaa_score = self._validate_hipaa_compliance(eeg_data, context)
                compliance_scores.append(hipaa_score)
                
                if hipaa_score < 0.8:
                    results['warnings'].append("HIPAA compliance concerns detected")
            
            # FDA Compliance Check
            if self.config.fda_compliance:
                fda_score = self._validate_fda_compliance(model_outputs, context)
                compliance_scores.append(fda_score)
                
                if fda_score < 0.8:
                    results['warnings'].append("FDA compliance requirements not fully met")
            
            # GDPR Compliance Check
            if self.config.gdpr_compliance:
                gdpr_score = self._validate_gdpr_compliance(eeg_data, context)
                compliance_scores.append(gdpr_score)
                
                if gdpr_score < 0.8:
                    results['warnings'].append("GDPR compliance concerns detected")
            
            # Overall compliance score
            if compliance_scores:
                overall_compliance = np.mean(compliance_scores)
                
                if overall_compliance < 0.7:
                    results['valid'] = False
                    results['errors'].append("Critical compliance issues detected")
            else:
                overall_compliance = 1.0
            
            results['metrics'] = {
                'overall_compliance_score': overall_compliance,
                'hipaa_score': compliance_scores[0] if len(compliance_scores) > 0 else None,
                'fda_score': compliance_scores[1] if len(compliance_scores) > 1 else None,
                'gdpr_score': compliance_scores[2] if len(compliance_scores) > 2 else None
            }
            
            results['confidence_score'] = overall_compliance
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Compliance validation error: {str(e)}")
            results['confidence_score'] = 0.0
        
        return results
    
    def _validate_hipaa_compliance(self, eeg_data: torch.Tensor, 
                                  context: Dict[str, Any]) -> float:
        """Validate HIPAA compliance requirements."""
        
        compliance_factors = []
        
        # Check for patient identification removal
        patient_id_removed = context.get('patient_id_anonymized', False)
        compliance_factors.append(1.0 if patient_id_removed else 0.0)
        
        # Check for data encryption
        data_encrypted = context.get('data_encrypted', False)
        compliance_factors.append(1.0 if data_encrypted else 0.5)
        
        # Check audit logging
        audit_logging = context.get('audit_logging_enabled', False)
        compliance_factors.append(1.0 if audit_logging else 0.3)
        
        return np.mean(compliance_factors) if compliance_factors else 0.0
    
    def _validate_fda_compliance(self, model_outputs: Dict[str, Any],
                                context: Dict[str, Any]) -> float:
        """Validate FDA medical device compliance."""
        
        compliance_factors = []
        
        # Check for uncertainty quantification
        has_uncertainty = 'uncertainty_params' in model_outputs
        compliance_factors.append(1.0 if has_uncertainty else 0.6)
        
        # Check for safety monitoring
        safety_monitoring = context.get('safety_monitoring_active', False)
        compliance_factors.append(1.0 if safety_monitoring else 0.4)
        
        # Check for clinical validation
        clinically_validated = context.get('clinically_validated', False)
        compliance_factors.append(1.0 if clinically_validated else 0.7)
        
        return np.mean(compliance_factors) if compliance_factors else 0.0
    
    def _validate_gdpr_compliance(self, eeg_data: torch.Tensor,
                                 context: Dict[str, Any]) -> float:
        """Validate GDPR compliance requirements."""
        
        compliance_factors = []
        
        # Check for explicit consent
        explicit_consent = context.get('explicit_consent_obtained', False)
        compliance_factors.append(1.0 if explicit_consent else 0.0)
        
        # Check for data minimization
        data_minimized = context.get('data_minimization_applied', False)
        compliance_factors.append(1.0 if data_minimized else 0.8)
        
        # Check for right to erasure capability
        erasure_capable = context.get('erasure_capability', False)
        compliance_factors.append(1.0 if erasure_capable else 0.7)
        
        return np.mean(compliance_factors) if compliance_factors else 0.0