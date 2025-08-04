"""Validation and quality assessment for synthetic EEG data."""

import numpy as np
from typing import Dict, Optional, List, Tuple
import warnings
from dataclasses import dataclass

try:
    import scipy.signal
    import scipy.stats
    from scipy.spatial.distance import wasserstein_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available for advanced validation")

try:
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available for advanced validation")


@dataclass
class ValidationMetrics:
    """Container for EEG validation metrics."""
    realism_score: float
    temporal_consistency: float
    spectral_similarity: float
    spatial_coherence: float
    statistical_fidelity: float
    artifact_score: float
    overall_quality: float


class SyntheticEEGValidator:
    """Comprehensive validation of synthetic EEG signals."""
    
    def __init__(self, sampling_rate: int = 1000):
        """Initialize validator.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Reference EEG characteristics
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Expected EEG properties
        self.expected_properties = {
            'amplitude_range': (5, 100),  # μV
            'frequency_peak': (8, 12),    # Alpha peak
            'spatial_correlation': (0.3, 0.8),
            'temporal_autocorr': (0.7, 0.95)
        }
    
    def validate(self,
                synthetic_eeg: np.ndarray,
                real_eeg_stats: Optional[str] = None) -> ValidationMetrics:
        """Comprehensive validation of synthetic EEG.
        
        Args:
            synthetic_eeg: Synthetic EEG signal (channels x samples)
            real_eeg_stats: Path to real EEG statistics file
            
        Returns:
            ValidationMetrics object with all validation scores
        """
        # Load reference statistics if provided
        reference_stats = self._load_reference_stats(real_eeg_stats) if real_eeg_stats else None
        
        # Compute individual validation metrics
        realism_score = self._assess_realism(synthetic_eeg, reference_stats)
        temporal_consistency = self._assess_temporal_consistency(synthetic_eeg)
        spectral_similarity = self._assess_spectral_similarity(synthetic_eeg, reference_stats)
        spatial_coherence = self._assess_spatial_coherence(synthetic_eeg)
        statistical_fidelity = self._assess_statistical_fidelity(synthetic_eeg, reference_stats)
        artifact_score = self._assess_artifacts(synthetic_eeg)
        
        # Compute overall quality score
        overall_quality = self._compute_overall_quality(
            realism_score, temporal_consistency, spectral_similarity,
            spatial_coherence, statistical_fidelity, artifact_score
        )
        
        return ValidationMetrics(
            realism_score=realism_score,
            temporal_consistency=temporal_consistency,
            spectral_similarity=spectral_similarity,
            spatial_coherence=spatial_coherence,
            statistical_fidelity=statistical_fidelity,
            artifact_score=artifact_score,
            overall_quality=overall_quality
        )
    
    def _load_reference_stats(self, stats_path: str) -> Optional[Dict]:
        """Load reference EEG statistics."""
        try:
            if stats_path.endswith('.npy'):
                return np.load(stats_path, allow_pickle=True).item()
            elif stats_path.endswith('.pkl'):
                import pickle
                with open(stats_path, 'rb') as f:
                    return pickle.load(f)
            else:
                warnings.warn(f"Unknown stats file format: {stats_path}")
                return None
        except Exception as e:
            warnings.warn(f"Could not load reference stats: {e}")
            return None
    
    def _assess_realism(self,
                       synthetic_eeg: np.ndarray,
                       reference_stats: Optional[Dict] = None) -> float:
        """Assess overall realism of synthetic EEG."""
        scores = []
        
        # Amplitude realism
        amplitude_score = self._check_amplitude_range(synthetic_eeg)
        scores.append(amplitude_score)
        
        # Frequency content realism
        freq_score = self._check_frequency_content(synthetic_eeg)
        scores.append(freq_score)
        
        # Noise characteristics
        noise_score = self._check_noise_characteristics(synthetic_eeg)
        scores.append(noise_score)
        
        # Compare with reference if available
        if reference_stats:
            ref_score = self._compare_with_reference(synthetic_eeg, reference_stats)
            scores.append(ref_score)
        
        return np.mean(scores)
    
    def _check_amplitude_range(self, eeg_data: np.ndarray) -> float:
        """Check if amplitudes are in realistic range."""
        min_amp, max_amp = self.expected_properties['amplitude_range']
        
        # Calculate RMS amplitude for each channel
        rms_amplitudes = np.sqrt(np.mean(eeg_data ** 2, axis=1))
        
        # Check if amplitudes are in expected range
        in_range = (rms_amplitudes >= min_amp) & (rms_amplitudes <= max_amp)
        score = np.mean(in_range)
        
        # Penalty for extreme values
        extreme_penalty = 0.0
        if np.any(rms_amplitudes < min_amp * 0.1):
            extreme_penalty += 0.3
        if np.any(rms_amplitudes > max_amp * 3):
            extreme_penalty += 0.3
        
        return max(0.0, score - extreme_penalty)
    
    def _check_frequency_content(self, eeg_data: np.ndarray) -> float:
        """Check if frequency content is realistic."""
        if not HAS_SCIPY:
            return 0.5  # Neutral score
        
        scores = []
        
        for ch_idx in range(eeg_data.shape[0]):
            signal = eeg_data[ch_idx]
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=min(1024, len(signal)//4))
            
            # Check for realistic alpha peak
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            if np.any(alpha_mask):
                alpha_power = np.mean(psd[alpha_mask])
                total_power = np.mean(psd)
                alpha_ratio = alpha_power / (total_power + 1e-8)
                
                # Good alpha ratio is typically 0.1-0.4
                alpha_score = 1.0 - abs(alpha_ratio - 0.25) / 0.25
                alpha_score = max(0.0, min(1.0, alpha_score))
                scores.append(alpha_score)
            
            # Check for 1/f characteristic
            log_freqs = np.log10(freqs[1:])  # Skip DC
            log_psd = np.log10(psd[1:])
            
            if len(log_freqs) > 10:
                # Fit linear trend
                slope, _, r_value, _, _ = scipy.stats.linregress(log_freqs, log_psd)
                
                # Good 1/f slope is typically -1 to -2
                slope_score = 1.0 - abs(slope + 1.5) / 1.5
                slope_score = max(0.0, min(1.0, slope_score))
                scores.append(slope_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _check_noise_characteristics(self, eeg_data: np.ndarray) -> float:
        """Check if noise characteristics are realistic."""
        scores = []
        
        for ch_idx in range(eeg_data.shape[0]):
            signal = eeg_data[ch_idx]
            
            # Check for Gaussian-like distribution
            # EEG should have approximately normal distribution with some skewness
            signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Measure kurtosis (should be close to 3 for normal)
            kurtosis = scipy.stats.kurtosis(signal_normalized) if HAS_SCIPY else 0
            kurtosis_score = 1.0 - min(1.0, abs(kurtosis) / 5.0)  # Penalty for extreme kurtosis
            scores.append(kurtosis_score)
            
            # Check for appropriate noise level
            # High-frequency noise should be present but not dominant
            if HAS_SCIPY:
                freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=min(512, len(signal)//4))
                
                # High frequency power (>50 Hz)
                hf_mask = freqs > 50
                if np.any(hf_mask):
                    hf_power = np.mean(psd[hf_mask])
                    total_power = np.mean(psd)
                    hf_ratio = hf_power / (total_power + 1e-8)
                    
                    # Good HF ratio is typically 0.05-0.2
                    noise_score = 1.0 - abs(hf_ratio - 0.125) / 0.125
                    noise_score = max(0.0, min(1.0, noise_score))
                    scores.append(noise_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_temporal_consistency(self, eeg_data: np.ndarray) -> float:
        """Assess temporal consistency and smoothness."""
        scores = []
        
        for ch_idx in range(eeg_data.shape[0]):
            signal = eeg_data[ch_idx]
            
            # Autocorrelation assessment
            if len(signal) > 100:
                # Calculate autocorrelation at lag 1
                autocorr_lag1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                
                # Good autocorrelation should be high but not perfect
                expected_min, expected_max = self.expected_properties['temporal_autocorr']
                if expected_min <= autocorr_lag1 <= expected_max:
                    autocorr_score = 1.0
                else:
                    autocorr_score = max(0.0, 1.0 - abs(autocorr_lag1 - 0.85) / 0.85)
                
                scores.append(autocorr_score)
            
            # Smoothness assessment
            # EEG should be relatively smooth (not too spiky)
            diff_signal = np.diff(signal)
            smoothness = 1.0 / (1.0 + np.std(diff_signal) / (np.std(signal) + 1e-8))
            smoothness_score = min(1.0, smoothness)
            scores.append(smoothness_score)
            
            # Check for unrealistic jumps
            large_jumps = np.sum(np.abs(diff_signal) > 5 * np.std(signal))
            jump_ratio = large_jumps / len(diff_signal)
            jump_score = max(0.0, 1.0 - jump_ratio * 10)  # Penalty for too many jumps
            scores.append(jump_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_spectral_similarity(self,
                                  synthetic_eeg: np.ndarray,
                                  reference_stats: Optional[Dict] = None) -> float:
        """Assess spectral similarity to real EEG."""
        if not HAS_SCIPY:
            return 0.5
        
        scores = []
        
        # Expected EEG band power ratios
        expected_ratios = {
            'delta': 0.15,
            'theta': 0.20,
            'alpha': 0.30,
            'beta': 0.25,
            'gamma': 0.10
        }
        
        for ch_idx in range(synthetic_eeg.shape[0]):
            signal = synthetic_eeg[ch_idx]
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=min(1024, len(signal)//4))
            
            # Calculate power in each band
            band_powers = {}
            total_power = np.sum(psd)
            
            for band_name, (low, high) in self.eeg_bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_power = np.sum(psd[band_mask])
                    band_powers[band_name] = band_power / (total_power + 1e-8)
            
            # Compare with expected ratios
            band_scores = []
            for band_name, expected_ratio in expected_ratios.items():
                if band_name in band_powers:
                    actual_ratio = band_powers[band_name]
                    ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
                    band_score = max(0.0, 1.0 - ratio_error)
                    band_scores.append(band_score)
            
            if band_scores:
                scores.append(np.mean(band_scores))
        
        # Compare with reference statistics if available
        if reference_stats and 'spectral_features' in reference_stats:
            ref_similarity = self._compare_spectral_features(synthetic_eeg, reference_stats['spectral_features'])
            scores.append(ref_similarity)
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_spatial_coherence(self, eeg_data: np.ndarray) -> float:
        """Assess spatial coherence between channels."""
        if eeg_data.shape[0] < 2:
            return 1.0  # Single channel case
        
        # Calculate correlation matrix between channels
        corr_matrix = np.corrcoef(eeg_data)
        
        # Remove diagonal (self-correlations)
        off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Good EEG should have moderate correlations between channels
        expected_min, expected_max = self.expected_properties['spatial_correlation']
        
        # Check proportion of correlations in expected range
        in_range = (off_diagonal >= expected_min) & (off_diagonal <= expected_max)
        coherence_score = np.mean(in_range)
        
        # Penalty for extreme correlations
        extreme_penalty = 0.0
        if np.any(off_diagonal > 0.95):  # Too highly correlated
            extreme_penalty += 0.2
        if np.any(off_diagonal < 0.0):   # Negative correlations (unusual)
            extreme_penalty += 0.1
        
        return max(0.0, coherence_score - extreme_penalty)
    
    def _assess_statistical_fidelity(self,
                                   synthetic_eeg: np.ndarray,
                                   reference_stats: Optional[Dict] = None) -> float:
        """Assess statistical fidelity of synthetic EEG."""
        scores = []
        
        # Basic statistical properties
        for ch_idx in range(synthetic_eeg.shape[0]):
            signal = synthetic_eeg[ch_idx]
            
            # Mean should be close to zero
            mean_score = 1.0 - min(1.0, abs(np.mean(signal)) / (np.std(signal) + 1e-8))
            scores.append(mean_score)
            
            # Variance should be reasonable
            variance = np.var(signal)
            if 10 <= variance <= 1000:  # Reasonable range for EEG
                var_score = 1.0
            else:
                var_score = max(0.0, 1.0 - abs(np.log10(variance) - 2) / 2)
            scores.append(var_score)
        
        # Compare with reference if available
        if reference_stats:
            if 'mean' in reference_stats and 'std' in reference_stats:
                ref_means = reference_stats['mean']
                ref_stds = reference_stats['std']
                
                synthetic_means = np.mean(synthetic_eeg, axis=1)
                synthetic_stds = np.std(synthetic_eeg, axis=1)
                
                if len(ref_means) == len(synthetic_means):
                    mean_similarity = 1.0 - np.mean(np.abs(synthetic_means - ref_means) / (ref_stds + 1e-8))
                    std_similarity = 1.0 - np.mean(np.abs(synthetic_stds - ref_stds) / (ref_stds + 1e-8))
                    
                    scores.extend([max(0.0, mean_similarity), max(0.0, std_similarity)])
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_artifacts(self, eeg_data: np.ndarray) -> float:
        """Assess presence of realistic vs unrealistic artifacts."""
        scores = []
        
        for ch_idx in range(eeg_data.shape[0]):
            signal = eeg_data[ch_idx]
            
            # Check for saturation (constant values)
            unique_values = len(np.unique(signal))
            if unique_values < len(signal) * 0.1:  # Too few unique values
                saturation_score = 0.0
            else:
                saturation_score = 1.0
            scores.append(saturation_score)
            
            # Check for extreme outliers
            z_scores = np.abs((signal - np.mean(signal)) / (np.std(signal) + 1e-8))
            extreme_outliers = np.sum(z_scores > 6)  # More than 6 standard deviations
            outlier_ratio = extreme_outliers / len(signal)
            outlier_score = max(0.0, 1.0 - outlier_ratio * 100)
            scores.append(outlier_score)
            
            # Check for NaN or infinite values
            invalid_values = np.sum(~np.isfinite(signal))
            if invalid_values == 0:
                validity_score = 1.0
            else:
                validity_score = 0.0  # Any invalid values are unacceptable
            scores.append(validity_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _compare_with_reference(self,
                              synthetic_eeg: np.ndarray,
                              reference_stats: Dict) -> float:
        """Compare synthetic EEG with reference statistics."""
        if not HAS_SCIPY:
            return 0.5
        
        scores = []
        
        # Compare distributions using Wasserstein distance
        if 'samples' in reference_stats:
            ref_samples = reference_stats['samples']
            
            for ch_idx in range(min(synthetic_eeg.shape[0], ref_samples.shape[0])):
                synthetic_ch = synthetic_eeg[ch_idx]
                reference_ch = ref_samples[ch_idx]
                
                # Normalize both signals
                synthetic_norm = (synthetic_ch - np.mean(synthetic_ch)) / (np.std(synthetic_ch) + 1e-8)
                reference_norm = (reference_ch - np.mean(reference_ch)) / (np.std(reference_ch) + 1e-8)
                
                # Calculate Wasserstein distance
                try:
                    distance = wasserstein_distance(synthetic_norm, reference_norm)
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    scores.append(similarity)
                except:
                    scores.append(0.5)  # Neutral score if calculation fails
        
        return np.mean(scores) if scores else 0.5
    
    def _compare_spectral_features(self,
                                 synthetic_eeg: np.ndarray,
                                 reference_features: Dict) -> float:
        """Compare spectral features with reference."""
        if not HAS_SCIPY:
            return 0.5
        
        scores = []
        
        for ch_idx in range(synthetic_eeg.shape[0]):
            signal = synthetic_eeg[ch_idx]
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=min(1024, len(signal)//4))
            
            # Extract features
            synthetic_features = self._extract_spectral_features(freqs, psd)
            
            # Compare with reference
            if f'channel_{ch_idx}' in reference_features:
                ref_features = reference_features[f'channel_{ch_idx}']
                
                feature_scores = []
                for feature_name, synthetic_value in synthetic_features.items():
                    if feature_name in ref_features:
                        ref_value = ref_features[feature_name]
                        relative_error = abs(synthetic_value - ref_value) / (abs(ref_value) + 1e-8)
                        feature_score = max(0.0, 1.0 - relative_error)
                        feature_scores.append(feature_score)
                
                if feature_scores:
                    scores.append(np.mean(feature_scores))
        
        return np.mean(scores) if scores else 0.5
    
    def _extract_spectral_features(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Extract spectral features from PSD."""
        features = {}
        
        # Peak frequency
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_idx]
        
        # Spectral centroid (weighted mean frequency)
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral bandwidth
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))
        
        # Band power ratios
        total_power = np.sum(psd)
        for band_name, (low, high) in self.eeg_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_power = np.sum(psd[band_mask])
                features[f'{band_name}_ratio'] = band_power / total_power
        
        return features
    
    def _compute_overall_quality(self,
                               realism_score: float,
                               temporal_consistency: float,
                               spectral_similarity: float,
                               spatial_coherence: float,
                               statistical_fidelity: float,
                               artifact_score: float) -> float:
        """Compute overall quality score."""
        # Weighted combination of all scores
        weights = {
            'realism': 0.25,
            'temporal': 0.20,
            'spectral': 0.20,
            'spatial': 0.15,
            'statistical': 0.15,
            'artifacts': 0.05
        }
        
        overall_score = (
            weights['realism'] * realism_score +
            weights['temporal'] * temporal_consistency +
            weights['spectral'] * spectral_similarity +
            weights['spatial'] * spatial_coherence +
            weights['statistical'] * statistical_fidelity +
            weights['artifacts'] * artifact_score
        )
        
        return np.clip(overall_score, 0.0, 1.0)
    
    def generate_report(self, metrics: ValidationMetrics) -> str:
        """Generate a human-readable validation report.
        
        Args:
            metrics: ValidationMetrics object
            
        Returns:
            Formatted validation report string
        """
        report = "=== Synthetic EEG Validation Report ===\n\n"
        
        report += f"Overall Quality Score: {metrics.overall_quality:.3f}\n\n"
        
        report += "Individual Metrics:\n"
        report += f"  Realism Score:        {metrics.realism_score:.3f}\n"
        report += f"  Temporal Consistency: {metrics.temporal_consistency:.3f}\n"
        report += f"  Spectral Similarity:  {metrics.spectral_similarity:.3f}\n"
        report += f"  Spatial Coherence:    {metrics.spatial_coherence:.3f}\n"
        report += f"  Statistical Fidelity: {metrics.statistical_fidelity:.3f}\n"
        report += f"  Artifact Score:       {metrics.artifact_score:.3f}\n\n"
        
        # Quality assessment
        if metrics.overall_quality >= 0.8:
            quality_label = "Excellent"
        elif metrics.overall_quality >= 0.6:
            quality_label = "Good"
        elif metrics.overall_quality >= 0.4:
            quality_label = "Fair"
        else:
            quality_label = "Poor"
        
        report += f"Quality Assessment: {quality_label}\n\n"
        
        # Recommendations
        report += "Recommendations:\n"
        if metrics.realism_score < 0.6:
            report += "  - Improve amplitude and frequency characteristics\n"
        if metrics.temporal_consistency < 0.6:
            report += "  - Enhance temporal smoothness and continuity\n"
        if metrics.spectral_similarity < 0.6:
            report += "  - Adjust frequency band power distributions\n"
        if metrics.spatial_coherence < 0.6:
            report += "  - Improve inter-channel correlations\n"
        if metrics.statistical_fidelity < 0.6:
            report += "  - Adjust statistical properties (mean, variance)\n"
        if metrics.artifact_score < 0.8:
            report += "  - Address unrealistic artifacts or outliers\n"
        
        if metrics.overall_quality >= 0.8:
            report += "  - Synthetic EEG quality is excellent!\n"
        
        return report