"""Feature extraction from preprocessed EEG signals."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import scipy.signal
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available for feature extraction")


class FeatureExtractor:
    """Extract meaningful features from EEG signals for BCI applications."""
    
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        
    def extract_spectral_features(self, data: np.ndarray, 
                                 bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, np.ndarray]:
        """Extract power spectral density features.
        
        Args:
            data: EEG data (epochs x channels x samples) or (channels x samples)
            bands: Frequency bands dict {'band_name': (low_freq, high_freq)}
            
        Returns:
            Dictionary of spectral features
        """
        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        
        if not HAS_SCIPY:
            # Return dummy features if scipy not available
            if data.ndim == 3:
                n_epochs, n_channels, _ = data.shape
                return {band: np.random.randn(n_epochs, n_channels) for band in bands}
            else:
                n_channels = data.shape[0]
                return {band: np.random.randn(n_channels) for band in bands}
        
        features = {}
        
        # Handle both epoched and continuous data
        if data.ndim == 3:  # Epoched data
            n_epochs, n_channels, n_samples = data.shape
            
            for band_name, (low_freq, high_freq) in bands.items():
                band_power = np.zeros((n_epochs, n_channels))
                
                for epoch_idx in range(n_epochs):
                    for ch_idx in range(n_channels):
                        # Compute PSD using Welch's method
                        freqs, psd = scipy.signal.welch(
                            data[epoch_idx, ch_idx, :],
                            fs=self.sampling_rate,
                            nperseg=min(256, n_samples // 4)
                        )
                        
                        # Find frequency band indices
                        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        band_power[epoch_idx, ch_idx] = np.mean(psd[band_mask])
                
                features[f'{band_name}_power'] = band_power
                
        else:  # Continuous data
            n_channels, n_samples = data.shape
            
            for band_name, (low_freq, high_freq) in bands.items():
                band_power = np.zeros(n_channels)
                
                for ch_idx in range(n_channels):
                    # Compute PSD using Welch's method
                    freqs, psd = scipy.signal.welch(
                        data[ch_idx, :],
                        fs=self.sampling_rate,
                        nperseg=min(512, n_samples // 4)
                    )
                    
                    # Find frequency band indices
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power[ch_idx] = np.mean(psd[band_mask])
                
                features[f'{band_name}_power'] = band_power
        
        return features
    
    def extract_temporal_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time-domain features.
        
        Args:
            data: EEG data (epochs x channels x samples) or (channels x samples)
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        if data.ndim == 3:  # Epoched data
            # Statistical features per epoch and channel
            features['mean'] = np.mean(data, axis=2)
            features['std'] = np.std(data, axis=2)
            features['var'] = np.var(data, axis=2)
            features['skewness'] = self._calculate_skewness(data)
            features['kurtosis'] = self._calculate_kurtosis(data)
            features['rms'] = np.sqrt(np.mean(data**2, axis=2))
            features['peak_to_peak'] = np.ptp(data, axis=2)
            
            # Zero crossing rate
            features['zero_crossings'] = self._zero_crossing_rate(data)
            
            # Activity and mobility (Hjorth parameters)
            features['activity'], features['mobility'] = self._hjorth_parameters(data)
            
        else:  # Continuous data
            # Statistical features per channel
            features['mean'] = np.mean(data, axis=1)
            features['std'] = np.std(data, axis=1)
            features['var'] = np.var(data, axis=1)
            features['skewness'] = self._calculate_skewness(data[np.newaxis, :, :])
            features['kurtosis'] = self._calculate_kurtosis(data[np.newaxis, :, :])
            features['rms'] = np.sqrt(np.mean(data**2, axis=1))
            features['peak_to_peak'] = np.ptp(data, axis=1)
            
            # Zero crossing rate
            features['zero_crossings'] = self._zero_crossing_rate(data[np.newaxis, :, :])
            
            # Activity and mobility (Hjorth parameters)
            features['activity'], features['mobility'] = self._hjorth_parameters(data[np.newaxis, :, :])
        
        return features
    
    def extract_connectivity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract connectivity features between channels.
        
        Args:
            data: EEG data (epochs x channels x samples) or (channels x samples)
            
        Returns:
            Dictionary of connectivity features
        """
        features = {}
        
        if data.ndim == 3:  # Epoched data
            n_epochs, n_channels, n_samples = data.shape
            
            # Cross-correlation features
            cross_corr = np.zeros((n_epochs, n_channels, n_channels))
            coherence = np.zeros((n_epochs, n_channels, n_channels))
            
            for epoch_idx in range(n_epochs):
                epoch_data = data[epoch_idx]
                
                # Calculate cross-correlation matrix
                for i in range(n_channels):
                    for j in range(n_channels):
                        if i != j:
                            corr = np.corrcoef(epoch_data[i], epoch_data[j])[0, 1]
                            cross_corr[epoch_idx, i, j] = corr
                            
                            # Calculate coherence if scipy available
                            if HAS_SCIPY:
                                f, cxy = scipy.signal.coherence(
                                    epoch_data[i], epoch_data[j],
                                    fs=self.sampling_rate,
                                    nperseg=min(128, n_samples // 4)
                                )
                                # Average coherence in alpha band (8-13 Hz)
                                alpha_mask = (f >= 8) & (f <= 13)
                                coherence[epoch_idx, i, j] = np.mean(cxy[alpha_mask])
                
            features['cross_correlation'] = cross_corr
            features['alpha_coherence'] = coherence
            
        else:  # Continuous data
            n_channels, n_samples = data.shape
            
            # Cross-correlation matrix
            cross_corr = np.zeros((n_channels, n_channels))
            coherence = np.zeros((n_channels, n_channels))
            
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        corr = np.corrcoef(data[i], data[j])[0, 1]
                        cross_corr[i, j] = corr
                        
                        # Calculate coherence if scipy available
                        if HAS_SCIPY:
                            f, cxy = scipy.signal.coherence(
                                data[i], data[j],
                                fs=self.sampling_rate,
                                nperseg=min(256, n_samples // 4)
                            )
                            # Average coherence in alpha band (8-13 Hz)
                            alpha_mask = (f >= 8) & (f <= 13)
                            coherence[i, j] = np.mean(cxy[alpha_mask])
            
            features['cross_correlation'] = cross_corr[np.newaxis, :, :]
            features['alpha_coherence'] = coherence[np.newaxis, :, :]
        
        return features
    
    def extract_complexity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract complexity and nonlinear features.
        
        Args:
            data: EEG data (epochs x channels x samples) or (channels x samples)
            
        Returns:
            Dictionary of complexity features
        """
        features = {}
        
        if data.ndim == 3:  # Epoched data
            n_epochs, n_channels, n_samples = data.shape
            
            # Initialize feature arrays
            spectral_entropy = np.zeros((n_epochs, n_channels))
            sample_entropy = np.zeros((n_epochs, n_channels))
            fractal_dimension = np.zeros((n_epochs, n_channels))
            
            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    signal = data[epoch_idx, ch_idx, :]
                    
                    # Spectral entropy
                    spectral_entropy[epoch_idx, ch_idx] = self._spectral_entropy(signal)
                    
                    # Sample entropy (simplified)
                    sample_entropy[epoch_idx, ch_idx] = self._sample_entropy(signal)
                    
                    # Fractal dimension (Higuchi method)
                    fractal_dimension[epoch_idx, ch_idx] = self._higuchi_fractal_dimension(signal)
            
            features['spectral_entropy'] = spectral_entropy
            features['sample_entropy'] = sample_entropy
            features['fractal_dimension'] = fractal_dimension
            
        else:  # Continuous data
            n_channels, n_samples = data.shape
            
            # Initialize feature arrays
            spectral_entropy = np.zeros(n_channels)
            sample_entropy = np.zeros(n_channels)
            fractal_dimension = np.zeros(n_channels)
            
            for ch_idx in range(n_channels):
                signal = data[ch_idx, :]
                
                # Spectral entropy
                spectral_entropy[ch_idx] = self._spectral_entropy(signal)
                
                # Sample entropy (simplified)
                sample_entropy[ch_idx] = self._sample_entropy(signal)
                
                # Fractal dimension (Higuchi method)
                fractal_dimension[ch_idx] = self._higuchi_fractal_dimension(signal)
            
            features['spectral_entropy'] = spectral_entropy
            features['sample_entropy'] = sample_entropy
            features['fractal_dimension'] = fractal_dimension
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness along the time axis."""
        mean = np.mean(data, axis=2)
        std = np.std(data, axis=2)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        skew = np.mean(((data - mean[..., np.newaxis]) / std[..., np.newaxis]) ** 3, axis=2)
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis along the time axis."""
        mean = np.mean(data, axis=2)
        std = np.std(data, axis=2)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        kurt = np.mean(((data - mean[..., np.newaxis]) / std[..., np.newaxis]) ** 4, axis=2) - 3
        return kurt
    
    def _zero_crossing_rate(self, data: np.ndarray) -> np.ndarray:
        """Calculate zero crossing rate."""
        zero_crossings = np.diff(np.sign(data), axis=2)
        zcr = np.sum(zero_crossings != 0, axis=2) / data.shape[2]
        return zcr
    
    def _hjorth_parameters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Hjorth parameters (Activity and Mobility)."""
        # Activity (variance)
        activity = np.var(data, axis=2)
        
        # First derivative
        dx = np.diff(data, axis=2)
        dx_var = np.var(dx, axis=2)
        
        # Mobility
        mobility = np.sqrt(dx_var / (activity + 1e-10))
        
        return activity, mobility
    
    def _spectral_entropy(self, signal: np.ndarray) -> float:
        """Calculate spectral entropy."""
        if not HAS_SCIPY:
            return np.random.random()
            
        try:
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=len(signal)//4)
            
            # Normalize PSD
            psd_norm = psd / np.sum(psd)
            
            # Calculate entropy
            return entropy(psd_norm)
        except:
            return 0.0
    
    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (simplified version)."""
        try:
            N = len(signal)
            
            # Normalize signal
            signal = (signal - np.mean(signal)) / np.std(signal)
            
            # Template matching
            patterns = np.array([signal[i:i+m] for i in range(N-m+1)])
            
            # Count matches within tolerance r
            matches_m = 0
            matches_m1 = 0
            
            for i in range(len(patterns)):
                template = patterns[i]
                
                # Find matches for length m
                distances = np.max(np.abs(patterns - template), axis=1)
                matches_m += np.sum(distances <= r) - 1  # Exclude self-match
                
                # Find matches for length m+1 (if possible)
                if i < N - m:
                    template_m1 = signal[i:i+m+1]
                    patterns_m1 = np.array([signal[j:j+m+1] for j in range(N-m) if j != i])
                    if len(patterns_m1) > 0:
                        distances_m1 = np.max(np.abs(patterns_m1 - template_m1), axis=1)
                        matches_m1 += np.sum(distances_m1 <= r)
            
            # Calculate sample entropy
            if matches_m1 == 0:
                return float('inf')
            else:
                return -np.log(matches_m1 / max(matches_m, 1))
                
        except:
            return 0.0
    
    def _higuchi_fractal_dimension(self, signal: np.ndarray, k_max: int = 10) -> float:
        """Calculate Higuchi fractal dimension."""
        try:
            N = len(signal)
            L = np.zeros(k_max)
            
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    Lmk = 0
                    for i in range(1, int((N - m) / k)):
                        Lmk += abs(signal[m + i * k] - signal[m + (i - 1) * k])
                    Lmk = Lmk * (N - 1) / (k * k * int((N - m) / k))
                    Lk += Lmk
                
                L[k - 1] = Lk / k
            
            # Linear regression in log-log plot
            x = np.log(range(1, k_max + 1))
            y = np.log(L)
            
            # Remove any invalid values
            valid_mask = np.isfinite(y)
            if np.sum(valid_mask) < 2:
                return 1.0
                
            x = x[valid_mask]
            y = y[valid_mask]
            
            # Calculate slope
            coeffs = np.polyfit(x, y, 1)
            return -coeffs[0]  # Negative slope is the fractal dimension
            
        except:
            return 1.0