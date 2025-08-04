"""Advanced artifact removal techniques for EEG signals."""

import numpy as np
from typing import Tuple, List, Optional
import warnings

try:
    import scipy.signal
    from scipy.linalg import eigh
    from sklearn.decomposition import FastICA
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available for advanced artifact removal")


class ArtifactRemover:
    """Advanced EEG artifact removal using multiple techniques."""
    
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        
    def remove_eye_artifacts(self, data: np.ndarray, 
                           method: str = 'ica') -> Tuple[np.ndarray, dict]:
        """Remove eye movement and blink artifacts.
        
        Args:
            data: EEG data (channels x samples)
            method: Removal method ('ica', 'regression', 'template')
            
        Returns:
            Cleaned data and removal metadata
        """
        if method == 'ica':
            return self._ica_eye_removal(data)
        elif method == 'regression':
            return self._regression_eye_removal(data)
        elif method == 'template':
            return self._template_eye_removal(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def remove_muscle_artifacts(self, data: np.ndarray,
                               threshold: float = 3.0) -> Tuple[np.ndarray, dict]:
        """Remove muscle artifacts using spectral analysis.
        
        Args:
            data: EEG data (channels x samples)
            threshold: Z-score threshold for artifact detection
            
        Returns:
            Cleaned data and removal metadata
        """
        if not HAS_SCIPY:
            return data, {'method': 'none', 'reason': 'scipy_unavailable'}
            
        # Calculate high-frequency power (30-100 Hz)
        high_freq_sos = scipy.signal.butter(4, [30, 100], 
                                           btype='band', 
                                           fs=self.sampling_rate, 
                                           output='sos')
        
        cleaned_data = data.copy()
        artifacts_removed = 0
        
        for ch_idx in range(data.shape[0]):
            # Extract high-frequency component
            hf_signal = scipy.signal.sosfilt(high_freq_sos, data[ch_idx])
            hf_power = np.abs(hf_signal) ** 2
            
            # Smooth power signal
            window_size = int(0.1 * self.sampling_rate)  # 100ms windows
            if window_size > 1:
                hf_power_smooth = scipy.signal.savgol_filter(hf_power, 
                                                            window_size, 3)
            else:
                hf_power_smooth = hf_power
            
            # Detect artifacts using z-score
            z_scores = (hf_power_smooth - np.mean(hf_power_smooth)) / np.std(hf_power_smooth)
            artifact_mask = np.abs(z_scores) > threshold
            
            if np.any(artifact_mask):
                # Replace artifacts with interpolated values
                good_samples = ~artifact_mask
                if np.sum(good_samples) > 10:
                    cleaned_data[ch_idx, artifact_mask] = np.interp(
                        np.where(artifact_mask)[0],
                        np.where(good_samples)[0],
                        data[ch_idx, good_samples]
                    )
                    artifacts_removed += np.sum(artifact_mask)
        
        metadata = {
            'method': 'spectral_muscle_removal',
            'threshold': threshold,
            'samples_removed': artifacts_removed,
            'percentage_removed': artifacts_removed / data.size * 100
        }
        
        return cleaned_data, metadata
    
    def remove_cardiac_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Remove cardiac artifacts using template subtraction.
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            Cleaned data and removal metadata
        """
        if not HAS_SCIPY:
            return data, {'method': 'none', 'reason': 'scipy_unavailable'}
            
        # Detect R-peaks in EEG (simplified approach)
        # In practice, would use ECG channel if available
        
        cleaned_data = data.copy()
        r_peaks_detected = 0
        
        for ch_idx in range(data.shape[0]):
            signal = data[ch_idx]
            
            # Bandpass filter for cardiac frequency (1-3 Hz)
            cardiac_sos = scipy.signal.butter(4, [1, 3], 
                                            btype='band',
                                            fs=self.sampling_rate,
                                            output='sos')
            cardiac_signal = scipy.signal.sosfilt(cardiac_sos, signal)
            
            # Find peaks (simplified R-peak detection)
            peaks, _ = scipy.signal.find_peaks(
                np.abs(cardiac_signal),
                height=np.std(cardiac_signal) * 2,
                distance=int(0.6 * self.sampling_rate)  # Min 600ms between beats
            )
            
            if len(peaks) > 3:
                # Create cardiac template
                template_window = int(0.4 * self.sampling_rate)  # 400ms window
                templates = []
                
                for peak in peaks:
                    start = max(0, peak - template_window // 2)
                    end = min(len(signal), peak + template_window // 2)
                    if end - start == template_window:
                        templates.append(signal[start:end])
                
                if len(templates) > 3:
                    # Average template
                    avg_template = np.mean(templates, axis=0)
                    
                    # Subtract template at each R-peak location
                    for peak in peaks:
                        start = max(0, peak - template_window // 2)
                        end = min(len(signal), peak + template_window // 2)
                        if end - start == template_window:
                            # Scale template to match amplitude
                            correlation = np.corrcoef(signal[start:end], avg_template)[0, 1]
                            if correlation > 0.5:  # Only if well correlated
                                scale = np.std(signal[start:end]) / np.std(avg_template)
                                cleaned_data[ch_idx, start:end] -= avg_template * scale * 0.8
                    
                    r_peaks_detected += len(peaks)
        
        metadata = {
            'method': 'cardiac_template_subtraction',
            'r_peaks_detected': r_peaks_detected,
            'channels_processed': data.shape[0]
        }
        
        return cleaned_data, metadata
    
    def _ica_eye_removal(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """ICA-based eye artifact removal."""
        if not HAS_SCIPY:
            return data, {'method': 'none', 'reason': 'scipy_unavailable'}
            
        try:
            # Apply FastICA
            n_components = min(data.shape[0] - 1, 15)
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            
            # Fit ICA on transposed data (samples x channels)
            sources = ica.fit_transform(data.T).T  # Back to (components x samples)
            mixing_matrix = ica.mixing_.T  # (channels x components)
            
            # Identify eye artifact components
            eye_components = []
            for i in range(sources.shape[0]):
                source = sources[i]
                
                # Check for eye artifact characteristics:
                # 1. High power in frontal channels
                # 2. Low frequency content
                # 3. Sharp transients (blinks)
                
                # Low frequency power (0.5-4 Hz)
                lf_sos = scipy.signal.butter(4, [0.5, 4], 
                                           btype='band',
                                           fs=self.sampling_rate,
                                           output='sos')
                lf_power = np.var(scipy.signal.sosfilt(lf_sos, source))
                total_power = np.var(source)
                
                # Sharp transient detection
                diff_signal = np.diff(source)
                sharp_transients = np.sum(np.abs(diff_signal) > 3 * np.std(diff_signal))
                
                # Frontal loading (assume first few channels are frontal)
                frontal_loading = np.sum(np.abs(mixing_matrix[:min(3, len(mixing_matrix)), i]))
                total_loading = np.sum(np.abs(mixing_matrix[:, i]))
                
                # Decision criteria
                if (lf_power / total_power > 0.3 and  # High low-freq power
                    sharp_transients > len(source) * 0.001 and  # Sharp transients
                    frontal_loading / total_loading > 0.4):  # Frontal loading
                    eye_components.append(i)
            
            # Limit to top 3 components to avoid over-removal
            eye_components = eye_components[:3]
            
            # Reconstruct data without eye components
            if eye_components:
                sources_clean = sources.copy()
                sources_clean[eye_components] = 0
                
                # Reconstruct signals
                cleaned_data = (mixing_matrix @ sources_clean)
            else:
                cleaned_data = data
                
            metadata = {
                'method': 'ica_eye_removal',
                'components_removed': eye_components,
                'total_components': sources.shape[0]
            }
            
            return cleaned_data, metadata
            
        except Exception as e:
            warnings.warn(f"ICA eye removal failed: {e}")
            return data, {'method': 'failed', 'error': str(e)}
    
    def _regression_eye_removal(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Regression-based eye artifact removal."""
        # Simplified version - assume first channel contains eye artifacts
        if data.shape[0] < 2:
            return data, {'method': 'insufficient_channels'}
            
        eog_channel = data[0]  # Assume first channel is most affected
        cleaned_data = data.copy()
        
        # Remove EOG from other channels using regression
        for ch_idx in range(1, data.shape[0]):
            eeg_channel = data[ch_idx]
            
            # Calculate regression coefficient
            coef = np.corrcoef(eeg_channel, eog_channel)[0, 1]
            
            # Apply correction if correlation is significant
            if abs(coef) > 0.3:
                # Scale EOG and subtract
                scale = np.std(eeg_channel) / np.std(eog_channel) * coef
                cleaned_data[ch_idx] = eeg_channel - scale * eog_channel
        
        metadata = {
            'method': 'regression_eye_removal',
            'eog_channel': 0,
            'channels_corrected': data.shape[0] - 1
        }
        
        return cleaned_data, metadata
    
    def _template_eye_removal(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Template-based blink artifact removal."""
        if not HAS_SCIPY:
            return data, {'method': 'none', 'reason': 'scipy_unavailable'}
            
        cleaned_data = data.copy()
        blinks_removed = 0
        
        for ch_idx in range(data.shape[0]):
            signal = data[ch_idx]
            
            # Detect blinks (sharp negative deflections in frontal channels)
            # Apply high-pass filter to enhance blinks
            hp_sos = scipy.signal.butter(4, 1, btype='high', 
                                       fs=self.sampling_rate, output='sos')
            filtered_signal = scipy.signal.sosfilt(hp_sos, signal)
            
            # Find negative peaks (blinks)
            peaks, properties = scipy.signal.find_peaks(
                -filtered_signal,
                height=2 * np.std(filtered_signal),
                width=(int(0.05 * self.sampling_rate), int(0.5 * self.sampling_rate)),
                distance=int(0.5 * self.sampling_rate)
            )
            
            if len(peaks) > 2:
                # Create blink template
                template_window = int(0.6 * self.sampling_rate)  # 600ms
                templates = []
                
                for peak in peaks:
                    start = max(0, peak - template_window // 3)
                    end = min(len(signal), peak + template_window * 2 // 3)
                    if end - start >= template_window // 2:
                        blink_segment = signal[start:end]
                        # Normalize template
                        blink_segment = (blink_segment - np.mean(blink_segment))
                        templates.append(blink_segment)
                
                if len(templates) > 2:
                    # Find common template length
                    min_len = min(len(t) for t in templates)
                    templates = [t[:min_len] for t in templates]
                    avg_template = np.mean(templates, axis=0)
                    
                    # Remove blinks using template subtraction
                    for peak in peaks:
                        start = max(0, peak - len(avg_template) // 3)
                        end = min(len(signal), start + len(avg_template))
                        
                        if end - start == len(avg_template):
                            segment = signal[start:end]
                            # Scale template to match segment
                            correlation = np.corrcoef(segment, avg_template)[0, 1]
                            if correlation < -0.5:  # Negative correlation for blinks
                                scale = np.std(segment) / np.std(avg_template)
                                cleaned_data[ch_idx, start:end] = (
                                    segment + avg_template * scale * abs(correlation)
                                )
                                blinks_removed += 1
        
        metadata = {
            'method': 'template_blink_removal',
            'blinks_removed': blinks_removed,
            'channels_processed': data.shape[0]
        }
        
        return cleaned_data, metadata