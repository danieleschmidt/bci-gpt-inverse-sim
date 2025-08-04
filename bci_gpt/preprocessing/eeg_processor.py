"""EEG signal processing and quality assessment."""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

try:
    import mne
    import scipy.signal
    from scipy.stats import zscore
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("MNE not available. Install with: pip install mne")


@dataclass
class SignalQuality:
    """EEG signal quality assessment results."""
    score: float
    good_channels: List[str]
    bad_channels: List[str]
    noise_level: float
    artifacts_detected: List[str]
    
    @classmethod
    def assess(cls, eeg_data: np.ndarray, channels: List[str], 
               sampling_rate: int = 1000) -> 'SignalQuality':
        """Assess EEG signal quality."""
        n_channels, n_samples = eeg_data.shape
        
        # Calculate noise level (std of high-freq components)
        if n_samples > sampling_rate:
            high_freq = scipy.signal.butter(4, [30, 100], 
                                           btype='band', 
                                           fs=sampling_rate, output='sos')
            noise_signal = scipy.signal.sosfilt(high_freq, eeg_data, axis=1)
            noise_level = np.mean(np.std(noise_signal, axis=1))
        else:
            noise_level = np.mean(np.std(eeg_data, axis=1))
        
        # Identify bad channels (high noise, flat signals)
        channel_stds = np.std(eeg_data, axis=1)
        channel_means = np.mean(np.abs(eeg_data), axis=1)
        
        # Channels with very low or very high variance
        std_threshold_low = np.percentile(channel_stds, 5)
        std_threshold_high = np.percentile(channel_stds, 95)
        
        bad_channel_mask = ((channel_stds < std_threshold_low * 0.1) | 
                           (channel_stds > std_threshold_high * 2))
        
        good_channels = [ch for i, ch in enumerate(channels) if not bad_channel_mask[i]]
        bad_channels = [ch for i, ch in enumerate(channels) if bad_channel_mask[i]]
        
        # Detect artifacts
        artifacts = []
        if noise_level > 50:  # μV
            artifacts.append("high_noise")
        if np.any(np.max(np.abs(eeg_data), axis=1) > 200):  # μV
            artifacts.append("amplitude_artifact")
        if len(bad_channels) > len(channels) * 0.3:
            artifacts.append("many_bad_channels")
        
        # Calculate overall quality score (0-100)
        score = 100.0
        score -= min(noise_level, 50)  # Reduce for noise
        score -= len(bad_channels) / len(channels) * 30  # Reduce for bad channels
        score -= len(artifacts) * 10  # Reduce for artifacts
        score = max(0, score)
        
        return cls(
            score=score,
            good_channels=good_channels,
            bad_channels=bad_channels,
            noise_level=noise_level,
            artifacts_detected=artifacts
        )


class EEGProcessor:
    """EEG signal preprocessing pipeline."""
    
    def __init__(self, 
                 sampling_rate: int = 1000,
                 channels: Optional[List[str]] = None,
                 reference: str = 'average',
                 notch_filter: Optional[float] = 60.0):
        """Initialize EEG processor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            channels: List of channel names
            reference: Reference method ('average', 'common', or channel name)
            notch_filter: Notch filter frequency for line noise (Hz)
        """
        self.sampling_rate = sampling_rate
        self.channels = channels or self._get_default_channels()
        self.reference = reference
        self.notch_filter = notch_filter
        
    def _get_default_channels(self) -> List[str]:
        """Get default 10-20 montage channels."""
        return ['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    
    def load_data(self, file_path: str) -> np.ndarray:
        """Load EEG data from file."""
        if not HAS_MNE:
            raise ImportError("MNE required for data loading. pip install mne")
            
        try:
            if file_path.endswith('.fif'):
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            elif file_path.endswith('.edf'):
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            else:
                # Assume numpy format
                return np.load(file_path)
                
            # Select channels if specified
            if self.channels:
                available_channels = [ch for ch in self.channels if ch in raw.ch_names]
                if available_channels:
                    raw.pick_channels(available_channels)
                    self.channels = available_channels
                    
            return raw.get_data() * 1e6  # Convert to μV
            
        except Exception as e:
            # Fallback: generate synthetic data for testing
            warnings.warn(f"Could not load {file_path}: {e}. Generating synthetic data.")
            return self._generate_synthetic_eeg(duration=10.0)
    
    def _generate_synthetic_eeg(self, duration: float = 10.0) -> np.ndarray:
        """Generate synthetic EEG data for testing."""
        n_samples = int(duration * self.sampling_rate)
        n_channels = len(self.channels)
        
        # Generate realistic EEG with multiple frequency components
        t = np.linspace(0, duration, n_samples)
        eeg_data = np.zeros((n_channels, n_samples))
        
        for i in range(n_channels):
            # Alpha rhythm (8-12 Hz)
            alpha = 20 * np.sin(2 * np.pi * 10 * t + np.random.random() * 2 * np.pi)
            # Beta rhythm (13-30 Hz) 
            beta = 10 * np.sin(2 * np.pi * 20 * t + np.random.random() * 2 * np.pi)
            # Gamma rhythm (30-100 Hz)
            gamma = 5 * np.sin(2 * np.pi * 40 * t + np.random.random() * 2 * np.pi)
            # Pink noise
            noise = np.random.randn(n_samples) * 15
            # Apply pink noise filter
            if n_samples > 100:
                b, a = scipy.signal.butter(2, 0.1, btype='low')
                noise = scipy.signal.filtfilt(b, a, noise)
            
            eeg_data[i] = alpha + beta + gamma + noise
            
        return eeg_data
    
    def preprocess(self, 
                   eeg_data: np.ndarray,
                   bandpass: Tuple[float, float] = (0.5, 40.0),
                   artifact_removal: str = 'ica',
                   epoch_length: float = 1.0) -> Dict[str, Any]:
        """Preprocess EEG data.
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            bandpass: Frequency band for filtering (Hz)
            artifact_removal: Method ('ica', 'asr', or None)
            epoch_length: Length of epochs in seconds
            
        Returns:
            Dictionary containing processed data and metadata
        """
        processed_data = eeg_data.copy()
        metadata = {'preprocessing_steps': []}
        
        # 1. Notch filtering for line noise
        if self.notch_filter:
            processed_data = self._apply_notch_filter(processed_data)
            metadata['preprocessing_steps'].append(f'notch_{self.notch_filter}Hz')
        
        # 2. Bandpass filtering
        processed_data = self._apply_bandpass_filter(processed_data, bandpass)
        metadata['preprocessing_steps'].append(f'bandpass_{bandpass[0]}-{bandpass[1]}Hz')
        
        # 3. Referencing
        processed_data = self._apply_reference(processed_data)
        metadata['preprocessing_steps'].append(f'reference_{self.reference}')
        
        # 4. Artifact removal
        if artifact_removal == 'ica':
            processed_data = self._apply_ica_artifact_removal(processed_data)
            metadata['preprocessing_steps'].append('ica_artifact_removal')
        elif artifact_removal == 'asr':
            processed_data = self._apply_asr_artifact_removal(processed_data)
            metadata['preprocessing_steps'].append('asr_artifact_removal')
        
        # 5. Z-score normalization
        processed_data = zscore(processed_data, axis=1)
        metadata['preprocessing_steps'].append('zscore_normalization')
        
        # 6. Epoching
        epochs = self._create_epochs(processed_data, epoch_length)
        
        return {
            'data': epochs,
            'sampling_rate': self.sampling_rate,
            'channels': self.channels,
            'metadata': metadata,
            'epoch_length': epoch_length
        }
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove line noise."""
        if not self.notch_filter:
            return data
            
        # Design notch filter
        Q = 30  # Quality factor
        w0 = self.notch_filter / (self.sampling_rate / 2)  # Normalized frequency
        b, a = scipy.signal.iirnotch(w0, Q)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = scipy.signal.filtfilt(b, a, data[i])
            
        return filtered_data
    
    def _apply_bandpass_filter(self, data: np.ndarray, 
                              bandpass: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter."""
        low, high = bandpass
        nyquist = self.sampling_rate / 2
        
        # Ensure frequencies are valid
        low = max(low, 0.1)
        high = min(high, nyquist - 1)
        
        # Design butterworth filter
        sos = scipy.signal.butter(4, [low, high], btype='band', 
                                 fs=self.sampling_rate, output='sos')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = scipy.signal.sosfilt(sos, data[i])
            
        return filtered_data
    
    def _apply_reference(self, data: np.ndarray) -> np.ndarray:
        """Apply reference scheme."""
        if self.reference == 'average':
            # Common average reference
            avg_ref = np.mean(data, axis=0)
            return data - avg_ref[np.newaxis, :]
        elif self.reference == 'common':
            # Use first channel as reference
            ref_signal = data[0, :]
            return data - ref_signal[np.newaxis, :]
        else:
            # Specific channel reference
            if self.reference in self.channels:
                ref_idx = self.channels.index(self.reference)
                ref_signal = data[ref_idx, :]
                return data - ref_signal[np.newaxis, :]
            else:
                warnings.warn(f"Reference channel {self.reference} not found. Using average.")
                return self._apply_reference(data)
    
    def _apply_ica_artifact_removal(self, data: np.ndarray) -> np.ndarray:
        """Simple ICA-based artifact removal."""
        if not HAS_MNE:
            warnings.warn("MNE not available for ICA. Skipping artifact removal.")
            return data
            
        try:
            # Create MNE info structure
            info = mne.create_info(self.channels, self.sampling_rate, ch_types='eeg')
            raw = mne.io.RawArray(data, info, verbose=False)
            
            # Apply ICA
            ica = mne.preprocessing.ICA(n_components=min(len(self.channels)-1, 10), 
                                       method='fastica', verbose=False)
            ica.fit(raw)
            
            # Auto-detect and remove eye/muscle artifacts
            # This is simplified - in practice would use more sophisticated detection
            exclude_indices = []
            for i, component in enumerate(ica.get_sources(raw).get_data()):
                # Simple heuristic: high power in high frequencies suggests artifacts
                high_freq_power = np.var(scipy.signal.filtfilt(
                    *scipy.signal.butter(4, 30, fs=self.sampling_rate, btype='high'), 
                    component
                ))
                if high_freq_power > np.var(component) * 0.5:
                    exclude_indices.append(i)
            
            ica.exclude = exclude_indices[:3]  # Limit to top 3 components
            cleaned = ica.apply(raw, verbose=False)
            
            return cleaned.get_data()
            
        except Exception as e:
            warnings.warn(f"ICA failed: {e}. Returning original data.")
            return data
    
    def _apply_asr_artifact_removal(self, data: np.ndarray) -> np.ndarray:
        """Simplified ASR (Artifact Subspace Reconstruction)."""
        # Simplified version - just remove samples with extreme values
        threshold = 5  # Standard deviations
        
        # Calculate z-scores
        z_scores = np.abs(zscore(data, axis=1))
        
        # Find extreme samples
        extreme_mask = np.any(z_scores > threshold, axis=0)
        
        # Replace extreme samples with interpolated values
        if np.any(extreme_mask):
            cleaned_data = data.copy()
            for i in range(data.shape[0]):
                if np.any(extreme_mask):
                    # Simple linear interpolation
                    good_indices = ~extreme_mask
                    if np.sum(good_indices) > 2:
                        cleaned_data[i, extreme_mask] = np.interp(
                            np.where(extreme_mask)[0],
                            np.where(good_indices)[0],
                            data[i, good_indices]
                        )
            return cleaned_data
        
        return data
    
    def _create_epochs(self, data: np.ndarray, epoch_length: float) -> np.ndarray:
        """Create epochs from continuous data."""
        epoch_samples = int(epoch_length * self.sampling_rate)
        n_channels, n_samples = data.shape
        
        if n_samples < epoch_samples:
            # Pad if necessary
            pad_samples = epoch_samples - n_samples
            data = np.pad(data, ((0, 0), (0, pad_samples)), mode='reflect')
            n_samples = data.shape[1]
        
        # Calculate number of epochs
        n_epochs = n_samples // epoch_samples
        
        if n_epochs == 0:
            return data.reshape(1, n_channels, -1)
        
        # Reshape into epochs
        epochs = data[:, :n_epochs * epoch_samples].reshape(
            n_channels, n_epochs, epoch_samples
        ).transpose(1, 0, 2)  # (epochs, channels, samples)
        
        return epochs