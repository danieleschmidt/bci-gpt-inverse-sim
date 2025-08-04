"""Data augmentation techniques for EEG signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
import warnings

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available for advanced augmentation")


class EEGAugmenter:
    """Data augmentation for EEG signals with neuroscience-informed techniques."""
    
    def __init__(self,
                 noise_std: float = 0.1,
                 time_shift_range: int = 50,
                 amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
                 frequency_shift_range: float = 2.0,
                 channel_dropout_prob: float = 0.1,
                 mixup_alpha: float = 0.2):
        """Initialize EEG augmenter.
        
        Args:
            noise_std: Standard deviation for additive noise
            time_shift_range: Maximum time shift in samples
            amplitude_scale_range: Range for amplitude scaling
            frequency_shift_range: Maximum frequency shift in Hz
            channel_dropout_prob: Probability of dropping each channel
            mixup_alpha: Alpha parameter for mixup augmentation
        """
        self.noise_std = noise_std
        self.time_shift_range = time_shift_range
        self.amplitude_scale_range = amplitude_scale_range
        self.frequency_shift_range = frequency_shift_range
        self.channel_dropout_prob = channel_dropout_prob
        self.mixup_alpha = mixup_alpha
        
        # Available augmentation techniques
        self.augmentations = [
            self._add_noise,
            self._time_shift,
            self._amplitude_scale,
            self._channel_dropout,
            self._frequency_masking,
            self._time_masking,
        ]
        
        if HAS_SCIPY:
            self.augmentations.extend([
                self._frequency_shift,
                self._bandpass_filter,
                self._phase_shift,
            ])
    
    def augment(self, 
                eeg_data: torch.Tensor,
                augment_prob: float = 0.5,
                num_augmentations: int = 2) -> torch.Tensor:
        """Apply random augmentations to EEG data.
        
        Args:
            eeg_data: EEG tensor (batch_size, channels, samples)
            augment_prob: Probability of applying each augmentation
            num_augmentations: Maximum number of augmentations to apply
            
        Returns:
            Augmented EEG data
        """
        augmented = eeg_data.clone()
        
        # Randomly select augmentations
        selected_augmentations = np.random.choice(
            len(self.augmentations),
            size=min(num_augmentations, len(self.augmentations)),
            replace=False
        )
        
        for aug_idx in selected_augmentations:
            if torch.rand(1) < augment_prob:
                augmentation_fn = self.augmentations[aug_idx]
                try:
                    augmented = augmentation_fn(augmented)
                except Exception as e:
                    warnings.warn(f"Augmentation {augmentation_fn.__name__} failed: {e}")
                    continue
        
        return augmented
    
    def _add_noise(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to EEG signals."""
        noise = torch.randn_like(eeg_data) * self.noise_std
        return eeg_data + noise
    
    def _time_shift(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply random time shifts to EEG signals."""
        batch_size, n_channels, n_samples = eeg_data.shape
        shifted_data = torch.zeros_like(eeg_data)
        
        for b in range(batch_size):
            # Random shift for each sample in batch
            shift = torch.randint(-self.time_shift_range, self.time_shift_range + 1, (1,)).item()
            
            if shift > 0:
                # Shift right (pad left)
                shifted_data[b, :, shift:] = eeg_data[b, :, :-shift]
                shifted_data[b, :, :shift] = eeg_data[b, :, :shift]  # Reflection padding
            elif shift < 0:
                # Shift left (pad right)
                shifted_data[b, :, :shift] = eeg_data[b, :, -shift:]
                shifted_data[b, :, shift:] = eeg_data[b, :, shift:]  # Reflection padding
            else:
                shifted_data[b] = eeg_data[b]
        
        return shifted_data
    
    def _amplitude_scale(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling to EEG signals."""
        batch_size, n_channels, n_samples = eeg_data.shape
        
        min_scale, max_scale = self.amplitude_scale_range
        scales = torch.empty(batch_size, n_channels, 1).uniform_(min_scale, max_scale)
        scales = scales.to(eeg_data.device)
        
        return eeg_data * scales
    
    def _channel_dropout(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Randomly drop (zero out) some EEG channels."""
        batch_size, n_channels, n_samples = eeg_data.shape
        
        # Create dropout mask
        dropout_mask = torch.rand(batch_size, n_channels, 1) > self.channel_dropout_prob
        dropout_mask = dropout_mask.to(eeg_data.device)
        
        return eeg_data * dropout_mask
    
    def _frequency_masking(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking in the spectral domain."""
        # FFT
        fft_data = torch.fft.rfft(eeg_data, dim=2)
        
        # Random frequency mask
        freq_bins = fft_data.shape[2]
        mask_width = torch.randint(1, freq_bins // 4, (1,)).item()
        mask_start = torch.randint(0, freq_bins - mask_width, (1,)).item()
        
        # Apply mask
        masked_fft = fft_data.clone()
        masked_fft[:, :, mask_start:mask_start + mask_width] = 0
        
        # Inverse FFT
        return torch.fft.irfft(masked_fft, n=eeg_data.shape[2], dim=2)
    
    def _time_masking(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply time masking to EEG signals."""
        batch_size, n_channels, n_samples = eeg_data.shape
        
        # Random time mask
        mask_width = torch.randint(1, n_samples // 10, (1,)).item()
        mask_start = torch.randint(0, n_samples - mask_width, (1,)).item()
        
        masked_data = eeg_data.clone()
        masked_data[:, :, mask_start:mask_start + mask_width] = 0
        
        return masked_data
    
    def _frequency_shift(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply frequency domain shifts using Hilbert transform."""
        if not HAS_SCIPY:
            return eeg_data
        
        shifted_data = torch.zeros_like(eeg_data)
        
        for b in range(eeg_data.shape[0]):
            for ch in range(eeg_data.shape[1]):
                signal = eeg_data[b, ch].cpu().numpy()
                
                # Random frequency shift
                shift_hz = np.random.uniform(-self.frequency_shift_range, self.frequency_shift_range)
                
                # Apply frequency shift using complex modulation
                t = np.linspace(0, len(signal)/1000, len(signal))  # Assume 1000 Hz sampling
                complex_signal = scipy.signal.hilbert(signal)
                shifted_signal = complex_signal * np.exp(2j * np.pi * shift_hz * t)
                
                shifted_data[b, ch] = torch.from_numpy(shifted_signal.real).float()
        
        return shifted_data.to(eeg_data.device)
    
    def _bandpass_filter(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply random bandpass filtering."""
        if not HAS_SCIPY:
            return eeg_data
        
        # Random frequency band
        low_freq = np.random.uniform(0.5, 8.0)
        high_freq = np.random.uniform(low_freq + 5, 50.0)
        
        filtered_data = torch.zeros_like(eeg_data)
        
        for b in range(eeg_data.shape[0]):
            for ch in range(eeg_data.shape[1]):
                signal = eeg_data[b, ch].cpu().numpy()
                
                # Design and apply filter
                try:
                    sos = scipy.signal.butter(4, [low_freq, high_freq], 
                                            btype='band', fs=1000, output='sos')
                    filtered_signal = scipy.signal.sosfilt(sos, signal)
                    filtered_data[b, ch] = torch.from_numpy(filtered_signal).float()
                except:
                    filtered_data[b, ch] = eeg_data[b, ch]
        
        return filtered_data.to(eeg_data.device)
    
    def _phase_shift(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply random phase shifts to EEG signals."""
        if not HAS_SCIPY:
            return eeg_data
        
        phase_shifted_data = torch.zeros_like(eeg_data)
        
        for b in range(eeg_data.shape[0]):
            for ch in range(eeg_data.shape[1]):
                signal = eeg_data[b, ch].cpu().numpy()
                
                # Random phase shift
                phase_shift = np.random.uniform(0, 2 * np.pi)
                
                # Apply phase shift using Hilbert transform
                analytic_signal = scipy.signal.hilbert(signal)
                phase_shifted_signal = np.real(analytic_signal * np.exp(1j * phase_shift))
                
                phase_shifted_data[b, ch] = torch.from_numpy(phase_shifted_signal).float()
        
        return phase_shifted_data.to(eeg_data.device)
    
    def mixup(self, 
              eeg_batch1: torch.Tensor,
              eeg_batch2: torch.Tensor,
              labels1: torch.Tensor,
              labels2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation between two batches.
        
        Args:
            eeg_batch1: First EEG batch
            eeg_batch2: Second EEG batch
            labels1: Labels for first batch
            labels2: Labels for second batch
            
        Returns:
            Mixed EEG data and labels
        """
        batch_size = eeg_batch1.shape[0]
        
        # Sample mixing coefficients
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
        lam = torch.from_numpy(lam).float().to(eeg_batch1.device)
        
        # Reshape lambda for broadcasting
        lam_eeg = lam.view(batch_size, 1, 1)
        lam_labels = lam.view(batch_size, 1)
        
        # Mix EEG data
        mixed_eeg = lam_eeg * eeg_batch1 + (1 - lam_eeg) * eeg_batch2
        
        # Mix labels (for regression targets)
        if labels1.dtype == torch.float:
            mixed_labels = lam_labels * labels1 + (1 - lam_labels) * labels2
        else:
            # For classification, return both labels and mixing coefficients
            mixed_labels = torch.stack([labels1, labels2, lam], dim=1)
        
        return mixed_eeg, mixed_labels
    
    def cutmix(self,
               eeg_batch1: torch.Tensor,
               eeg_batch2: torch.Tensor,
               labels1: torch.Tensor,
               labels2: torch.Tensor,
               beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation for EEG data.
        
        Args:
            eeg_batch1: First EEG batch
            eeg_batch2: Second EEG batch
            labels1: Labels for first batch
            labels2: Labels for second batch
            beta: Beta parameter for cut ratio
            
        Returns:
            CutMix EEG data and labels
        """
        batch_size, n_channels, n_samples = eeg_batch1.shape
        
        # Sample cut ratio
        lam = np.random.beta(beta, beta)
        
        # Random cut coordinates
        cut_length = int(n_samples * np.sqrt(1 - lam))
        cut_start = np.random.randint(0, n_samples - cut_length + 1)
        cut_end = cut_start + cut_length
        
        # Create mixed data
        mixed_eeg = eeg_batch1.clone()
        mixed_eeg[:, :, cut_start:cut_end] = eeg_batch2[:, :, cut_start:cut_end]
        
        # Calculate actual mixing ratio
        actual_lam = 1 - (cut_end - cut_start) / n_samples
        
        # Mix labels
        if labels1.dtype == torch.float:
            mixed_labels = actual_lam * labels1 + (1 - actual_lam) * labels2
        else:
            mixed_labels = torch.stack([
                labels1, labels2, 
                torch.full_like(labels1, actual_lam, dtype=torch.float)
            ], dim=1)
        
        return mixed_eeg, mixed_labels
    
    def temporal_jittering(self, eeg_data: torch.Tensor, jitter_std: float = 0.1) -> torch.Tensor:
        """Apply temporal jittering to EEG signals.
        
        Args:
            eeg_data: EEG tensor (batch_size, channels, samples)
            jitter_std: Standard deviation of jittering noise
            
        Returns:
            Temporally jittered EEG data
        """
        batch_size, n_channels, n_samples = eeg_data.shape
        
        # Create jittering offsets
        jitter = torch.randn(batch_size, n_channels, n_samples) * jitter_std
        jitter = torch.cumsum(jitter, dim=2)  # Cumulative sum for smooth jittering
        
        # Apply jittering by interpolation
        jittered_data = torch.zeros_like(eeg_data)
        
        for b in range(batch_size):
            for ch in range(n_channels):
                # Create new time indices
                original_indices = torch.arange(n_samples, dtype=torch.float)
                jittered_indices = original_indices + jitter[b, ch]
                
                # Clamp to valid range
                jittered_indices = torch.clamp(jittered_indices, 0, n_samples - 1)
                
                # Interpolate
                jittered_data[b, ch] = torch.from_numpy(
                    np.interp(
                        original_indices.numpy(),
                        jittered_indices.numpy(),
                        eeg_data[b, ch].numpy()
                    )
                ).float()
        
        return jittered_data.to(eeg_data.device)
    
    def electrode_noise(self, eeg_data: torch.Tensor, noise_prob: float = 0.1) -> torch.Tensor:
        """Simulate electrode noise and artifacts.
        
        Args:
            eeg_data: EEG tensor (batch_size, channels, samples)
            noise_prob: Probability of adding noise to each channel
            
        Returns:
            EEG data with simulated electrode noise
        """
        batch_size, n_channels, n_samples = eeg_data.shape
        noisy_data = eeg_data.clone()
        
        for b in range(batch_size):
            for ch in range(n_channels):
                if torch.rand(1) < noise_prob:
                    # Different types of electrode noise
                    noise_type = torch.randint(0, 3, (1,)).item()
                    
                    if noise_type == 0:
                        # High-frequency noise
                        noise = torch.randn(n_samples) * 0.2
                        noisy_data[b, ch] += noise
                    elif noise_type == 1:
                        # Baseline drift
                        drift = torch.linspace(0, torch.randn(1) * 0.5, n_samples)
                        noisy_data[b, ch] += drift
                    else:
                        # Sudden amplitude changes
                        change_points = torch.randint(0, n_samples, (2,))
                        start, end = torch.min(change_points), torch.max(change_points)
                        amplitude_change = torch.randn(1) * 0.3
                        noisy_data[b, ch, start:end] += amplitude_change
        
        return noisy_data