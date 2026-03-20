"""EEG Band Power Feature Extraction for BCI-GPT."""

import numpy as np
from typing import Dict, List


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


class BandPowerExtractor:
    """Extracts frequency band power features from EEG signals using FFT.
    
    Computes power in delta, theta, alpha, and beta bands.
    """

    def __init__(self, sample_rate: int = 256):
        """Initialize the band power extractor.
        
        Args:
            sample_rate: EEG sampling rate in Hz.
        """
        self.sample_rate = sample_rate
        self.bands = BANDS

    def extract(self, signal_1d: np.ndarray) -> Dict[str, float]:
        """Extract band powers from a 1D EEG signal.
        
        Power is computed as sum(|FFT|^2) within each frequency band.
        
        Args:
            signal_1d: 1D array of shape (n_samples,).
            
        Returns:
            Dict mapping band name to power value.
        """
        n = len(signal_1d)
        fft_vals = np.fft.rfft(signal_1d)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        power_spectrum = np.abs(fft_vals) ** 2

        band_powers = {}
        for band_name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_powers[band_name] = float(np.sum(power_spectrum[mask]))

        return band_powers

    def extract_multichannel(self, signal_channels: np.ndarray) -> np.ndarray:
        """Extract band powers from a multi-channel EEG signal, flattened.
        
        Args:
            signal_channels: 2D array of shape (n_channels, n_samples).
            
        Returns:
            1D feature vector of shape (n_channels * n_bands,).
        """
        features = []
        for ch_signal in signal_channels:
            band_powers = self.extract(ch_signal)
            # Keep consistent order
            for band_name in self.bands:
                features.append(band_powers[band_name])
        return np.array(features)

    def extract_batch(self, signals: np.ndarray) -> np.ndarray:
        """Extract features from a batch of multi-channel EEG signals.
        
        Args:
            signals: 3D array of shape (n_samples, n_channels, n_timepoints).
            
        Returns:
            2D array of shape (n_samples, n_channels * n_bands).
        """
        return np.array([self.extract_multichannel(s) for s in signals])
