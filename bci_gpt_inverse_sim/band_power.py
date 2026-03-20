"""
Band Power Extractor for BCI-GPT Inverse Simulation.

Extracts standard EEG frequency band powers (delta, theta, alpha, beta, gamma)
from multi-channel EEG signals using Welch's method.
"""

import numpy as np
from scipy.signal import welch

# Standard EEG frequency bands: (name, low_hz, high_hz)
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 100.0),
]

N_BANDS = len(BANDS)


class BandPowerExtractor:
    """
    Extracts per-channel band powers from a multi-channel EEG signal.

    Uses Welch's power spectral density estimation. Returns a feature vector
    of length n_channels * n_bands (5 bands × 8 channels = 40 features by default).
    """

    def __init__(self, bands: list | None = None):
        """
        Parameters
        ----------
        bands : list of (name, low_hz, high_hz), optional
            Frequency bands to extract. Defaults to standard EEG bands.
        """
        self.bands = bands if bands is not None else BANDS

    def extract(self, eeg_signal: np.ndarray, sr: int = 256) -> np.ndarray:
        """
        Extract band power features from an EEG signal.

        Parameters
        ----------
        eeg_signal : np.ndarray
            Shape (n_channels, n_samples).
        sr : int
            Sampling rate in Hz.

        Returns
        -------
        np.ndarray
            Feature vector of shape (n_channels * n_bands,).
        """
        n_channels, n_samples = eeg_signal.shape
        features = []

        for ch in range(n_channels):
            channel_data = eeg_signal[ch]
            # Welch's PSD — nperseg capped at signal length
            nperseg = min(256, n_samples)
            freqs, psd = welch(channel_data, fs=sr, nperseg=nperseg)

            for _, low, high in self.bands:
                # Integrate PSD within the band (trapezoidal rule)
                band_mask = (freqs >= low) & (freqs < high)
                if band_mask.any():
                    band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
                else:
                    band_power = 0.0
                # Ensure non-negative (trapz can give tiny negatives due to float precision)
                features.append(max(band_power, 0.0))

        return np.array(features, dtype=np.float64)

    @property
    def n_features(self) -> int:
        """Return expected feature vector length for a given signal (inferred from usage)."""
        # Computed at extract time based on signal channels; this is a helper.
        return -1  # Dynamic; use extract() to get actual size

    def feature_names(self, n_channels: int = 8) -> list[str]:
        """Return human-readable feature names."""
        names = []
        for ch in range(n_channels):
            for band_name, _, _ in self.bands:
                names.append(f"ch{ch}_{band_name}")
        return names
