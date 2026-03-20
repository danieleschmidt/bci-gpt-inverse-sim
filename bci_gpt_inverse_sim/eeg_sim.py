"""
EEG Signal Simulator for BCI-GPT Inverse Simulation.

Generates synthetic EEG patterns for 10 vocabulary "words",
each with a distinct dominant frequency signature across 8 channels.
"""

import numpy as np

# Dominant frequencies for each of 10 word classes (in Hz)
WORD_FREQUENCIES = [1.5, 3.0, 5.5, 7.0, 9.5, 12.0, 18.0, 25.0, 35.0, 45.0]

N_WORDS = 10
N_CHANNELS = 8


class EEGSignalSim:
    """
    Simulates EEG signals for a vocabulary of 10 words.

    Each word has a unique dominant frequency. Channels show slight
    variations around the dominant frequency to mimic real EEG spatial patterns.
    Gaussian noise is added to all signals.
    """

    def __init__(self, noise_std: float = 0.1, seed: int | None = None):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.word_frequencies = WORD_FREQUENCIES
        self.n_words = N_WORDS
        self.n_channels = N_CHANNELS

    def generate(self, word_id: int, duration: float = 1.0, sr: int = 256) -> np.ndarray:
        """
        Generate a synthetic EEG signal for a given word ID.

        Parameters
        ----------
        word_id : int
            Word index in [0, 9].
        duration : float
            Signal duration in seconds.
        sr : int
            Sampling rate in Hz.

        Returns
        -------
        np.ndarray
            EEG signal of shape (n_channels, n_samples).
        """
        if not (0 <= word_id < self.n_words):
            raise ValueError(f"word_id must be in [0, {self.n_words - 1}], got {word_id}")

        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        base_freq = self.word_frequencies[word_id]
        signal = np.zeros((self.n_channels, n_samples))

        for ch in range(self.n_channels):
            # Slight per-channel frequency variation (±5%)
            freq_variation = 1.0 + 0.05 * (ch - self.n_channels / 2) / self.n_channels
            freq = base_freq * freq_variation

            # Dominant sinusoidal component
            amplitude = 1.0 + 0.1 * self.rng.standard_normal()
            phase = self.rng.uniform(0, 2 * np.pi)
            channel_signal = amplitude * np.sin(2 * np.pi * freq * t + phase)

            # Add a harmonic at 2x frequency (weaker)
            harmonic_amp = 0.3 * amplitude
            channel_signal += harmonic_amp * np.sin(2 * np.pi * 2 * freq * t + phase)

            # Add Gaussian noise
            noise = self.rng.normal(0, self.noise_std, n_samples)
            signal[ch] = channel_signal + noise

        return signal
