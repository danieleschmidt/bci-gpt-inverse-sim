"""EEG Signal Simulator for BCI-GPT inverse simulation."""

import numpy as np
from typing import Tuple, List

VOCABULARY = ["move", "stop", "left", "right", "up", "down", "yes", "no", "hello", "goodbye"]


class EEGSignalSim:
    """Simulates EEG signals associated with different vocabulary words.
    
    Each word generates a distinct multi-channel EEG signal based on
    sinusoidal components seeded by the word index, plus Gaussian noise.
    """

    def __init__(
        self,
        n_channels: int = 8,
        sample_rate: int = 256,
        duration: float = 1.0,
        noise_level: float = 0.1,
    ):
        """Initialize the EEG signal simulator.
        
        Args:
            n_channels: Number of EEG channels.
            sample_rate: Sampling rate in Hz.
            duration: Signal duration in seconds.
            noise_level: Standard deviation of Gaussian noise.
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.duration = duration
        self.noise_level = noise_level
        self.n_samples = int(sample_rate * duration)

    def generate_word(self, word: str) -> np.ndarray:
        """Generate an EEG signal for a given vocabulary word.
        
        Args:
            word: A word from VOCABULARY.
            
        Returns:
            EEG signal array of shape (n_channels, n_samples).
            
        Raises:
            ValueError: If word is not in VOCABULARY.
        """
        if word not in VOCABULARY:
            raise ValueError(f"Word '{word}' not in VOCABULARY: {VOCABULARY}")

        word_idx = VOCABULARY.index(word)
        rng = np.random.RandomState(seed=word_idx * 42 + 7)

        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        signal = np.zeros((self.n_channels, self.n_samples))

        # Base frequencies for each word — spread across delta to beta bands
        base_freqs = [
            1.0 + word_idx * 0.5,   # varies in delta/theta
            4.0 + word_idx * 0.3,   # varies in theta/alpha
            8.0 + word_idx * 0.7,   # varies in alpha
            13.0 + word_idx * 0.4,  # varies in beta
        ]

        for ch in range(self.n_channels):
            ch_rng = np.random.RandomState(seed=word_idx * 100 + ch)
            # Each channel gets different amplitude weights per frequency
            amplitudes = ch_rng.uniform(0.5, 2.0, len(base_freqs))
            phases = ch_rng.uniform(0, 2 * np.pi, len(base_freqs))

            for freq, amp, phase in zip(base_freqs, amplitudes, phases):
                signal[ch] += amp * np.sin(2 * np.pi * freq * t + phase)

            # Add Gaussian noise
            noise = rng.normal(0, self.noise_level, self.n_samples)
            signal[ch] += noise

        return signal

    def generate_dataset(
        self, n_per_word: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a dataset of EEG signals for all vocabulary words.
        
        Args:
            n_per_word: Number of samples to generate per word.
            
        Returns:
            Tuple of (signals, labels) where:
                signals: array of shape (n_words * n_per_word, n_channels, n_samples)
                labels: array of shape (n_words * n_per_word,) with word indices
        """
        signals: List[np.ndarray] = []
        labels: List[int] = []

        for word_idx, word in enumerate(VOCABULARY):
            for sample_i in range(n_per_word):
                # Vary noise seed per sample to get slightly different signals
                base_signal = self.generate_word(word)
                rng = np.random.RandomState(seed=word_idx * 1000 + sample_i)
                noise = rng.normal(0, self.noise_level * 0.5, base_signal.shape)
                signals.append(base_signal + noise)
                labels.append(word_idx)

        return np.array(signals), np.array(labels)
