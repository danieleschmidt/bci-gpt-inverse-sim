"""
Full BCI-GPT Inverse Simulation Pipeline.

Demonstrates the end-to-end flow:
    EEG Simulation → Band Power Extraction → Inverse Mapping → Top-K Decoding
"""

import numpy as np

from .eeg_sim import EEGSignalSim, N_WORDS, N_CHANNELS
from .band_power import BandPowerExtractor, N_BANDS
from .inverse_mapper import InverseMapper
from .decoder import TopKDecoder

# Vocabulary for display purposes
VOCAB = [
    "apple", "brain", "cloud", "delta", "echo",
    "flame", "graph", "hello", "image", "joy"
]


class BCIPipeline:
    """
    End-to-end BCI-GPT inverse simulation pipeline.

    Parameters
    ----------
    sr : int
        Sampling rate in Hz.
    duration : float
        EEG segment duration in seconds.
    seed : int or None
        Global random seed.
    """

    def __init__(self, sr: int = 256, duration: float = 1.0, seed: int | None = 0):
        self.sr = sr
        self.duration = duration
        self.n_features = N_CHANNELS * N_BANDS

        self.sim = EEGSignalSim(seed=seed)
        self.extractor = BandPowerExtractor()
        self.mapper = InverseMapper(n_features=self.n_features, n_vocab=N_WORDS, seed=seed)
        self.decoder = TopKDecoder()

    def run(self, word_id: int, k: int = 3) -> dict:
        """
        Run the full pipeline for a given word ID.

        Parameters
        ----------
        word_id : int
            Target word in [0, 9].
        k : int
            Number of top-k results to return.

        Returns
        -------
        dict with keys:
            - word_id: int
            - word: str
            - eeg_shape: tuple
            - features: np.ndarray
            - logits: np.ndarray
            - top_k: list of (word_id, probability)
        """
        # Step 1: Simulate EEG
        eeg = self.sim.generate(word_id, duration=self.duration, sr=self.sr)

        # Step 2: Extract band powers
        features = self.extractor.extract(eeg, sr=self.sr)

        # Step 3: Map to logits
        logits = self.mapper.forward(features)

        # Step 4: Decode top-k
        top_k = self.decoder.decode(logits, k=k)

        return {
            "word_id": word_id,
            "word": VOCAB[word_id],
            "eeg_shape": eeg.shape,
            "features": features,
            "logits": logits,
            "top_k": top_k,
        }

    def run_all(self, k: int = 3) -> list[dict]:
        """Run the pipeline for all 10 words and return results."""
        return [self.run(word_id, k=k) for word_id in range(N_WORDS)]


def demo():
    """Print a demo of the pipeline output for all words."""
    print("=" * 60)
    print("BCI-GPT Inverse Simulation — Pipeline Demo")
    print("=" * 60)

    pipeline = BCIPipeline(sr=256, duration=1.0, seed=42)

    for word_id in range(N_WORDS):
        result = pipeline.run(word_id, k=3)
        print(f"\nWord {word_id:2d} '{result['word']:6s}' | EEG shape: {result['eeg_shape']} | "
              f"Features: {len(result['features'])}d")
        print("  Top-3 predictions:")
        for rank, (wid, prob) in enumerate(result["top_k"], 1):
            marker = "✓" if wid == word_id else " "
            print(f"    {rank}. {VOCAB[wid]:6s} (word {wid:2d}) — {prob:.4f} {marker}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
