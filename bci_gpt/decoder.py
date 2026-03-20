"""Decoder: Top-k decoding from EEG logits to vocabulary words."""

import numpy as np
from typing import List, Tuple

from .eeg_sim import VOCABULARY
from .features import BandPowerExtractor
from .mapper import InverseMapper


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def top_k_decode(
    logits: np.ndarray,
    k: int = 3,
    vocab: List[str] = VOCABULARY,
) -> List[Tuple[str, float]]:
    """Decode top-k words from logits.
    
    Args:
        logits: Raw logits of shape (vocab_size,).
        k: Number of top candidates to return.
        vocab: List of vocabulary words.
        
    Returns:
        List of (word, probability) tuples sorted by probability descending.
    """
    probs = _softmax(logits)
    k = min(k, len(vocab))
    top_indices = np.argsort(probs)[::-1][:k]
    return [(vocab[i], float(probs[i])) for i in top_indices]


def full_pipeline(
    eeg_signal: np.ndarray,
    extractor: BandPowerExtractor,
    mapper: InverseMapper,
    vocab: List[str] = VOCABULARY,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Run the full BCI decoding pipeline on a raw EEG signal.
    
    Args:
        eeg_signal: Multi-channel EEG signal of shape (n_channels, n_samples).
        extractor: Fitted BandPowerExtractor instance.
        mapper: Trained InverseMapper instance.
        vocab: List of vocabulary words.
        k: Number of top candidates to return.
        
    Returns:
        List of (word, probability) tuples sorted by probability descending.
    """
    features = extractor.extract_multichannel(eeg_signal)
    logits = mapper.forward(features)
    return top_k_decode(logits, k=k, vocab=vocab)
