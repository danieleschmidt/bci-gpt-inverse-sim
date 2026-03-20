"""
Top-K Decoder for BCI-GPT Inverse Simulation.

Converts vocabulary logits to probabilities via softmax, then returns
the top-k most likely words and their probabilities.
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over a 1-D logit vector.

    Parameters
    ----------
    logits : np.ndarray
        Shape (n,).

    Returns
    -------
    np.ndarray
        Probability distribution of shape (n,). Sums to 1.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)  # Numerical stability
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


class TopKDecoder:
    """
    Decodes vocabulary logits to a ranked list of (word_id, probability) pairs.
    """

    def decode(self, logits: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Return top-k words by probability.

        Parameters
        ----------
        logits : np.ndarray
            Shape (n_vocab,). Raw logits (not yet softmaxed).
        k : int
            Number of top candidates to return. Clamped to n_vocab.

        Returns
        -------
        list of (word_id, probability)
            Sorted by probability descending. Length = min(k, n_vocab).
        """
        logits = np.asarray(logits, dtype=np.float64)
        n_vocab = len(logits)
        k = min(k, n_vocab)

        probs = softmax(logits)

        # Partial sort: get top-k indices
        top_indices = np.argpartition(probs, -k)[-k:]
        # Sort these k indices by probability descending
        top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

        return [(int(idx), float(probs[idx])) for idx in top_indices]
