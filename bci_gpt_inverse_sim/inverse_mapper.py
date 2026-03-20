"""
Inverse Mapper for BCI-GPT Inverse Simulation.

Maps band-power feature vectors to vocabulary logits using a linear model:
    logits = W.T @ features + b

This "inverse" mapping (EEG features → word logits) is the core of the
BCI-GPT inverse simulation concept.
"""

import numpy as np


class InverseMapper:
    """
    Linear inverse mapper: feature vector → vocabulary logits.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector (e.g. n_channels * n_bands).
    n_vocab : int
        Number of vocabulary items (word classes). Default: 10.
    seed : int or None
        Random seed for weight initialization.
    """

    def __init__(self, n_features: int, n_vocab: int = 10, seed: int | None = 42):
        self.n_features = n_features
        self.n_vocab = n_vocab
        rng = np.random.default_rng(seed)

        # W: (n_features, n_vocab), b: (n_vocab,)
        # Xavier / Glorot-style initialization
        scale = np.sqrt(2.0 / (n_features + n_vocab))
        self.W = rng.normal(0, scale, size=(n_features, n_vocab))
        self.b = np.zeros(n_vocab)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Compute vocabulary logits from a feature vector.

        Parameters
        ----------
        features : np.ndarray
            Feature vector of shape (n_features,).

        Returns
        -------
        np.ndarray
            Logits of shape (n_vocab,).
        """
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.n_features,):
            raise ValueError(
                f"Expected feature vector of shape ({self.n_features},), "
                f"got {features.shape}"
            )
        return self.W.T @ features + self.b

    def set_weights(self, W: np.ndarray, b: np.ndarray) -> None:
        """
        Manually set the weight matrix and bias vector.

        Parameters
        ----------
        W : np.ndarray
            Shape (n_features, n_vocab).
        b : np.ndarray
            Shape (n_vocab,).
        """
        if W.shape != (self.n_features, self.n_vocab):
            raise ValueError(f"Expected W shape {(self.n_features, self.n_vocab)}, got {W.shape}")
        if b.shape != (self.n_vocab,):
            raise ValueError(f"Expected b shape ({self.n_vocab},), got {b.shape}")
        self.W = W.copy()
        self.b = b.copy()
