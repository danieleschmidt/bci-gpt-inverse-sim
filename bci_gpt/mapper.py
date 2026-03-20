"""Inverse Mapper: Maps EEG band power features to vocabulary logits."""

import numpy as np
from typing import List, Dict


class InverseMapper:
    """2-layer MLP that maps EEG band features to vocabulary logits.
    
    Implemented in pure numpy with cross-entropy loss and backpropagation.
    """

    def __init__(
        self,
        feature_dim: int,
        vocab_size: int = 10,
        hidden_dim: int = 64,
    ):
        """Initialize the InverseMapper.
        
        Args:
            feature_dim: Input feature dimension (n_channels * n_bands).
            vocab_size: Number of vocabulary words (output classes).
            hidden_dim: Number of hidden units in the MLP.
        """
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Initialize weights with Xavier initialization
        rng = np.random.RandomState(seed=42)
        scale1 = np.sqrt(2.0 / feature_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.randn(feature_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, vocab_size) * scale2
        self.b2 = np.zeros(vocab_size)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable softmax
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the MLP.
        
        Args:
            x: Input feature vector of shape (feature_dim,) or (batch, feature_dim).
            
        Returns:
            Logits of shape (vocab_size,) or (batch, vocab_size).
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        h = self._relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        if single:
            return logits[0]
        return logits

    def _forward_with_cache(self, x: np.ndarray):
        """Forward pass caching intermediates for backprop."""
        h_pre = x @ self.W1 + self.b1
        h = self._relu(h_pre)
        logits = h @ self.W2 + self.b2
        return logits, h, h_pre

    def fit(
        self,
        X: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> List[float]:
        """Train the MLP using mini-batch gradient descent.
        
        Args:
            X: Feature matrix of shape (n_samples, feature_dim).
            y_labels: Integer labels of shape (n_samples,).
            epochs: Number of training epochs.
            lr: Learning rate.
            
        Returns:
            List of loss values per epoch.
        """
        history = []
        n = len(X)
        batch_size = min(32, n)

        rng = np.random.RandomState(seed=0)

        for epoch in range(epochs):
            # Shuffle data
            perm = rng.permutation(n)
            X_shuf = X[perm]
            y_shuf = y_labels[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                Xb = X_shuf[start : start + batch_size]
                yb = y_shuf[start : start + batch_size]
                bs = len(Xb)

                # Forward
                logits, h, h_pre = self._forward_with_cache(Xb)
                probs = self._softmax(logits)

                # Cross-entropy loss
                log_probs = np.log(probs[np.arange(bs), yb] + 1e-12)
                loss = -np.mean(log_probs)
                epoch_loss += loss
                n_batches += 1

                # Backward
                dlogits = probs.copy()
                dlogits[np.arange(bs), yb] -= 1
                dlogits /= bs

                dW2 = h.T @ dlogits
                db2 = dlogits.sum(axis=0)

                dh = dlogits @ self.W2.T
                dh_pre = dh * self._relu_grad(h_pre)

                dW1 = Xb.T @ dh_pre
                db1 = dh_pre.sum(axis=0)

                # Update
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            history.append(epoch_loss / n_batches)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for batch input."""
        logits = self.forward(X)
        return np.argmax(logits, axis=-1)
