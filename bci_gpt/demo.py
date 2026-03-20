"""Demo: End-to-end BCI-GPT inverse simulation pipeline."""

import numpy as np
from .eeg_sim import EEGSignalSim, VOCABULARY
from .features import BandPowerExtractor
from .mapper import InverseMapper
from .decoder import full_pipeline


def demo():
    """Run the full BCI-GPT inverse simulation demo.
    
    - Generates synthetic EEG dataset (10 words x 20 samples)
    - Extracts band power features
    - Trains an InverseMapper MLP
    - Evaluates accuracy on held-out data
    - Shows top-3 decoded words for a test signal
    """
    print("=== BCI-GPT Inverse Simulation Demo ===\n")

    # 1. Generate dataset
    print("Generating EEG dataset...")
    sim = EEGSignalSim(n_channels=8, sample_rate=256, duration=1.0, noise_level=0.1)
    signals, labels = sim.generate_dataset(n_per_word=20)
    print(f"  Dataset shape: {signals.shape}, labels shape: {labels.shape}")

    # 2. Split train/test (80/20)
    rng = np.random.RandomState(seed=42)
    n = len(signals)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = perm[:split]
    test_idx = perm[split:]

    X_train_raw = signals[train_idx]
    y_train = labels[train_idx]
    X_test_raw = signals[test_idx]
    y_test = labels[test_idx]

    # 3. Extract features
    print("Extracting band power features...")
    extractor = BandPowerExtractor(sample_rate=256)
    X_train = extractor.extract_batch(X_train_raw)
    X_test = extractor.extract_batch(X_test_raw)
    print(f"  Feature dim: {X_train.shape[1]} ({sim.n_channels} channels x 4 bands)")

    # 4. Normalize features
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mu) / std
    X_test_norm = (X_test - mu) / std

    # 5. Train InverseMapper
    print("Training InverseMapper (200 epochs)...")
    feature_dim = X_train.shape[1]
    mapper = InverseMapper(feature_dim=feature_dim, vocab_size=10, hidden_dim=64)
    history = mapper.fit(X_train_norm, y_train, epochs=200, lr=0.01)
    print(f"  Initial loss: {history[0]:.4f}  →  Final loss: {history[-1]:.4f}")

    # 6. Evaluate accuracy
    preds = mapper.predict(X_test_norm)
    accuracy = np.mean(preds == y_test)
    print(f"\nTest accuracy: {accuracy:.1%} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
    print(f"  (Random baseline: 10.0%)")

    # 7. Top-3 decode a test signal
    test_signal = X_test_raw[0]
    true_word = VOCABULARY[y_test[0]]
    print(f"\nDecoding test signal (true word: '{true_word}'):")
    results = full_pipeline(test_signal, extractor, mapper)
    for rank, (word, prob) in enumerate(results, 1):
        marker = " ✓" if word == true_word else ""
        print(f"  {rank}. {word:10s}  {prob:.3f}{marker}")

    print("\nDemo complete.")
    return accuracy


if __name__ == "__main__":
    demo()
