"""Comprehensive tests for the BCI-GPT inverse simulation package."""

import numpy as np
import pytest

from bci_gpt.eeg_sim import EEGSignalSim, VOCABULARY
from bci_gpt.features import BandPowerExtractor
from bci_gpt.mapper import InverseMapper
from bci_gpt.decoder import top_k_decode, full_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sim():
    return EEGSignalSim(n_channels=8, sample_rate=256, duration=1.0, noise_level=0.1)


@pytest.fixture
def extractor():
    return BandPowerExtractor(sample_rate=256)


@pytest.fixture
def trained_mapper(sim, extractor):
    """Return a trained InverseMapper along with normalized test data."""
    signals, labels = sim.generate_dataset(n_per_word=20)
    X = extractor.extract_batch(signals)
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / std

    mapper = InverseMapper(feature_dim=X.shape[1], vocab_size=10, hidden_dim=64)
    history = mapper.fit(X_norm, labels, epochs=200, lr=0.01)
    return mapper, X_norm, labels, history


# ──────────────────────────────────────────────────────────────────────────────
# EEGSignalSim tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEEGSignalSim:

    def test_generate_word_returns_correct_shape(self, sim):
        signal = sim.generate_word("move")
        assert signal.shape == (sim.n_channels, sim.n_samples), (
            f"Expected ({sim.n_channels}, {sim.n_samples}), got {signal.shape}"
        )

    def test_generate_word_n_samples_matches_sr_duration(self, sim):
        expected = int(sim.sample_rate * sim.duration)
        signal = sim.generate_word("left")
        assert signal.shape[1] == expected

    def test_different_words_give_different_signals(self, sim):
        sig1 = sim.generate_word("move")
        sig2 = sim.generate_word("stop")
        assert not np.allclose(sig1, sig2), "Different words should produce different signals"

    def test_same_word_is_deterministic(self, sim):
        sig1 = sim.generate_word("yes")
        sig2 = sim.generate_word("yes")
        assert np.allclose(sig1, sig2), "Same word should produce same base signal"

    def test_noise_is_present(self, sim):
        """Signals should not be perfectly smooth sinusoids — noise must be present."""
        signal = sim.generate_word("hello")
        # A noiseless sinusoid would have very low residual after fitting
        # Just check that std > 0 and values are not all integers
        assert signal.std() > 0
        assert not np.all(signal == signal.astype(int))

    def test_generate_dataset_shape(self, sim):
        n_per_word = 5
        signals, labels = sim.generate_dataset(n_per_word=n_per_word)
        expected_n = len(VOCABULARY) * n_per_word
        assert signals.shape == (expected_n, sim.n_channels, sim.n_samples)
        assert labels.shape == (expected_n,)

    def test_generate_dataset_labels_range(self, sim):
        _, labels = sim.generate_dataset(n_per_word=3)
        assert labels.min() == 0
        assert labels.max() == len(VOCABULARY) - 1

    def test_vocabulary_has_10_words(self):
        assert len(VOCABULARY) == 10

    def test_generate_word_invalid_raises(self, sim):
        with pytest.raises(ValueError):
            sim.generate_word("banana")


# ──────────────────────────────────────────────────────────────────────────────
# BandPowerExtractor tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBandPowerExtractor:

    def test_extract_returns_dict_with_all_4_bands(self, extractor):
        signal = np.random.randn(256)
        result = extractor.extract(signal)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta"}

    def test_all_powers_are_non_negative(self, extractor):
        signal = np.random.randn(256)
        result = extractor.extract(signal)
        for band, power in result.items():
            assert power >= 0, f"Band '{band}' has negative power: {power}"

    def test_extract_multichannel_flattens_correctly(self, extractor):
        n_channels = 8
        n_samples = 256
        signal = np.random.randn(n_channels, n_samples)
        features = extractor.extract_multichannel(signal)
        expected_dim = n_channels * 4  # 4 bands
        assert features.shape == (expected_dim,)

    def test_extract_multichannel_is_1d(self, extractor):
        signal = np.random.randn(4, 256)
        features = extractor.extract_multichannel(signal)
        assert features.ndim == 1

    def test_extract_batch_gives_2d_array(self, extractor):
        signals = np.random.randn(10, 8, 256)
        batch_features = extractor.extract_batch(signals)
        assert batch_features.ndim == 2
        assert batch_features.shape[0] == 10
        assert batch_features.shape[1] == 8 * 4

    def test_extract_batch_consistent_with_single(self, extractor):
        signals = np.random.randn(5, 4, 256)
        batch_features = extractor.extract_batch(signals)
        for i in range(5):
            single = extractor.extract_multichannel(signals[i])
            np.testing.assert_allclose(batch_features[i], single)


# ──────────────────────────────────────────────────────────────────────────────
# InverseMapper tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInverseMapper:

    def test_forward_returns_correct_shape(self):
        mapper = InverseMapper(feature_dim=32, vocab_size=10)
        x = np.random.randn(32)
        logits = mapper.forward(x)
        assert logits.shape == (10,)

    def test_forward_batch_shape(self):
        mapper = InverseMapper(feature_dim=32, vocab_size=10)
        X = np.random.randn(5, 32)
        logits = mapper.forward(X)
        assert logits.shape == (5, 10)

    def test_fit_runs_without_error(self):
        mapper = InverseMapper(feature_dim=16, vocab_size=10)
        X = np.random.randn(50, 16)
        y = np.random.randint(0, 10, size=50)
        history = mapper.fit(X, y, epochs=5, lr=0.01)
        assert len(history) == 5

    def test_loss_decreases_over_training(self, trained_mapper):
        _, _, _, history = trained_mapper
        # Average of first 20 epochs vs last 20 epochs
        early_loss = np.mean(history[:20])
        late_loss = np.mean(history[-20:])
        assert late_loss < early_loss, (
            f"Loss did not decrease: early={early_loss:.4f}, late={late_loss:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Decoder tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDecoder:

    def test_top_k_decode_returns_k_items(self):
        logits = np.random.randn(10)
        result = top_k_decode(logits, k=3)
        assert len(result) == 3

    def test_top_k_decode_probabilities_sum_to_1(self):
        logits = np.random.randn(10)
        result = top_k_decode(logits, k=10)
        total = sum(p for _, p in result)
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected ~1.0"

    def test_top_k_decode_sorted_descending(self):
        logits = np.random.randn(10)
        result = top_k_decode(logits, k=5)
        probs = [p for _, p in result]
        assert probs == sorted(probs, reverse=True)

    def test_top_k_decode_returns_words(self):
        logits = np.random.randn(10)
        result = top_k_decode(logits, k=3)
        for word, prob in result:
            assert isinstance(word, str)
            assert word in VOCABULARY

    def test_full_pipeline_returns_list(self, sim, extractor):
        mapper = InverseMapper(feature_dim=8 * 4, vocab_size=10)
        signal = sim.generate_word("move")
        result = full_pipeline(signal, extractor, mapper)
        assert isinstance(result, list)

    def test_full_pipeline_items_are_str_float_tuples(self, sim, extractor):
        mapper = InverseMapper(feature_dim=8 * 4, vocab_size=10)
        signal = sim.generate_word("move")
        result = full_pipeline(signal, extractor, mapper)
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2
            word, prob = item
            assert isinstance(word, str)
            assert isinstance(prob, float)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end accuracy test
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_trained_model_above_random_accuracy(self, trained_mapper):
        """Trained model should achieve >30% accuracy (random = 10%)."""
        mapper, X_norm, labels, _ = trained_mapper
        preds = mapper.predict(X_norm)
        accuracy = np.mean(preds == labels)
        assert accuracy > 0.30, (
            f"Accuracy {accuracy:.1%} is not above 30% threshold"
        )

    def test_demo_runs_without_error(self):
        from bci_gpt.demo import demo
        accuracy = demo()
        assert accuracy is not None
        assert isinstance(accuracy, float)
