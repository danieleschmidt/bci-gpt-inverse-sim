"""
Tests for bci_gpt_inverse_sim package.

Covers: EEG shape, band ranges, feature extraction, logit shapes,
softmax normalization, top-k ordering, and pipeline integration.
"""

import numpy as np
import pytest

from bci_gpt_inverse_sim.eeg_sim import EEGSignalSim, N_WORDS, N_CHANNELS, WORD_FREQUENCIES
from bci_gpt_inverse_sim.band_power import BandPowerExtractor, BANDS, N_BANDS
from bci_gpt_inverse_sim.inverse_mapper import InverseMapper
from bci_gpt_inverse_sim.decoder import TopKDecoder, softmax
from bci_gpt_inverse_sim.pipeline import BCIPipeline


# ---------------------------------------------------------------------------
# EEGSignalSim tests
# ---------------------------------------------------------------------------

class TestEEGSignalSim:
    def setup_method(self):
        self.sim = EEGSignalSim(seed=0)

    def test_output_shape_default(self):
        """generate() should return (n_channels, n_samples) for default params."""
        eeg = self.sim.generate(0)
        assert eeg.shape == (N_CHANNELS, 256), f"Expected (8, 256), got {eeg.shape}"

    def test_output_shape_custom_sr_duration(self):
        """generate() shape matches duration * sr."""
        eeg = self.sim.generate(2, duration=2.0, sr=512)
        assert eeg.shape == (N_CHANNELS, 1024)

    def test_all_word_ids_valid(self):
        """All word IDs 0–9 should produce valid signals."""
        for word_id in range(N_WORDS):
            eeg = self.sim.generate(word_id)
            assert eeg.shape[0] == N_CHANNELS
            assert eeg.shape[1] > 0

    def test_invalid_word_id_raises(self):
        """word_id outside [0, 9] should raise ValueError."""
        with pytest.raises(ValueError):
            self.sim.generate(10)
        with pytest.raises(ValueError):
            self.sim.generate(-1)

    def test_signals_are_distinct(self):
        """Different word IDs should produce numerically distinct signals."""
        eeg0 = EEGSignalSim(seed=0).generate(0)
        eeg1 = EEGSignalSim(seed=0).generate(1)
        # Different word IDs produce different signals (different dominant frequencies)
        assert not np.allclose(eeg0, eeg1), "Word 0 and word 1 should not produce identical signals"

    def test_word_frequencies_count(self):
        """There should be exactly 10 word frequencies."""
        assert len(WORD_FREQUENCIES) == N_WORDS == 10

    def test_no_nan_in_output(self):
        """EEG signal should not contain NaN or Inf."""
        for word_id in range(N_WORDS):
            eeg = self.sim.generate(word_id)
            assert np.all(np.isfinite(eeg)), f"Non-finite values in word {word_id} signal"


# ---------------------------------------------------------------------------
# BandPowerExtractor tests
# ---------------------------------------------------------------------------

class TestBandPowerExtractor:
    def setup_method(self):
        self.extractor = BandPowerExtractor()
        self.sim = EEGSignalSim(seed=1)

    def test_feature_vector_length(self):
        """Feature vector should have length n_channels * n_bands."""
        eeg = self.sim.generate(0)
        features = self.extractor.extract(eeg, sr=256)
        assert len(features) == N_CHANNELS * N_BANDS

    def test_features_non_negative(self):
        """All band powers should be non-negative."""
        for word_id in range(N_WORDS):
            eeg = self.sim.generate(word_id)
            features = self.extractor.extract(eeg, sr=256)
            assert np.all(features >= 0), f"Negative band power in word {word_id}"

    def test_bands_cover_correct_ranges(self):
        """Band definitions should include all standard EEG bands."""
        band_names = [b[0] for b in BANDS]
        for expected in ("delta", "theta", "alpha", "beta", "gamma"):
            assert expected in band_names

    def test_feature_names_length(self):
        """feature_names() should return n_channels * n_bands names."""
        names = self.extractor.feature_names(n_channels=N_CHANNELS)
        assert len(names) == N_CHANNELS * N_BANDS

    def test_dominant_frequency_reflected(self):
        """
        A pure sine at a known frequency should have highest power
        in the corresponding band.
        """
        sr = 512
        duration = 2.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)

        # Alpha band: 10 Hz sine, single channel
        alpha_signal = np.sin(2 * np.pi * 10 * t).reshape(1, -1)
        features = self.extractor.extract(alpha_signal, sr=sr)
        # features has 1 channel × 5 bands = 5 values: delta, theta, alpha, beta, gamma
        band_names = [b[0] for b in BANDS]
        alpha_idx = band_names.index("alpha")
        assert features[alpha_idx] == max(features), (
            f"Alpha band should have highest power for 10 Hz signal; "
            f"features={features}"
        )


# ---------------------------------------------------------------------------
# InverseMapper tests
# ---------------------------------------------------------------------------

class TestInverseMapper:
    def setup_method(self):
        self.n_features = N_CHANNELS * N_BANDS  # 40
        self.mapper = InverseMapper(n_features=self.n_features, seed=42)

    def test_logits_shape(self):
        """forward() should return logits of shape (n_vocab,)."""
        features = np.ones(self.n_features)
        logits = self.mapper.forward(features)
        assert logits.shape == (10,)

    def test_logits_are_finite(self):
        """Logits should contain no NaN or Inf."""
        features = np.random.default_rng(0).standard_normal(self.n_features)
        logits = self.mapper.forward(features)
        assert np.all(np.isfinite(logits))

    def test_wrong_feature_shape_raises(self):
        """forward() with wrong feature shape should raise ValueError."""
        with pytest.raises(ValueError):
            self.mapper.forward(np.ones(self.n_features + 1))

    def test_set_weights(self):
        """set_weights() should update W and b correctly."""
        W = np.eye(self.n_features, 10)
        b = np.ones(10)
        self.mapper.set_weights(W, b)
        np.testing.assert_array_equal(self.mapper.W, W)
        np.testing.assert_array_equal(self.mapper.b, b)


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------

class TestDecoder:
    def setup_method(self):
        self.decoder = TopKDecoder()

    def test_softmax_sums_to_one(self):
        """softmax output should sum to 1."""
        logits = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
        probs = softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_softmax_all_positive(self):
        """softmax output should be all positive."""
        logits = np.array([-100.0, 0.0, 100.0])
        probs = softmax(logits)
        assert np.all(probs > 0)

    def test_softmax_numerical_stability(self):
        """softmax should not overflow with large logits."""
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_topk_returns_k_items(self):
        """decode() should return exactly k items."""
        logits = np.arange(10, dtype=float)
        results = self.decoder.decode(logits, k=3)
        assert len(results) == 3

    def test_topk_sorted_descending(self):
        """decode() results should be sorted by probability descending."""
        logits = np.array([0.1, 5.0, 2.0, 8.0, 1.0, 3.0, 0.5, 4.0, 7.0, 6.0])
        results = self.decoder.decode(logits, k=5)
        probs = [p for _, p in results]
        assert probs == sorted(probs, reverse=True)

    def test_topk_highest_logit_is_first(self):
        """Word with highest logit should have highest probability and appear first."""
        logits = np.zeros(10)
        logits[7] = 10.0  # Word 7 should dominate
        results = self.decoder.decode(logits, k=3)
        assert results[0][0] == 7

    def test_topk_k_clamped_to_vocab(self):
        """decode() with k > n_vocab should return n_vocab items."""
        logits = np.arange(10, dtype=float)
        results = self.decoder.decode(logits, k=20)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def setup_method(self):
        self.pipeline = BCIPipeline(sr=256, duration=1.0, seed=0)

    def test_pipeline_run_returns_expected_keys(self):
        """run() should return a dict with all expected keys."""
        result = self.pipeline.run(0, k=3)
        for key in ("word_id", "word", "eeg_shape", "features", "logits", "top_k"):
            assert key in result

    def test_pipeline_eeg_shape(self):
        """Pipeline EEG shape should match (N_CHANNELS, sr * duration)."""
        result = self.pipeline.run(0)
        assert result["eeg_shape"] == (N_CHANNELS, 256)

    def test_pipeline_feature_length(self):
        """Pipeline features should have correct length."""
        result = self.pipeline.run(0)
        assert len(result["features"]) == N_CHANNELS * N_BANDS

    def test_pipeline_logits_shape(self):
        """Pipeline logits should have shape (n_vocab,)."""
        result = self.pipeline.run(0)
        assert result["logits"].shape == (10,)

    def test_pipeline_run_all(self):
        """run_all() should return results for all 10 words."""
        results = self.pipeline.run_all(k=3)
        assert len(results) == N_WORDS
        for i, result in enumerate(results):
            assert result["word_id"] == i
