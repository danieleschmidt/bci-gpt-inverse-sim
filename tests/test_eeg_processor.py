"""Tests for EEG processing functionality."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from bci_gpt.preprocessing.eeg_processor import EEGProcessor, SignalQuality


class TestEEGProcessor:
    """Test cases for EEG processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = EEGProcessor(
            sampling_rate=1000,
            channels=['Fz', 'Cz', 'Pz'],
            reference='average'
        )
        
        # Create synthetic EEG data
        self.n_channels = 3
        self.n_samples = 2000
        self.test_data = self._create_test_eeg_data()
    
    def _create_test_eeg_data(self):
        """Create synthetic EEG data for testing."""
        np.random.seed(42)
        
        # Generate realistic EEG with different frequency components
        t = np.linspace(0, 2, self.n_samples)  # 2 seconds
        eeg_data = np.zeros((self.n_channels, self.n_samples))
        
        for ch in range(self.n_channels):
            # Alpha rhythm
            alpha = 20 * np.sin(2 * np.pi * 10 * t)
            # Beta rhythm  
            beta = 10 * np.sin(2 * np.pi * 20 * t)
            # Noise
            noise = np.random.normal(0, 5, self.n_samples)
            
            eeg_data[ch] = alpha + beta + noise
            
        return eeg_data
    
    def test_processor_initialization(self):
        """Test EEG processor initialization."""
        assert self.processor.sampling_rate == 1000
        assert len(self.processor.channels) == 3
        assert self.processor.reference == 'average'
    
    def test_generate_synthetic_eeg(self):
        """Test synthetic EEG generation."""
        synthetic_data = self.processor._generate_synthetic_eeg(duration=1.0)
        
        assert synthetic_data.shape == (3, 1000)  # 3 channels, 1000 samples
        assert not np.all(synthetic_data == 0)  # Should not be all zeros
        
        # Check for realistic amplitude range
        assert np.all(np.abs(synthetic_data) < 200)  # Reasonable μV range
    
    def test_preprocess_basic(self):
        """Test basic preprocessing pipeline."""
        result = self.processor.preprocess(
            self.test_data,
            bandpass=(1, 30),
            artifact_removal=None,
            epoch_length=1.0
        )
        
        assert 'data' in result
        assert 'sampling_rate' in result
        assert 'channels' in result
        assert 'metadata' in result
        
        processed_data = result['data']
        assert processed_data.ndim == 3  # epochs x channels x samples
        assert processed_data.shape[1] == self.n_channels
    
    def test_notch_filter(self):
        """Test notch filter application."""
        # Add 60 Hz line noise
        t = np.linspace(0, 2, self.n_samples)
        line_noise = 10 * np.sin(2 * np.pi * 60 * t)
        noisy_data = self.test_data.copy()
        noisy_data[0] += line_noise
        
        # Apply notch filter
        filtered_data = self.processor._apply_notch_filter(noisy_data)
        
        # Check that 60 Hz component is reduced
        assert not np.array_equal(filtered_data, noisy_data)
        assert np.std(filtered_data[0]) < np.std(noisy_data[0])
    
    def test_bandpass_filter(self):
        """Test bandpass filter."""
        filtered_data = self.processor._apply_bandpass_filter(
            self.test_data, (8, 12)  # Alpha band
        )
        
        assert filtered_data.shape == self.test_data.shape
        assert not np.array_equal(filtered_data, self.test_data)
    
    def test_reference_methods(self):
        """Test different referencing methods."""
        # Average reference
        avg_ref_data = self.processor._apply_reference(self.test_data)
        assert avg_ref_data.shape == self.test_data.shape
        
        # Common reference
        self.processor.reference = 'common'
        common_ref_data = self.processor._apply_reference(self.test_data)
        assert common_ref_data.shape == self.test_data.shape
        
        # Specific channel reference
        self.processor.reference = 'Cz'
        spec_ref_data = self.processor._apply_reference(self.test_data)
        assert spec_ref_data.shape == self.test_data.shape
    
    def test_create_epochs(self):
        """Test epoch creation."""
        epochs = self.processor._create_epochs(self.test_data, epoch_length=1.0)
        
        expected_epochs = self.n_samples // 1000  # 1000 samples per epoch
        assert epochs.shape[0] == expected_epochs
        assert epochs.shape[1] == self.n_channels
        assert epochs.shape[2] == 1000
    
    def test_ica_artifact_removal(self):
        """Test ICA artifact removal."""
        # This test might be skipped if MNE is not available
        try:
            cleaned_data = self.processor._apply_ica_artifact_removal(self.test_data)
            assert cleaned_data.shape == self.test_data.shape
        except ImportError:
            pytest.skip("MNE not available for ICA testing")
    
    def test_asr_artifact_removal(self):
        """Test ASR artifact removal."""
        # Add extreme artifacts
        artifact_data = self.test_data.copy()
        artifact_data[0, 500:600] = 1000  # Large artifact
        
        cleaned_data = self.processor._apply_asr_artifact_removal(artifact_data)
        
        assert cleaned_data.shape == artifact_data.shape
        # Check that extreme values are reduced
        assert np.max(cleaned_data[0, 500:600]) < np.max(artifact_data[0, 500:600])
    
    def test_load_data_fallback(self):
        """Test data loading with fallback to synthetic data."""
        with patch('warnings.warn'):
            data = self.processor.load_data("nonexistent_file.fif")
            
            # Should return synthetic data
            assert data.shape[0] == len(self.processor.channels)
            assert data.shape[1] > 0


class TestSignalQuality:
    """Test cases for signal quality assessment."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.channels = ['Fz', 'Cz', 'Pz', 'F3', 'F4']
        self.n_samples = 5000
        self.sampling_rate = 1000
        
    def _create_good_quality_eeg(self):
        """Create good quality EEG data."""
        np.random.seed(42)
        n_channels = len(self.channels)
        
        t = np.linspace(0, 5, self.n_samples)
        eeg_data = np.zeros((n_channels, self.n_samples))
        
        for ch in range(n_channels):
            # Realistic EEG components
            alpha = 15 * np.sin(2 * np.pi * 10 * t + ch * 0.1)
            beta = 8 * np.sin(2 * np.pi * 20 * t + ch * 0.2)
            noise = np.random.normal(0, 3, self.n_samples)
            
            eeg_data[ch] = alpha + beta + noise
            
        return eeg_data
    
    def _create_poor_quality_eeg(self):
        """Create poor quality EEG data."""
        np.random.seed(123)
        n_channels = len(self.channels)
        eeg_data = np.zeros((n_channels, self.n_samples))
        
        for ch in range(n_channels):
            if ch == 0:
                # Flat channel
                eeg_data[ch] = np.ones(self.n_samples) * 0.1
            elif ch == 1:
                # Very noisy channel
                eeg_data[ch] = np.random.normal(0, 100, self.n_samples)
            else:
                # Normal channel
                t = np.linspace(0, 5, self.n_samples)
                eeg_data[ch] = 10 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 5, self.n_samples)
                
        return eeg_data
    
    def test_assess_good_quality(self):
        """Test assessment of good quality EEG."""
        good_eeg = self._create_good_quality_eeg()
        
        quality = SignalQuality.assess(
            good_eeg, self.channels, self.sampling_rate
        )
        
        assert quality.score > 50  # Should be reasonable quality
        assert len(quality.good_channels) >= 3  # Most channels should be good
        assert quality.noise_level > 0  # Should detect some noise
        assert isinstance(quality.artifacts_detected, list)
    
    def test_assess_poor_quality(self):
        """Test assessment of poor quality EEG."""
        poor_eeg = self._create_poor_quality_eeg()
        
        quality = SignalQuality.assess(
            poor_eeg, self.channels, self.sampling_rate
        )
        
        assert quality.score < 80  # Should detect quality issues
        assert len(quality.bad_channels) > 0  # Should identify bad channels
        assert "many_bad_channels" in quality.artifacts_detected or len(quality.bad_channels) > 0
    
    def test_empty_data(self):
        """Test assessment with minimal data."""
        empty_data = np.zeros((len(self.channels), 100))
        
        quality = SignalQuality.assess(
            empty_data, self.channels, self.sampling_rate
        )
        
        assert quality.score < 50  # Should detect poor quality
        assert len(quality.bad_channels) > 0  # Should identify flat channels
    
    def test_single_channel(self):
        """Test assessment with single channel."""
        single_channel_data = self._create_good_quality_eeg()[:1, :]
        
        quality = SignalQuality.assess(
            single_channel_data, ['Cz'], self.sampling_rate
        )
        
        assert isinstance(quality.score, float)
        assert len(quality.good_channels) <= 1
        assert len(quality.bad_channels) <= 1


if __name__ == "__main__":
    pytest.main([__file__])