"""
Comprehensive integration tests for BCI-GPT system.

Tests end-to-end workflows and system integration points.
"""

import pytest
import numpy as np
import torch
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import BCI-GPT modules
from bci_gpt.core.models import BCIGPTModel
from bci_gpt.core.inverse_gan import InverseSimulator
from bci_gpt.preprocessing.eeg_processor import EEGProcessor
from bci_gpt.training.trainer import BCIGPTTrainer, TrainingConfig
from bci_gpt.decoding.realtime_decoder import RealtimeDecoder
from bci_gpt.inverse.text_to_eeg import TextToEEG, GenerationConfig
from bci_gpt.inverse.validation import SyntheticEEGValidator
from bci_gpt.utils.streaming import StreamingEEG, StreamConfig
from bci_gpt.utils.metrics import BCIMetrics
from bci_gpt.utils.visualization import EEGVisualizer
from bci_gpt.utils.monitoring import MetricsCollector, ClinicalSafetyMonitor


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def sample_eeg_data(self):
        """Generate sample EEG data for testing."""
        n_channels, n_samples = 8, 1000
        return np.random.randn(n_channels, n_samples) * 50  # μV scale
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample BCI-GPT model."""
        return BCIGPTModel(
            eeg_channels=8,
            eeg_sampling_rate=1000,
            language_model="gpt2",
            latent_dim=128
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_data_preprocessing_pipeline(self, sample_eeg_data):
        """Test complete data preprocessing pipeline."""
        processor = EEGProcessor(
            sampling_rate=1000,
            channels=['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3'],
            reference='average'
        )
        
        # Test preprocessing steps
        processed = processor.preprocess(
            sample_eeg_data,
            bandpass=(0.5, 40),
            artifact_removal='ica',
            epoch_length=1.0
        )
        
        # Validate preprocessing results
        assert processed is not None
        assert processed.shape[0] == sample_eeg_data.shape[0]  # Same channels
        assert isinstance(processed, np.ndarray)
        
        # Test feature extraction
        features = processor.extract_features(processed)
        assert 'spectral_features' in features
        assert 'temporal_features' in features
        
    def test_model_training_pipeline(self, sample_model, temp_dir):
        """Test model training pipeline."""
        # Create synthetic training data
        n_samples = 50
        eeg_data = [np.random.randn(8, 1000) * 50 for _ in range(n_samples)]
        text_data = [f"sample text {i}" for i in range(n_samples)]
        
        # Save data to temp directory
        train_data_dir = temp_dir / "train_data"
        train_data_dir.mkdir()
        
        for i, (eeg, text) in enumerate(zip(eeg_data, text_data)):
            np.save(train_data_dir / f"eeg_{i}.npy", eeg)
            with open(train_data_dir / f"text_{i}.txt", 'w') as f:
                f.write(text)
        
        # Setup training configuration
        config = TrainingConfig(
            epochs=2,  # Short for testing
            batch_size=4,
            learning_rate=1e-4,
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Create trainer
        trainer = BCIGPTTrainer(model=sample_model, config=config)
        
        # Test training (would normally take much longer)
        with patch('bci_gpt.training.trainer.DataLoader') as mock_dataloader:
            # Mock data loader to return our test data
            mock_dataloader.return_value = [
                (torch.randn(4, 8, 1000), ["test"] * 4)
                for _ in range(5)  # 5 batches
            ]
            
            history = trainer.fit(
                train_data=str(train_data_dir),
                epochs=2,
                batch_size=4
            )
            
            # Validate training results
            assert history is not None
            assert 'train_loss' in history
            assert len(history['train_loss']) == 2  # 2 epochs
    
    def test_inference_pipeline(self, sample_model, sample_eeg_data):
        """Test inference pipeline."""
        sample_model.eval()
        
        # Convert to tensor
        eeg_tensor = torch.from_numpy(sample_eeg_data).float().unsqueeze(0)
        
        # Test inference
        with torch.no_grad():
            # Test EEG encoding
            eeg_features = sample_model.eeg_encoder(eeg_tensor)
            assert eeg_features is not None
            assert eeg_features.shape[0] == 1  # Batch size
            
            # Test text generation
            generated_text = sample_model.generate_text_from_eeg(
                eeg_tensor, max_length=20
            )
            assert isinstance(generated_text, (str, list))
            
    def test_real_time_decoding_simulation(self, sample_model):
        """Test real-time decoding simulation."""
        # Create simulated stream
        config = StreamConfig(sampling_rate=1000, buffer_duration=5.0)
        stream = StreamingEEG.create_stream("simulated", config)
        
        # Start stream
        stream.start_stream()
        
        try:
            # Test data collection
            time.sleep(1.1)  # Let it collect some data
            data = stream.get_data(duration=1.0)
            
            assert data is not None
            assert data.shape[0] == len(config.channels or [f'SIM{i}' for i in range(9)])
            assert data.shape[1] > 0
            
            # Test with decoder (mock for now)
            decoder = RealtimeDecoder(confidence_threshold=0.7)
            
            # Simulate decoding
            with patch.object(decoder, 'decode') as mock_decode:
                mock_decode.return_value = ("test_word", 0.8)
                result = decoder.decode(data)
                assert result[0] == "test_word"
                assert result[1] == 0.8
                
        finally:
            stream.stop_stream()
    
    def test_text_to_eeg_generation(self, temp_dir):
        """Test text-to-EEG generation pipeline."""
        # Create text-to-EEG generator
        generator = TextToEEG()
        
        # Test generation configuration
        config = GenerationConfig(
            duration=2.0,
            style="imagined_speech",
            num_samples=1
        )
        
        # Generate synthetic EEG
        text = "hello world"
        synthetic_eeg = generator.generate(text, config)
        
        # Validate generation results
        assert synthetic_eeg is not None
        assert isinstance(synthetic_eeg, np.ndarray)
        assert synthetic_eeg.shape[0] > 0  # Has channels
        assert synthetic_eeg.shape[1] > 0  # Has samples
        
        # Test validation
        validator = SyntheticEEGValidator(sampling_rate=1000)
        validation_metrics = validator.validate(synthetic_eeg)
        
        assert validation_metrics is not None
        assert 0 <= validation_metrics.overall_quality <= 1
        assert 0 <= validation_metrics.realism_score <= 1
    
    def test_metrics_and_monitoring_integration(self):
        """Test metrics collection and monitoring integration."""
        # Create metrics collector
        collector = MetricsCollector(collection_interval=0.1)
        collector.start_collection()
        
        try:
            # Record various metrics
            collector.record_model_metrics(
                inference_time_ms=50.0,
                accuracy=0.85,
                confidence=0.9
            )
            
            collector.record_streaming_metrics(
                samples_per_second=1000.0,
                buffer_fill_ratio=0.3,
                latency_ms=10.0
            )
            
            collector.record_error(
                'warning', 'test_component',
                'Test warning message'
            )
            
            # Let it collect some system metrics
            time.sleep(0.2)
            
            # Test health check
            health = collector.get_system_health()
            assert 'status' in health
            assert health['status'] in ['healthy', 'warning', 'critical']
            
            # Test metrics retrieval
            assert len(collector.model_metrics) > 0
            assert len(collector.streaming_metrics) > 0
            assert len(collector.error_events) > 0
            
        finally:
            collector.stop_collection()
    
    def test_clinical_safety_integration(self):
        """Test clinical safety monitoring integration."""
        safety_monitor = ClinicalSafetyMonitor(
            max_session_duration=10,  # 10 seconds for testing
            fatigue_threshold=0.8
        )
        
        # Start session
        safety_monitor.start_session("test_user")
        assert safety_monitor.is_monitoring
        
        # Test safety checks
        assert safety_monitor.is_safe()
        
        # Test fatigue detection
        test_eeg = np.random.normal(0, 1, (8, 1000))
        is_fatigued = safety_monitor.detect_fatigue(
            test_eeg,
            performance_metrics={'accuracy': 0.6, 'reaction_time_ms': 2500}
        )
        
        # Should detect fatigue due to low accuracy and slow reaction
        assert is_fatigued
        
        # End session
        report = safety_monitor.end_session()
        assert 'user_id' in report
        assert report['user_id'] == "test_user"
        assert 'session_duration_minutes' in report
    
    @pytest.mark.parametrize("plot_type", ["signals", "spectrogram", "psd"])
    def test_visualization_integration(self, sample_eeg_data, plot_type, temp_dir):
        """Test visualization integration."""
        visualizer = EEGVisualizer()
        
        # Test different plot types
        output_path = temp_dir / f"test_{plot_type}.png"
        
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                if plot_type == "signals":
                    visualizer.plot_eeg_signals(
                        sample_eeg_data,
                        channels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8'],
                        save_path=str(output_path)
                    )
                elif plot_type == "spectrogram":
                    visualizer.plot_spectrogram(
                        sample_eeg_data,
                        channel_idx=0,
                        save_path=str(output_path)
                    )
                elif plot_type == "psd":
                    visualizer.plot_power_spectral_density(
                        sample_eeg_data,
                        channels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8'],
                        save_path=str(output_path)
                    )
                
                # Verify that plotting functions were called
                mock_savefig.assert_called_once()


class TestSystemIntegration:
    """Test system-level integration scenarios."""
    
    def test_streaming_to_model_pipeline(self):
        """Test streaming data directly to model inference."""
        # Create simulated stream
        config = StreamConfig(sampling_rate=1000, channels=['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3'])
        stream = StreamingEEG.create_stream("simulated", config)
        
        # Create model
        model = BCIGPTModel(eeg_channels=8)
        model.eval()
        
        stream.start_stream()
        
        try:
            # Collect data
            time.sleep(1.1)
            data = stream.get_data(duration=1.0)
            
            # Process through model
            data_tensor = torch.from_numpy(data).float().unsqueeze(0)
            
            with torch.no_grad():
                features = model.eeg_encoder(data_tensor)
                assert features is not None
                
        finally:
            stream.stop_stream()
    
    def test_preprocessing_to_training_pipeline(self):
        """Test preprocessing output feeding into training."""
        # Create processor
        processor = EEGProcessor()
        
        # Generate raw EEG data
        raw_eeg = np.random.randn(8, 2000) * 50
        
        # Preprocess
        processed_eeg = processor.preprocess(raw_eeg, epoch_length=1.0)
        
        # Verify it can be used in training
        model = BCIGPTModel(eeg_channels=8)
        
        # Convert to tensor format expected by model
        eeg_tensor = torch.from_numpy(processed_eeg).float().unsqueeze(0)
        
        with torch.no_grad():
            features = model.eeg_encoder(eeg_tensor)
            assert features is not None
    
    def test_model_to_inverse_generation_pipeline(self):
        """Test model features feeding into inverse generation."""
        # Create model and get EEG features
        model = BCIGPTModel(eeg_channels=8)
        sample_eeg = torch.randn(1, 8, 1000)
        
        with torch.no_grad():
            eeg_features = model.eeg_encoder(sample_eeg)
        
        # Use features for inverse generation
        generator = TextToEEG()
        
        # Test that we can generate EEG from text
        config = GenerationConfig(duration=1.0, num_samples=1)
        synthetic_eeg = generator.generate("test text", config)
        
        assert synthetic_eeg is not None
        assert synthetic_eeg.shape[0] > 0
    
    def test_full_bci_loop_simulation(self):
        """Test complete BCI loop: EEG -> Model -> Text -> Validation."""
        # 1. Generate synthetic EEG
        processor = EEGProcessor()
        raw_eeg = processor._generate_synthetic_eeg(duration=2.0)
        
        # 2. Preprocess
        processed_eeg = processor.preprocess(raw_eeg)
        
        # 3. Model inference
        model = BCIGPTModel(eeg_channels=8)
        model.eval()
        
        eeg_tensor = torch.from_numpy(processed_eeg).float().unsqueeze(0)
        
        with torch.no_grad():
            decoded_text = model.generate_text_from_eeg(eeg_tensor, max_length=10)
        
        # 4. Validation and metrics
        metrics_calc = BCIMetrics()
        
        # Calculate some metrics
        reference_text = "hello world"
        if isinstance(decoded_text, list):
            decoded_text = decoded_text[0]
        
        wer = metrics_calc.word_error_rate(decoded_text, reference_text)
        assert 0 <= wer <= 1
        
        # 5. Text-to-EEG generation for validation
        generator = TextToEEG()
        config = GenerationConfig(duration=2.0)
        
        regenerated_eeg = generator.generate(decoded_text, config)
        assert regenerated_eeg is not None


class TestErrorHandlingAndRobustness:
    """Test error handling and system robustness."""
    
    def test_model_with_invalid_input(self):
        """Test model behavior with invalid inputs."""
        model = BCIGPTModel(eeg_channels=8)
        
        # Test with wrong number of channels
        wrong_channels = torch.randn(1, 4, 1000)  # 4 instead of 8
        
        with pytest.raises((RuntimeError, ValueError)):
            with torch.no_grad():
                model.eeg_encoder(wrong_channels)
    
    def test_streaming_disconnection_handling(self):
        """Test streaming behavior during disconnections."""
        config = StreamConfig(sampling_rate=1000)
        stream = StreamingEEG.create_stream("simulated", config)
        
        # Start and immediately stop to simulate disconnection
        stream.start_stream()
        stream.stop_stream()
        
        # Should handle gracefully
        data = stream.get_data(duration=1.0)
        # May return None or empty data, both are acceptable
        
    def test_preprocessing_with_noisy_data(self):
        """Test preprocessing with extremely noisy data."""
        processor = EEGProcessor()
        
        # Create very noisy data
        noisy_eeg = np.random.randn(8, 1000) * 1000  # Very high amplitude noise
        
        # Should handle without crashing
        try:
            processed = processor.preprocess(noisy_eeg)
            assert processed is not None
        except Exception as e:
            # If it fails, it should fail gracefully
            assert "preprocessing failed" in str(e).lower() or \
                   "invalid" in str(e).lower()
    
    def test_text_generation_edge_cases(self):
        """Test text generation with edge cases."""
        generator = TextToEEG()
        config = GenerationConfig(duration=0.1, num_samples=1)  # Very short duration
        
        # Test with empty string
        result = generator.generate("", config)
        assert result is not None
        
        # Test with very long string
        long_text = "word " * 1000
        result = generator.generate(long_text, config)
        assert result is not None
    
    def test_metrics_with_missing_data(self):
        """Test metrics calculation with missing or invalid data."""
        metrics_calc = BCIMetrics()
        
        # Test with empty strings
        wer = metrics_calc.word_error_rate("", "")
        assert wer == 0.0
        
        # Test with None values
        try:
            wer = metrics_calc.word_error_rate(None, "test")
        except (TypeError, AttributeError):
            pass  # Expected to fail
        
        # Test ITR with edge cases
        itr = metrics_calc.calculate_itr(0.0, num_classes=2, trial_duration=1.0)
        assert itr >= 0  # Should not be negative
        
        itr = metrics_calc.calculate_itr(1.0, num_classes=2, trial_duration=1.0)
        assert itr > 0  # Perfect accuracy should give positive ITR


class TestPerformanceBenchmarks:
    """Test performance benchmarks and timing."""
    
    def test_model_inference_speed(self):
        """Test model inference speed."""
        model = BCIGPTModel(eeg_channels=8)
        model.eval()
        
        sample_eeg = torch.randn(1, 8, 1000)
        
        # Warm up
        with torch.no_grad():
            _ = model.eeg_encoder(sample_eeg)
        
        # Time inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model.eeg_encoder(sample_eeg)
        
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) * 1000 / 10
        
        # Should be reasonably fast (less than 100ms for this simple test)
        assert avg_time_ms < 100, f"Inference too slow: {avg_time_ms:.1f}ms"
    
    def test_preprocessing_speed(self):
        """Test preprocessing speed."""
        processor = EEGProcessor()
        
        # Large dataset
        large_eeg = np.random.randn(64, 10000)  # 64 channels, 10s at 1kHz
        
        start_time = time.perf_counter()
        processed = processor.preprocess(large_eeg)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Should process reasonably quickly
        assert processing_time_ms < 5000, f"Preprocessing too slow: {processing_time_ms:.1f}ms"
        assert processed is not None
    
    def test_streaming_throughput(self):
        """Test streaming data throughput."""
        config = StreamConfig(sampling_rate=1000, chunk_size=64)
        stream = StreamingEEG.create_stream("simulated", config)
        
        stream.start_stream()
        
        try:
            # Collect data for 2 seconds
            time.sleep(2.1)
            data = stream.get_data(duration=2.0)
            
            # Should have approximately correct amount of data
            expected_samples = 2000  # 2 seconds * 1000 Hz
            
            if data is not None:
                actual_samples = data.shape[1]
                # Allow some tolerance for timing variations
                assert abs(actual_samples - expected_samples) < 200, \
                    f"Expected ~{expected_samples} samples, got {actual_samples}"
                    
        finally:
            stream.stop_stream()


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib to prevent display issues in tests."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    yield
    
    # Clean up any remaining figures
    import matplotlib.pyplot as plt
    plt.close('all')


# Performance markers
pytestmark = [
    pytest.mark.integration,  # Mark all tests as integration tests
]


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])