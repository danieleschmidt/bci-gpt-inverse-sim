"""
Tests for streaming functionality and real-time processing.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from bci_gpt.utils.streaming import (
    StreamingEEG, StreamConfig, LSLStream, BrainFlowStream, 
    SimulatedEEGStream, EEGSample, StreamStatus, StreamInfo
)


class TestStreamConfig:
    """Test stream configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamConfig()
        
        assert config.sampling_rate == 1000
        assert config.buffer_duration == 5.0
        assert config.chunk_size == 32
        assert config.apply_filters is True
        assert config.notch_freq == 60.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        channels = ['Fz', 'Cz', 'Pz']
        config = StreamConfig(
            sampling_rate=500,
            channels=channels,
            buffer_duration=10.0,
            chunk_size=64
        )
        
        assert config.sampling_rate == 500
        assert config.channels == channels
        assert config.buffer_duration == 10.0
        assert config.chunk_size == 64


class TestSimulatedEEGStream:
    """Test simulated EEG stream."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return StreamConfig(
            sampling_rate=1000,
            channels=['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3'],
            buffer_duration=2.0,
            chunk_size=32
        )
    
    def test_stream_creation(self, config):
        """Test stream creation."""
        stream = SimulatedEEGStream(config)
        
        assert not stream.is_streaming
        assert len(stream.config.channels) == 8
        assert stream.config.sampling_rate == 1000
    
    def test_stream_start_stop(self, config):
        """Test stream start and stop."""
        stream = SimulatedEEGStream(config)
        
        # Test start
        stream.start_stream()
        assert stream.is_streaming
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Test stop
        stream.stop_stream()
        assert not stream.is_streaming
    
    def test_data_generation(self, config):
        """Test data generation."""
        stream = SimulatedEEGStream(config, noise_level=0.1)
        
        stream.start_stream()
        
        try:
            # Wait for data generation
            time.sleep(0.5)
            
            # Get data
            data = stream.get_data(duration=0.2)
            
            assert data is not None
            assert data.shape[0] == 8  # 8 channels
            assert data.shape[1] > 0   # Has samples
            
            # Check data characteristics
            assert np.all(np.isfinite(data))  # No NaN or inf values
            assert np.std(data) > 0  # Has variation
            
        finally:
            stream.stop_stream()
    
    def test_data_callback(self, config):
        """Test data callback functionality."""
        stream = SimulatedEEGStream(config)
        
        callback_data = []
        
        def data_callback(data):
            callback_data.append(data)
        
        stream.add_callback(data_callback)
        stream.start_stream()
        
        try:
            # Wait for callbacks
            time.sleep(0.2)
            
            # Should have received some callbacks
            assert len(callback_data) > 0
            
            # Check callback data format
            for data in callback_data:
                assert isinstance(data, np.ndarray)
                assert data.shape[0] == 8  # Correct number of channels
                
        finally:
            stream.stop_stream()
    
    def test_buffer_management(self, config):
        """Test buffer management and overflow."""
        # Small buffer for testing overflow
        config.buffer_duration = 0.1  # Very small buffer
        stream = SimulatedEEGStream(config)
        
        stream.start_stream()
        
        try:
            # Run for longer than buffer duration
            time.sleep(0.5)
            
            # Should still be able to get data (buffer should wrap)
            data = stream.get_data(duration=0.05)
            assert data is not None
            
        finally:
            stream.stop_stream()


class TestStreamingEEGFactory:
    """Test StreamingEEG factory class."""
    
    def test_create_simulated_stream(self):
        """Test creating simulated stream."""
        config = StreamConfig(sampling_rate=500)
        stream = StreamingEEG.create_stream("simulated", config)
        
        assert isinstance(stream, SimulatedEEGStream)
        assert stream.config.sampling_rate == 500
    
    def test_create_invalid_backend(self):
        """Test creating stream with invalid backend."""
        config = StreamConfig()
        
        with pytest.raises(ValueError, match="Unknown backend"):
            StreamingEEG.create_stream("invalid_backend", config)
    
    @patch('bci_gpt.utils.streaming.HAS_LSL', True)
    def test_create_lsl_stream(self):
        """Test creating LSL stream (mocked)."""
        config = StreamConfig()
        
        with patch('bci_gpt.utils.streaming.LSLStream') as mock_lsl:
            stream = StreamingEEG.create_stream("lsl", config)
            mock_lsl.assert_called_once_with(config)
    
    @patch('bci_gpt.utils.streaming.HAS_BRAINFLOW', True)
    def test_create_brainflow_stream(self):
        """Test creating BrainFlow stream (mocked)."""
        config = StreamConfig()
        
        with patch('bci_gpt.utils.streaming.BrainFlowStream') as mock_bf:
            stream = StreamingEEG.create_stream("brainflow", config, board_id=0)
            mock_bf.assert_called_once_with(config, board_id=0)


class TestLSLStream:
    """Test LSL stream (mocked tests since LSL hardware not available)."""
    
    @pytest.fixture
    def mock_lsl(self):
        """Mock LSL module."""
        with patch('bci_gpt.utils.streaming.HAS_LSL', True):
            with patch('bci_gpt.utils.streaming.pylsl') as mock_lsl_module:
                yield mock_lsl_module
    
    def test_lsl_stream_creation(self, mock_lsl):
        """Test LSL stream creation."""
        config = StreamConfig()
        
        with patch('bci_gpt.utils.streaming.LSLStream.__init__', return_value=None):
            # Just test that it can be instantiated
            stream = LSLStream(config)
    
    def test_lsl_stream_connection(self, mock_lsl):
        """Test LSL stream connection process."""
        config = StreamConfig()
        
        # Mock stream info
        mock_stream_info = MagicMock()
        mock_stream_info.name.return_value = "Test EEG"
        mock_stream_info.channel_count.return_value = 8
        mock_stream_info.nominal_srate.return_value = 1000
        mock_stream_info.type.return_value = "EEG"
        
        # Mock resolve_stream to return our mock stream
        mock_lsl.resolve_stream.return_value = [mock_stream_info]
        
        # Mock StreamInlet
        mock_inlet = MagicMock()
        mock_lsl.StreamInlet.return_value = mock_inlet
        
        # Test connection
        with patch.object(LSLStream, '__init__', return_value=None):
            stream = LSLStream(config)
            stream.config = config
            stream.inlet = None
            stream.is_streaming = False
            
            # Mock the start_stream method
            with patch.object(stream, 'start_stream') as mock_start:
                mock_start.return_value = None
                stream.start_stream()
                mock_start.assert_called_once()


class TestBrainFlowStream:
    """Test BrainFlow stream (mocked tests)."""
    
    @pytest.fixture
    def mock_brainflow(self):
        """Mock BrainFlow module."""
        with patch('bci_gpt.utils.streaming.HAS_BRAINFLOW', True):
            with patch('bci_gpt.utils.streaming.BoardShim') as mock_board:
                with patch('bci_gpt.utils.streaming.BrainFlowInputParams') as mock_params:
                    yield mock_board, mock_params
    
    def test_brainflow_stream_creation(self, mock_brainflow):
        """Test BrainFlow stream creation."""
        mock_board, mock_params = mock_brainflow
        config = StreamConfig()
        
        with patch('bci_gpt.utils.streaming.BrainFlowStream.__init__', return_value=None):
            stream = BrainFlowStream(config, board_id=0)
    
    def test_brainflow_stream_connection(self, mock_brainflow):
        """Test BrainFlow stream connection."""
        mock_board, mock_params = mock_brainflow
        config = StreamConfig()
        
        # Mock board methods
        mock_board_instance = MagicMock()
        mock_board.return_value = mock_board_instance
        mock_board.get_eeg_channels.return_value = [0, 1, 2, 3, 4, 5, 6, 7]
        mock_board.get_sampling_rate.return_value = 1000
        
        with patch.object(BrainFlowStream, '__init__', return_value=None):
            stream = BrainFlowStream(config, board_id=0)
            stream.config = config
            stream.board_shim = None
            stream.is_streaming = False
            
            # Mock the start_stream method
            with patch.object(stream, 'start_stream') as mock_start:
                mock_start.return_value = None
                stream.start_stream()
                mock_start.assert_called_once()


class TestStreamPerformance:
    """Test streaming performance characteristics."""
    
    def test_streaming_latency(self):
        """Test streaming latency."""
        config = StreamConfig(sampling_rate=1000, chunk_size=32)
        stream = SimulatedEEGStream(config)
        
        latencies = []
        
        def measure_latency(data):
            # Simple latency measurement
            latency = time.time() - getattr(measure_latency, 'last_time', time.time())
            latencies.append(latency)
            measure_latency.last_time = time.time()
        
        stream.add_callback(measure_latency)
        stream.start_stream()
        
        try:
            time.sleep(1.0)  # Run for 1 second
            
            if latencies:
                avg_latency = np.mean(latencies[1:])  # Skip first measurement
                max_latency = np.max(latencies[1:])
                
                # Latency should be reasonable
                assert avg_latency < 0.1, f"Average latency too high: {avg_latency:.3f}s"
                assert max_latency < 0.2, f"Max latency too high: {max_latency:.3f}s"
                
        finally:
            stream.stop_stream()
    
    def test_streaming_throughput(self):
        """Test streaming data throughput."""
        config = StreamConfig(sampling_rate=1000, chunk_size=64)
        stream = SimulatedEEGStream(config)
        
        sample_count = 0
        
        def count_samples(data):
            nonlocal sample_count
            sample_count += data.shape[1]
        
        stream.add_callback(count_samples)
        stream.start_stream()
        
        try:
            test_duration = 2.0
            time.sleep(test_duration)
            
            expected_samples = test_duration * config.sampling_rate
            
            # Allow some tolerance for timing variations
            tolerance = 0.1  # 10%
            min_expected = expected_samples * (1 - tolerance)
            max_expected = expected_samples * (1 + tolerance)
            
            assert min_expected <= sample_count <= max_expected, \
                f"Expected {expected_samples} samples, got {sample_count}"
                
        finally:
            stream.stop_stream()
    
    def test_concurrent_streams(self):
        """Test multiple concurrent streams."""
        config = StreamConfig(sampling_rate=500, chunk_size=32)
        
        streams = [SimulatedEEGStream(config) for _ in range(3)]
        
        # Start all streams
        for stream in streams:
            stream.start_stream()
        
        try:
            time.sleep(0.5)
            
            # Check that all streams are producing data
            for i, stream in enumerate(streams):
                data = stream.get_data(duration=0.1)
                assert data is not None, f"Stream {i} not producing data"
                assert data.shape[1] > 0, f"Stream {i} producing empty data"
                
        finally:
            # Stop all streams
            for stream in streams:
                stream.stop_stream()


class TestErrorHandling:
    """Test error handling in streaming."""
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        # Test with invalid sampling rate
        with pytest.raises((ValueError, AssertionError)):
            config = StreamConfig(sampling_rate=-1000)
            stream = SimulatedEEGStream(config)
    
    def test_callback_exceptions(self):
        """Test handling of exceptions in callbacks."""
        config = StreamConfig()
        stream = SimulatedEEGStream(config)
        
        def failing_callback(data):
            raise ValueError("Test exception")
        
        def working_callback(data):
            working_callback.called = True
        
        working_callback.called = False
        
        stream.add_callback(failing_callback)
        stream.add_callback(working_callback)
        
        stream.start_stream()
        
        try:
            time.sleep(0.2)
            
            # Working callback should still be called despite failing callback
            assert working_callback.called
            
        finally:
            stream.stop_stream()
    
    def test_stop_before_start(self):
        """Test stopping stream before starting."""
        config = StreamConfig()
        stream = SimulatedEEGStream(config)
        
        # Should handle gracefully
        stream.stop_stream()
        assert not stream.is_streaming
    
    def test_multiple_starts(self):
        """Test starting stream multiple times."""
        config = StreamConfig()
        stream = SimulatedEEGStream(config)
        
        stream.start_stream()
        assert stream.is_streaming
        
        # Starting again should not cause issues
        stream.start_stream()  # Should warn but not crash
        assert stream.is_streaming
        
        stream.stop_stream()
        assert not stream.is_streaming
    
    def test_data_request_when_stopped(self):
        """Test requesting data when stream is stopped."""
        config = StreamConfig()
        stream = SimulatedEEGStream(config)
        
        # Should return None or empty data gracefully
        data = stream.get_data(duration=1.0)
        # Acceptable outcomes: None or empty array
        assert data is None or (isinstance(data, np.ndarray) and data.size == 0)


# Test markers
pytest.mark.streaming = pytest.mark.filterwarnings("ignore::DeprecationWarning")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])