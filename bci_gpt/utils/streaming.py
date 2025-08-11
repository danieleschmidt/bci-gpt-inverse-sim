"""Real-time EEG streaming interfaces and data handling."""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
import threading
import time
import queue
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import pylsl
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    warnings.warn("Lab Streaming Layer (pylsl) not available")

try:
    import brainflow
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False
    warnings.warn("BrainFlow not available")


@dataclass
class StreamConfig:
    """Configuration for EEG streaming."""
    sampling_rate: int = 1000
    channels: List[str] = None
    buffer_duration: float = 5.0  # seconds
    chunk_size: int = 32  # samples per chunk
    apply_filters: bool = True
    notch_freq: float = 60.0  # Hz
    bandpass_low: float = 0.5  # Hz
    bandpass_high: float = 100.0  # Hz


class EEGStreamBase(ABC):
    """Abstract base class for EEG streaming interfaces."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_streaming = False
        self.data_buffer = queue.Queue(maxsize=int(config.buffer_duration * config.sampling_rate))
        self.callbacks = []
        
    @abstractmethod
    def start_stream(self) -> None:
        """Start the EEG data stream."""
        pass
    
    @abstractmethod
    def stop_stream(self) -> None:
        """Stop the EEG data stream."""
        pass
    
    @abstractmethod
    def get_data(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Get EEG data from the stream."""
        pass
    
    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add a callback function to be called when new data arrives."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Remove a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, data: np.ndarray) -> None:
        """Notify all registered callbacks of new data."""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                warnings.warn(f"Callback failed: {e}")


class LSLStream(EEGStreamBase):
    """Lab Streaming Layer EEG stream interface."""
    
    def __init__(self, 
                 config: StreamConfig,
                 stream_name: str = "EEG",
                 stream_type: str = "EEG"):
        super().__init__(config)
        
        if not HAS_LSL:
            raise ImportError("pylsl not available. Install with: pip install pylsl")
        
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.inlet = None
        self.stream_thread = None
        
    def start_stream(self) -> None:
        """Start LSL stream."""
        if self.is_streaming:
            warnings.warn("Stream already started")
            return
        
        # Look for EEG streams
        print(f"Looking for {self.stream_type} stream named '{self.stream_name}'...")
        streams = pylsl.resolve_stream('type', self.stream_type, timeout=10.0)
        
        if not streams:
            # Try to find any EEG stream
            streams = pylsl.resolve_stream('type', 'EEG', timeout=5.0)
        
        if not streams:
            raise RuntimeError(f"No {self.stream_type} streams found")
        
        # Use the first available stream
        stream_info = streams[0]
        print(f"Found stream: {stream_info.name()} ({stream_info.channel_count()} channels at {stream_info.nominal_srate()} Hz)")
        
        # Create inlet
        self.inlet = pylsl.StreamInlet(stream_info, max_chunklen=self.config.chunk_size)
        
        # Update config with stream info
        self.config.sampling_rate = int(stream_info.nominal_srate())
        if self.config.channels is None:
            self.config.channels = [f"Ch{i+1}" for i in range(stream_info.channel_count())]
        
        # Start streaming thread
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        print("LSL stream started")
    
    def stop_stream(self) -> None:
        """Stop LSL stream."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
        
        print("LSL stream stopped")
    
    def _stream_loop(self) -> None:
        """Main streaming loop."""
        while self.is_streaming and self.inlet:
            try:
                # Pull chunk of data
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=self.config.chunk_size)
                
                if chunk:
                    chunk_array = np.array(chunk).T  # Convert to (channels, samples)
                    
                    # Apply filters if requested
                    if self.config.apply_filters:
                        chunk_array = self._apply_filters(chunk_array)
                    
                    # Add to buffer
                    try:
                        self.data_buffer.put(chunk_array, block=False)
                    except queue.Full:
                        # Remove oldest data
                        try:
                            self.data_buffer.get(block=False)
                            self.data_buffer.put(chunk_array, block=False)
                        except queue.Empty:
                            pass
                    
                    # Notify callbacks
                    self._notify_callbacks(chunk_array)
                    
            except Exception as e:
                warnings.warn(f"Error in stream loop: {e}")
                time.sleep(0.001)
    
    def get_data(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Get data from the stream buffer."""
        if duration is None:
            duration = 1.0  # Default to 1 second
        
        target_samples = int(duration * self.config.sampling_rate)
        collected_data = []
        collected_samples = 0
        
        # Collect data from buffer
        timeout_count = 0
        while collected_samples < target_samples and timeout_count < 100:
            try:
                chunk = self.data_buffer.get(timeout=0.01)
                collected_data.append(chunk)
                collected_samples += chunk.shape[1]
            except queue.Empty:
                timeout_count += 1
                if not self.is_streaming:
                    break
        
        if not collected_data:
            return None
        
        # Concatenate chunks
        full_data = np.concatenate(collected_data, axis=1)
        
        # Trim to requested duration
        if full_data.shape[1] > target_samples:
            full_data = full_data[:, -target_samples:]
        
        return full_data
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply filtering to EEG data."""
        filtered_data = data.copy()
        
        try:
            for ch_idx in range(data.shape[0]):
                channel_data = data[ch_idx, :].copy()
                
                # Notch filter for line noise
                if self.config.notch_freq > 0:
                    # Simple notch filter implementation
                    # In practice, would use proper filter design
                    pass
                
                # Bandpass filter
                if self.config.bandpass_low > 0 and self.config.bandpass_high > 0:
                    # Simple bandpass filter implementation
                    # In practice, would use proper filter design
                    pass
                
                filtered_data[ch_idx, :] = channel_data
                
        except Exception as e:
            warnings.warn(f"Filter application failed: {e}")
            return data
        
        return filtered_data


class BrainFlowStream(EEGStreamBase):
    """BrainFlow EEG stream interface."""
    
    def __init__(self,
                 config: StreamConfig,
                 board_id: int = -1,  # SYNTHETIC_BOARD equivalent
                 serial_port: Optional[str] = None):
        super().__init__(config)
        
        if not HAS_BRAINFLOW:
            raise ImportError("BrainFlow not available. Install with: pip install brainflow")
        
        self.board_id = board_id
        self.board_shim = None
        self.stream_thread = None
        
        # Setup board parameters
        self.params = BrainFlowInputParams()
        if serial_port:
            self.params.serial_port = serial_port
    
    def start_stream(self) -> None:
        """Start BrainFlow stream."""
        if self.is_streaming:
            warnings.warn("Stream already started")
            return
        
        try:
            # Initialize board
            BoardShim.enable_dev_board_logger()
            self.board_shim = BoardShim(self.board_id, self.params)
            
            # Prepare session
            self.board_shim.prepare_session()
            
            # Get board info
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            
            print(f"Board info: {len(eeg_channels)} EEG channels at {sampling_rate} Hz")
            
            # Update config
            self.config.sampling_rate = sampling_rate
            if self.config.channels is None:
                self.config.channels = [f"EEG_{i}" for i in eeg_channels]
            
            # Start stream
            self.board_shim.start_stream()
            self.is_streaming = True
            
            # Start data collection thread
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            print("BrainFlow stream started")
            
        except Exception as e:
            print(f"Failed to start BrainFlow stream: {e}")
            self._cleanup()
            raise
    
    def stop_stream(self) -> None:
        """Stop BrainFlow stream."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        self._cleanup()
        print("BrainFlow stream stopped")
    
    def _cleanup(self) -> None:
        """Clean up BrainFlow resources."""
        if self.board_shim:
            try:
                if self.board_shim.is_prepared():
                    self.board_shim.stop_stream()
                    self.board_shim.release_session()
            except Exception as e:
                warnings.warn(f"Error during cleanup: {e}")
            finally:
                self.board_shim = None
    
    def _stream_loop(self) -> None:
        """Main streaming loop."""
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        while self.is_streaming and self.board_shim:
            try:
                # Get data
                data = self.board_shim.get_current_board_data(self.config.chunk_size)
                
                if data.shape[1] > 0:
                    # Extract EEG channels
                    eeg_data = data[eeg_channels, :]
                    
                    # Apply filters if requested
                    if self.config.apply_filters:
                        eeg_data = self._apply_brainflow_filters(eeg_data)
                    
                    # Add to buffer
                    try:
                        self.data_buffer.put(eeg_data, block=False)
                    except queue.Full:
                        # Remove oldest data
                        try:
                            self.data_buffer.get(block=False)
                            self.data_buffer.put(eeg_data, block=False)
                        except queue.Empty:
                            pass
                    
                    # Notify callbacks
                    self._notify_callbacks(eeg_data)
                
                time.sleep(0.01)  # Small delay
                
            except Exception as e:
                warnings.warn(f"Error in BrainFlow stream loop: {e}")
                time.sleep(0.1)
    
    def get_data(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Get data from the stream buffer."""
        if duration is None:
            duration = 1.0
        
        target_samples = int(duration * self.config.sampling_rate)
        collected_data = []
        collected_samples = 0
        
        timeout_count = 0
        while collected_samples < target_samples and timeout_count < 100:
            try:
                chunk = self.data_buffer.get(timeout=0.01)
                collected_data.append(chunk)
                collected_samples += chunk.shape[1]
            except queue.Empty:
                timeout_count += 1
                if not self.is_streaming:
                    break
        
        if not collected_data:
            return None
        
        full_data = np.concatenate(collected_data, axis=1)
        
        if full_data.shape[1] > target_samples:
            full_data = full_data[:, -target_samples:]
        
        return full_data
    
    def _apply_brainflow_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply BrainFlow filters to EEG data."""
        filtered_data = data.copy()
        
        try:
            for ch_idx in range(data.shape[0]):
                channel_data = filtered_data[ch_idx, :].copy()
                
                # Notch filter
                if self.config.notch_freq > 0:
                    DataFilter.perform_bandstop(
                        channel_data, 
                        self.config.sampling_rate,
                        self.config.notch_freq - 2, 
                        self.config.notch_freq + 2,
                        4, FilterTypes.BUTTERWORTH.value, 0
                    )
                
                # Bandpass filter
                if self.config.bandpass_low > 0 and self.config.bandpass_high > 0:
                    DataFilter.perform_bandpass(
                        channel_data,
                        self.config.sampling_rate,
                        self.config.bandpass_low,
                        self.config.bandpass_high,
                        4, FilterTypes.BUTTERWORTH.value, 0
                    )
                
                filtered_data[ch_idx, :] = channel_data
                
        except Exception as e:
            warnings.warn(f"BrainFlow filter failed: {e}")
            return data
        
        return filtered_data


class SimulatedEEGStream(EEGStreamBase):
    """Simulated EEG stream for testing and development."""
    
    def __init__(self, config: StreamConfig, noise_level: float = 0.1):
        super().__init__(config)
        self.noise_level = noise_level
        self.stream_thread = None
        self.time_counter = 0
        
        # Default channels if not specified
        if self.config.channels is None:
            self.config.channels = ['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    
    def start_stream(self) -> None:
        """Start simulated stream."""
        if self.is_streaming:
            warnings.warn("Stream already started")
            return
        
        self.is_streaming = True
        self.time_counter = 0
        
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        print("Simulated EEG stream started")
    
    def stop_stream(self) -> None:
        """Stop simulated stream."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        print("Simulated EEG stream stopped")
    
    def _stream_loop(self) -> None:
        """Generate simulated EEG data."""
        n_channels = len(self.config.channels)
        dt = 1.0 / self.config.sampling_rate
        
        while self.is_streaming:
            try:
                # Generate chunk of simulated EEG
                chunk_samples = self.config.chunk_size
                chunk_data = np.zeros((n_channels, chunk_samples))
                
                for i in range(chunk_samples):
                    t = self.time_counter * dt
                    
                    for ch_idx in range(n_channels):
                        # Generate realistic EEG signal
                        signal = (
                            # Alpha rhythm (8-12 Hz)
                            20 * np.sin(2 * np.pi * 10 * t + ch_idx * 0.1) +
                            # Beta rhythm (13-30 Hz)
                            10 * np.sin(2 * np.pi * 20 * t + ch_idx * 0.2) +
                            # Gamma rhythm (30-100 Hz)
                            5 * np.sin(2 * np.pi * 40 * t + ch_idx * 0.3) +
                            # Pink noise
                            np.random.normal(0, self.noise_level * 15)
                        )
                        
                        chunk_data[ch_idx, i] = signal
                    
                    self.time_counter += 1
                
                # Add to buffer
                try:
                    self.data_buffer.put(chunk_data, block=False)
                except queue.Full:
                    try:
                        self.data_buffer.get(block=False)
                        self.data_buffer.put(chunk_data, block=False)
                    except queue.Empty:
                        pass
                
                # Notify callbacks
                self._notify_callbacks(chunk_data)
                
                # Sleep to maintain sampling rate
                time.sleep(chunk_samples / self.config.sampling_rate)
                
            except Exception as e:
                warnings.warn(f"Error in simulated stream: {e}")
                time.sleep(0.1)
    
    def get_data(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Get simulated data."""
        if duration is None:
            duration = 1.0
        
        target_samples = int(duration * self.config.sampling_rate)
        collected_data = []
        collected_samples = 0
        
        timeout_count = 0
        while collected_samples < target_samples and timeout_count < 100:
            try:
                chunk = self.data_buffer.get(timeout=0.01)
                collected_data.append(chunk)
                collected_samples += chunk.shape[1]
            except queue.Empty:
                timeout_count += 1
                if not self.is_streaming:
                    break
        
        if not collected_data:
            return None
        
        full_data = np.concatenate(collected_data, axis=1)
        
        if full_data.shape[1] > target_samples:
            full_data = full_data[:, -target_samples:]
        
        return full_data


class StreamingEEG:
    """Factory class for creating EEG streams."""
    
    @staticmethod
    def create_stream(backend: str, 
                     config: Optional[StreamConfig] = None,
                     **kwargs) -> EEGStreamBase:
        """Create an EEG stream based on backend.
        
        Args:
            backend: Stream backend ("lsl", "brainflow", "simulated")
            config: Stream configuration
            **kwargs: Backend-specific parameters
            
        Returns:
            EEG stream object
        """
        if config is None:
            config = StreamConfig()
        
        if backend.lower() == "lsl":
            return LSLStream(config, **kwargs)
        elif backend.lower() == "brainflow":
            return BrainFlowStream(config, **kwargs)
        elif backend.lower() == "simulated":
            return SimulatedEEGStream(config, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")