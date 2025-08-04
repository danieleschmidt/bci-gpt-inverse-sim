"""Real-time EEG decoding for thought-to-text conversion."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import threading
import time
import warnings
from dataclasses import dataclass

from ..core.models import BCIGPTModel
from ..preprocessing.eeg_processor import EEGProcessor
from .confidence_estimation import ConfidenceEstimator


@dataclass
class DecodingResult:
    """Result from real-time decoding."""
    text: str
    confidence: float
    token_probabilities: np.ndarray
    processing_time: float
    timestamp: float


class RealtimeDecoder:
    """Real-time EEG decoder for continuous thought-to-text conversion."""
    
    def __init__(self,
                 model_checkpoint: str,
                 device: str = "cuda",
                 buffer_size: int = 1000,  # milliseconds
                 confidence_threshold: float = 0.7,
                 sampling_rate: int = 1000,
                 channels: Optional[List[str]] = None,
                 overlap_ratio: float = 0.5):
        """Initialize real-time decoder.
        
        Args:
            model_checkpoint: Path to trained BCI-GPT model
            device: Device for inference ("cuda" or "cpu")
            buffer_size: Buffer size in milliseconds
            confidence_threshold: Minimum confidence for output
            sampling_rate: EEG sampling rate
            channels: EEG channel names
            overlap_ratio: Overlap ratio between consecutive windows
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.sampling_rate = sampling_rate
        self.overlap_ratio = overlap_ratio
        
        # Load model
        try:
            self.model = BCIGPTModel.from_pretrained(model_checkpoint)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Could not load model from {model_checkpoint}: {e}")
            # Create a dummy model for testing
            self.model = BCIGPTModel()
            self.model.to(self.device)
            self.model.eval()
        
        # EEG processor
        self.eeg_processor = EEGProcessor(
            sampling_rate=sampling_rate,
            channels=channels
        )
        
        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator()
        
        # Buffer for incoming EEG data
        buffer_samples = int(buffer_size * sampling_rate / 1000)
        self.eeg_buffer = deque(maxlen=buffer_samples * 2)  # Double size for overlap
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.results_queue = deque(maxlen=100)
        
        # Statistics
        self.processing_times = deque(maxlen=100)
        self.confidence_scores = deque(maxlen=100)
        
    def start_decoding(self, eeg_stream) -> None:
        """Start real-time decoding from EEG stream.
        
        Args:
            eeg_stream: EEG data stream object with .get_data() method
        """
        if self.is_running:
            warnings.warn("Decoder already running")
            return
            
        self.eeg_stream = eeg_stream
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        print(f"Real-time decoder started on {self.device}")
    
    def stop_decoding(self) -> None:
        """Stop real-time decoding."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        print("Real-time decoder stopped")
    
    def get_text(self) -> Optional[str]:
        """Get the latest decoded text.
        
        Returns:
            Decoded text string or None if no new results
        """
        if self.results_queue:
            result = self.results_queue.popleft()
            return result.text
        return None
    
    def get_latest_result(self) -> Optional[DecodingResult]:
        """Get the latest decoding result with metadata.
        
        Returns:
            DecodingResult object or None if no new results
        """
        if self.results_queue:
            return self.results_queue.popleft()
        return None
    
    def get_token_probabilities(self) -> Optional[np.ndarray]:
        """Get token probabilities from latest decoding.
        
        Returns:
            Token probabilities array or None
        """
        if self.results_queue:
            # Peek at latest result without removing it
            result = self.results_queue[-1]
            return result.token_probabilities
        return None
    
    def get_top_k_tokens(self, k: int = 5) -> Optional[List[Tuple[str, float]]]:
        """Get top-k token predictions.
        
        Args:
            k: Number of top tokens to return
            
        Returns:
            List of (token_string, probability) tuples
        """
        token_probs = self.get_token_probabilities()
        if token_probs is None or not hasattr(self.model, 'tokenizer') or not self.model.tokenizer:
            return None
            
        # Get top-k indices
        top_k_indices = np.argsort(token_probs)[-k:][::-1]
        
        # Convert to tokens and probabilities
        tokenizer = self.model.tokenizer
        top_k_tokens = []
        
        for idx in top_k_indices:
            try:
                token_str = tokenizer.decode([idx])
                prob = token_probs[idx]
                top_k_tokens.append((token_str, prob))
            except:
                continue
                
        return top_k_tokens
    
    def get_statistics(self) -> Dict[str, float]:
        """Get decoding performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.processing_times:
            return {}
            
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            'buffer_utilization': len(self.eeg_buffer) / self.eeg_buffer.maxlen,
            'results_queue_size': len(self.results_queue)
        }
    
    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        window_samples = int(self.buffer_size * self.sampling_rate / 1000)
        step_samples = int(window_samples * (1 - self.overlap_ratio))
        
        last_process_time = 0
        process_interval = step_samples / self.sampling_rate  # seconds
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to process
                if current_time - last_process_time < process_interval:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                # Get new EEG data
                if hasattr(self.eeg_stream, 'get_data'):
                    new_data = self.eeg_stream.get_data()
                    if new_data is not None and len(new_data) > 0:
                        # Add to buffer (assuming data is channels x samples)
                        if new_data.ndim == 2:
                            for sample_idx in range(new_data.shape[1]):
                                self.eeg_buffer.append(new_data[:, sample_idx])
                
                # Check if we have enough data to process
                if len(self.eeg_buffer) >= window_samples:
                    # Extract window of data
                    window_data = np.array(list(self.eeg_buffer)[-window_samples:])
                    window_data = window_data.T  # Convert to channels x samples
                    
                    # Process the window
                    result = self._process_window(window_data, current_time)
                    
                    if result and result.confidence >= self.confidence_threshold:
                        self.results_queue.append(result)
                    
                    last_process_time = current_time
                    
            except Exception as e:
                warnings.warn(f"Error in processing loop: {e}")
                time.sleep(0.01)
    
    def _process_window(self, eeg_data: np.ndarray, timestamp: float) -> Optional[DecodingResult]:
        """Process a single window of EEG data.
        
        Args:
            eeg_data: EEG data window (channels x samples)
            timestamp: Timestamp of the window
            
        Returns:
            DecodingResult or None if processing failed
        """
        start_time = time.time()
        
        try:
            # Preprocess EEG data
            processed_data = self.eeg_processor.preprocess(
                eeg_data,
                epoch_length=self.buffer_size / 1000.0
            )
            
            # Convert to torch tensor
            if processed_data['data'].ndim == 3:
                # Take first epoch if multiple epochs
                eeg_tensor = torch.FloatTensor(processed_data['data'][0]).unsqueeze(0)
            else:
                eeg_tensor = torch.FloatTensor(processed_data['data']).unsqueeze(0)
                
            eeg_tensor = eeg_tensor.to(self.device)
            
            # Forward pass through model
            with torch.no_grad():
                outputs = self.model(eeg_tensor)
                logits = outputs['logits']
                
                # Get token probabilities
                if logits.dim() == 3:
                    # Take last time step for sequence prediction
                    token_logits = logits[0, -1, :]  # (vocab_size,)
                else:
                    token_logits = logits[0]  # (vocab_size,)
                
                token_probs = F.softmax(token_logits, dim=0).cpu().numpy()
                
                # Decode to text
                if hasattr(self.model, 'tokenizer') and self.model.tokenizer:
                    # Sample from distribution
                    token_id = torch.multinomial(F.softmax(token_logits, dim=0), 1).item()
                    text = self.model.tokenizer.decode([token_id], skip_special_tokens=True)
                else:
                    # Fallback text generation
                    text = self._simple_token_decode(token_probs)
                
                # Estimate confidence
                confidence = self.confidence_estimator.estimate_confidence(
                    token_probs, 
                    eeg_tensor.cpu().numpy()
                )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processing_times.append(processing_time)
            self.confidence_scores.append(confidence)
            
            return DecodingResult(
                text=text,
                confidence=confidence,
                token_probabilities=token_probs,
                processing_time=processing_time,
                timestamp=timestamp
            )
            
        except Exception as e:
            warnings.warn(f"Error processing EEG window: {e}")
            return None
    
    def _simple_token_decode(self, token_probs: np.ndarray) -> str:
        """Simple token decoding when tokenizer is not available.
        
        Args:
            token_probs: Token probability distribution
            
        Returns:
            Decoded text string
        """
        # Simple mapping of high probability to common words
        max_prob_idx = np.argmax(token_probs)
        max_prob = token_probs[max_prob_idx]
        
        # Simple heuristic mapping
        if max_prob > 0.8:
            common_words = ["the", "and", "a", "to", "of", "in", "I", "you", "is", "it"]
            word_idx = max_prob_idx % len(common_words)
            return common_words[word_idx]
        elif max_prob > 0.5:
            return "word"
        else:
            return ""


class StreamBuffer:
    """Thread-safe buffer for streaming EEG data."""
    
    def __init__(self, buffer_size: int, n_channels: int):
        self.buffer = deque(maxlen=buffer_size)
        self.n_channels = n_channels
        self.lock = threading.Lock()
        
    def add_data(self, new_data: np.ndarray) -> None:
        """Add new EEG data to buffer.
        
        Args:
            new_data: New EEG data (channels x samples or samples x channels)
        """
        with self.lock:
            if new_data.ndim == 1:
                # Single sample
                self.buffer.append(new_data)
            elif new_data.ndim == 2:
                # Multiple samples
                if new_data.shape[0] == self.n_channels:
                    # channels x samples
                    for i in range(new_data.shape[1]):
                        self.buffer.append(new_data[:, i])
                else:
                    # samples x channels
                    for i in range(new_data.shape[0]):
                        self.buffer.append(new_data[i, :])
    
    def get_window(self, window_size: int) -> Optional[np.ndarray]:
        """Get a window of data from buffer.
        
        Args:
            window_size: Number of samples to retrieve
            
        Returns:
            EEG data window (channels x samples) or None if insufficient data
        """
        with self.lock:
            if len(self.buffer) < window_size:
                return None
                
            # Get last window_size samples
            window_data = np.array(list(self.buffer)[-window_size:])
            
            # Ensure correct shape (channels x samples)
            if window_data.shape[1] == self.n_channels:
                return window_data.T
            else:
                return window_data