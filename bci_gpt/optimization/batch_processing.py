"""
Batch processing optimization for BCI-GPT.

This module provides efficient batch processing capabilities for:
- EEG signal batch processing
- Model inference batching
- Dynamic batch sizing
- Memory-efficient processing
- Parallel batch execution
"""

import time
import threading
import queue
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import logging
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 32
    min_batch_size: int = 1
    timeout_ms: float = 100.0
    memory_limit_mb: int = 1024
    adaptive_sizing: bool = True
    parallel_workers: int = 4


@dataclass
class ProcessingResult:
    """Result from batch processing."""
    batch_id: str
    items: List[Any]
    results: List[Any]
    processing_time_ms: float
    success: bool
    error: Optional[str] = None


class BatchProcessor:
    """Efficient batch processing for BCI-GPT operations."""
    
    def __init__(self, 
                 processing_func: Callable,
                 config: BatchConfig = None):
        """Initialize batch processor.
        
        Args:
            processing_func: Function to process batches
            config: Batch processing configuration
        """
        self.processing_func = processing_func
        self.config = config or BatchConfig()
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.batch_counter = 0
        self.is_running = False
        self.workers = []
        
        # Performance tracking
        self.performance_stats = {
            'total_batches': 0,
            'total_items': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'throughput_items_per_sec': 0.0
        }
        
        # Adaptive batch sizing
        self.current_batch_size = self.config.max_batch_size
        self.recent_times = []
        
        logger.info(f"BatchProcessor initialized with config: {self.config}")
    
    def start(self):
        """Start batch processing workers."""
        if self.is_running:
            logger.warning("BatchProcessor already running")
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.config.parallel_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.config.parallel_workers} batch processing workers")
    
    def stop(self):
        """Stop batch processing."""
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        logger.info("Batch processing stopped")
    
    def submit(self, item: Any, callback: Callable = None) -> str:
        """Submit item for batch processing.
        
        Args:
            item: Item to process
            callback: Optional callback for result
            
        Returns:
            Batch ID for tracking
        """
        batch_id = f"batch_{self.batch_counter}"
        self.batch_counter += 1
        
        self.input_queue.put({
            'batch_id': batch_id,
            'item': item,
            'callback': callback,
            'timestamp': time.time()
        })
        
        return batch_id
    
    def submit_batch(self, items: List[Any], callback: Callable = None) -> str:
        """Submit multiple items as a pre-formed batch.
        
        Args:
            items: List of items to process
            callback: Optional callback for results
            
        Returns:
            Batch ID for tracking
        """
        batch_id = f"batch_{self.batch_counter}"
        self.batch_counter += 1
        
        for item in items:
            self.input_queue.put({
                'batch_id': batch_id,
                'item': item,
                'callback': callback,
                'timestamp': time.time()
            })
        
        return batch_id
    
    def get_result(self, timeout: float = None) -> Optional[ProcessingResult]:
        """Get a processed result.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            Processing result or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker_loop(self):
        """Main worker loop for batch processing."""
        while self.is_running:
            try:
                # Collect batch
                batch = self._collect_batch()
                
                if not batch:
                    time.sleep(0.01)  # Brief pause if no work
                    continue
                
                # Process batch
                result = self._process_batch(batch)
                
                # Update performance stats
                self._update_stats(result)
                
                # Adapt batch size if enabled
                if self.config.adaptive_sizing:
                    self._adapt_batch_size(result.processing_time_ms)
                
                # Send result
                self.output_queue.put(result)
                
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                time.sleep(0.1)  # Pause on error
    
    def _collect_batch(self) -> List[Dict]:
        """Collect items into a batch."""
        batch = []
        start_time = time.time()
        timeout_sec = self.config.timeout_ms / 1000.0
        
        while (len(batch) < self.current_batch_size and 
               time.time() - start_time < timeout_sec):
            
            try:
                item = self.input_queue.get(timeout=0.01)
                batch.append(item)
                
                # If we have minimum batch size and timeout reached, process
                if (len(batch) >= self.config.min_batch_size and 
                    time.time() - start_time >= timeout_sec * 0.5):
                    break
                    
            except queue.Empty:
                # No more items immediately available
                if len(batch) >= self.config.min_batch_size:
                    break
                continue
        
        return batch
    
    def _process_batch(self, batch: List[Dict]) -> ProcessingResult:
        """Process a batch of items."""
        if not batch:
            return ProcessingResult(
                batch_id="empty",
                items=[],
                results=[],
                processing_time_ms=0.0,
                success=True
            )
        
        batch_id = batch[0]['batch_id']
        items = [item['item'] for item in batch]
        
        start_time = time.time()
        
        try:
            # Process the batch using the provided function
            if HAS_TORCH and hasattr(self.processing_func, '__self__'):
                # Handle PyTorch model inference
                results = self._torch_batch_process(items)
            else:
                # Handle generic function
                results = self.processing_func(items)
            
            processing_time = (time.time() - start_time) * 1000.0
            
            # Execute callbacks if provided
            for item, result in zip(batch, results):
                if item.get('callback'):
                    try:
                        item['callback'](result)
                    except Exception as e:
                        logger.error(f"Callback error: {str(e)}")
            
            return ProcessingResult(
                batch_id=batch_id,
                items=items,
                results=results,
                processing_time_ms=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000.0
            logger.error(f"Batch processing error: {str(e)}")
            
            return ProcessingResult(
                batch_id=batch_id,
                items=items,
                results=[None] * len(items),
                processing_time_ms=processing_time,
                success=False,
                error=str(e)
            )
    
    def _torch_batch_process(self, items: List[Any]) -> List[Any]:
        """Process batch using PyTorch optimizations."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available for batch processing")
        
        # Convert items to tensor if needed
        if isinstance(items[0], np.ndarray):
            batch_tensor = torch.from_numpy(np.stack(items))
        elif isinstance(items[0], torch.Tensor):
            batch_tensor = torch.stack(items)
        else:
            # Generic processing
            return self.processing_func(items)
        
        # Process with the function
        with torch.no_grad():
            batch_result = self.processing_func(batch_tensor)
        
        # Convert back to list
        if isinstance(batch_result, torch.Tensor):
            return list(batch_result)
        else:
            return batch_result
    
    def _update_stats(self, result: ProcessingResult):
        """Update performance statistics."""
        self.performance_stats['total_batches'] += 1
        self.performance_stats['total_items'] += len(result.items)
        
        # Update averages
        total_batches = self.performance_stats['total_batches']
        self.performance_stats['avg_batch_size'] = (
            self.performance_stats['total_items'] / total_batches
        )
        
        # Update average processing time
        prev_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (prev_avg * (total_batches - 1) + result.processing_time_ms) / total_batches
        )
        
        # Update throughput
        if result.processing_time_ms > 0:
            batch_throughput = len(result.items) / (result.processing_time_ms / 1000.0)
            self.performance_stats['throughput_items_per_sec'] = (
                (self.performance_stats['throughput_items_per_sec'] * (total_batches - 1) + 
                 batch_throughput) / total_batches
            )
    
    def _adapt_batch_size(self, processing_time_ms: float):
        """Adapt batch size based on performance."""
        self.recent_times.append(processing_time_ms)
        
        # Keep only recent times
        if len(self.recent_times) > 10:
            self.recent_times.pop(0)
        
        if len(self.recent_times) < 3:
            return
        
        avg_time = sum(self.recent_times) / len(self.recent_times)
        
        # Increase batch size if processing is fast
        if avg_time < self.config.timeout_ms * 0.5:
            self.current_batch_size = min(
                self.current_batch_size + 1,
                self.config.max_batch_size
            )
        
        # Decrease batch size if processing is slow
        elif avg_time > self.config.timeout_ms * 1.5:
            self.current_batch_size = max(
                self.current_batch_size - 1,
                self.config.min_batch_size
            )
        
        logger.debug(f"Adapted batch size to {self.current_batch_size} (avg_time: {avg_time:.1f}ms)")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        stats['current_batch_size'] = self.current_batch_size
        stats['queue_size'] = self.input_queue.qsize()
        stats['pending_results'] = self.output_queue.qsize()
        return stats


class EEGBatchProcessor(BatchProcessor):
    """Specialized batch processor for EEG data."""
    
    def __init__(self, model, config: BatchConfig = None):
        """Initialize EEG batch processor.
        
        Args:
            model: BCI-GPT model for processing
            config: Batch configuration
        """
        super().__init__(self._process_eeg_batch, config)
        self.model = model
        
        # EEG-specific configuration
        self.expected_channels = getattr(model, 'n_channels', 9)
        self.expected_sampling_rate = getattr(model, 'sampling_rate', 1000)
    
    def _process_eeg_batch(self, eeg_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of EEG data.
        
        Args:
            eeg_batch: List of EEG arrays
            
        Returns:
            List of processing results
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for EEG batch processing")
        
        # Validate and convert EEG data
        validated_batch = []
        for eeg_data in eeg_batch:
            if not self._validate_eeg_data(eeg_data):
                raise ValueError(f"Invalid EEG data shape: {eeg_data.shape}")
            validated_batch.append(eeg_data)
        
        # Stack into batch tensor
        batch_tensor = torch.from_numpy(np.stack(validated_batch)).float()
        
        # Process with model
        self.model.eval()
        with torch.no_grad():
            results = self.model(batch_tensor)
        
        # Convert results to list of dictionaries
        batch_results = []
        for i in range(len(eeg_batch)):
            result = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value[i].cpu().numpy()
                else:
                    result[key] = value
            batch_results.append(result)
        
        return batch_results
    
    def _validate_eeg_data(self, eeg_data: np.ndarray) -> bool:
        """Validate EEG data format."""
        if not isinstance(eeg_data, np.ndarray):
            return False
        
        if eeg_data.ndim != 2:
            return False
        
        channels, samples = eeg_data.shape
        if channels != self.expected_channels:
            logger.warning(f"Expected {self.expected_channels} channels, got {channels}")
            return False
        
        if samples < 100:  # Minimum reasonable sample count
            return False
        
        if not np.all(np.isfinite(eeg_data)):
            return False
        
        return True


# Convenience functions
def create_eeg_batch_processor(model, 
                              max_batch_size: int = 16,
                              timeout_ms: float = 50.0) -> EEGBatchProcessor:
    """Create optimized EEG batch processor.
    
    Args:
        model: BCI-GPT model
        max_batch_size: Maximum batch size
        timeout_ms: Batch timeout in milliseconds
        
    Returns:
        Configured EEG batch processor
    """
    config = BatchConfig(
        max_batch_size=max_batch_size,
        min_batch_size=1,
        timeout_ms=timeout_ms,
        adaptive_sizing=True,
        parallel_workers=2  # EEG processing is usually CPU/GPU bound
    )
    
    return EEGBatchProcessor(model, config)


def create_inference_batch_processor(inference_func: Callable,
                                   max_batch_size: int = 32,
                                   timeout_ms: float = 100.0) -> BatchProcessor:
    """Create general inference batch processor.
    
    Args:
        inference_func: Function for inference
        max_batch_size: Maximum batch size
        timeout_ms: Batch timeout in milliseconds
        
    Returns:
        Configured batch processor
    """
    config = BatchConfig(
        max_batch_size=max_batch_size,
        min_batch_size=1,
        timeout_ms=timeout_ms,
        adaptive_sizing=True,
        parallel_workers=4
    )
    
    return BatchProcessor(inference_func, config)