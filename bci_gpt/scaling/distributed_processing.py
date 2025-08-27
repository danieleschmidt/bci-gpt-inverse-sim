#!/usr/bin/env python3
"""
Distributed Processing System for BCI-GPT
Generation 3: High-throughput distributed neural signal processing
"""

import asyncio
import json
import logging
import time
import threading
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import hashlib
import pickle

@dataclass
class ProcessingTask:
    """Task for distributed processing."""
    task_id: str
    task_type: str
    data: Any
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

@dataclass
class WorkerNode:
    """Distributed worker node information."""
    worker_id: str
    host: str
    port: int
    capabilities: List[str]
    max_concurrent_tasks: int = 4
    current_tasks: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    total_tasks_completed: int = 0
    average_task_duration: float = 0.0
    is_healthy: bool = True

class TaskQueue:
    """Priority-based task queue for distributed processing."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queue = Queue(maxsize=max_size)
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit task to queue."""
        with self.lock:
            try:
                self.queue.put(task, timeout=1)
                self.pending_tasks[task.task_id] = task
                self.logger.debug(f"Task submitted: {task.task_id}")
                return task.task_id
            except:
                raise Exception("Task queue is full")
    
    def get_task(self, worker_id: str, timeout: float = 1.0) -> Optional[ProcessingTask]:
        """Get next task for worker."""
        try:
            task = self.queue.get(timeout=timeout)
            with self.lock:
                task.started_at = datetime.now()
                task.worker_id = worker_id
                if task.task_id in self.pending_tasks:
                    del self.pending_tasks[task.task_id]
            return task
        except Empty:
            return None
    
    def complete_task(self, task: ProcessingTask):
        """Mark task as completed."""
        with self.lock:
            task.completed_at = datetime.now()
            self.completed_tasks[task.task_id] = task
    
    def fail_task(self, task: ProcessingTask, error: str):
        """Mark task as failed."""
        with self.lock:
            task.error = error
            task.completed_at = datetime.now()
            self.failed_tasks[task.task_id] = task
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task status."""
        with self.lock:
            if task_id in self.pending_tasks:
                return self.pending_tasks[task_id]
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "pending": len(self.pending_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "queue_size": self.queue.qsize()
            }

class EEGProcessor:
    """Distributed EEG signal processor."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_eeg(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess EEG signal."""
        try:
            # Simulate EEG preprocessing
            signal = eeg_data.get("data", [])
            sampling_rate = eeg_data.get("sampling_rate", 1000)
            
            # Mock preprocessing steps
            processed_signal = []
            if isinstance(signal, list) and len(signal) > 0:
                # Simulate filtering, artifact removal, etc.
                for sample in signal:
                    # Simple high-pass filter simulation
                    processed_sample = sample * 0.9 + 0.1 * (sample if sample > 0.1 else 0)
                    processed_signal.append(processed_sample)
            
            result = {
                "preprocessed_data": processed_signal,
                "sampling_rate": sampling_rate,
                "processing_time": time.time(),
                "features_extracted": len(processed_signal),
                "quality_score": min(1.0, len(processed_signal) / 1000.0)
            }
            
            # Simulate processing time
            time.sleep(0.1)
            
            return result
            
        except Exception as e:
            raise Exception(f"EEG preprocessing failed: {e}")
    
    def extract_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from preprocessed EEG."""
        try:
            signal = preprocessed_data.get("preprocessed_data", [])
            
            if not signal:
                return {"features": [], "feature_names": []}
            
            # Mock feature extraction
            features = {
                "spectral_power_alpha": sum(abs(x) for x in signal[::10]) / len(signal),
                "spectral_power_beta": sum(abs(x) for x in signal[::5]) / len(signal),
                "temporal_complexity": len(set(signal[:100])) / 100.0 if len(signal) >= 100 else 0,
                "signal_variance": sum((x - sum(signal)/len(signal))**2 for x in signal) / len(signal),
                "peak_frequency": 10.0 + (sum(signal) % 20),  # Mock peak frequency
            }
            
            # Simulate feature extraction time
            time.sleep(0.05)
            
            return {
                "features": list(features.values()),
                "feature_names": list(features.keys()),
                "extraction_time": time.time(),
                "feature_quality": min(1.0, len(features) / 5.0)
            }
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {e}")
    
    def predict_text(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict text from EEG features."""
        try:
            feature_vector = features.get("features", [])
            
            if not feature_vector:
                return {"predicted_text": "", "confidence": 0.0}
            
            # Mock text prediction
            # In production, would use actual BCI-GPT model
            mock_vocabulary = [
                "hello", "world", "yes", "no", "help", "stop", 
                "more", "please", "thank", "you", "good", "morning"
            ]
            
            # Simple prediction based on features
            feature_sum = sum(abs(f) for f in feature_vector)
            word_index = int(feature_sum * 100) % len(mock_vocabulary)
            predicted_word = mock_vocabulary[word_index]
            
            # Confidence based on feature quality
            confidence = min(1.0, feature_sum / len(feature_vector)) if feature_vector else 0.0
            confidence = max(0.1, min(0.95, confidence))
            
            # Simulate prediction time
            time.sleep(0.02)
            
            return {
                "predicted_text": predicted_word,
                "confidence": confidence,
                "prediction_time": time.time(),
                "token_probabilities": {word: confidence * 0.8 for word in mock_vocabulary[:3]}
            }
            
        except Exception as e:
            raise Exception(f"Text prediction failed: {e}")

class DistributedWorker:
    """Distributed processing worker."""
    
    def __init__(self, 
                 worker_id: str,
                 max_concurrent_tasks: int = 4,
                 supported_tasks: List[str] = None):
        
        self.worker_id = worker_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.supported_tasks = supported_tasks or ["preprocess_eeg", "extract_features", "predict_text"]
        
        self.is_running = False
        self.current_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.eeg_processor = EEGProcessor()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        self.logger = logging.getLogger(__name__)
    
    def start(self, task_queue: TaskQueue):
        """Start worker processing."""
        self.is_running = True
        self.task_queue = task_queue
        
        self.logger.info(f"Worker {self.worker_id} started")
        
        # Start worker loop
        worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        worker_thread.start()
    
    def stop(self):
        """Stop worker processing."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self):
        """Main worker processing loop."""
        while self.is_running:
            try:
                if self.current_tasks < self.max_concurrent_tasks:
                    task = self.task_queue.get_task(self.worker_id, timeout=1.0)
                    
                    if task and task.task_type in self.supported_tasks:
                        # Submit task to executor
                        future = self.executor.submit(self._process_task, task)
                        self.current_tasks += 1
                        
                        # Handle completion asynchronously
                        future.add_done_callback(lambda f: self._task_completed(f))
                else:
                    time.sleep(0.1)  # Brief pause when at capacity
                
            except Exception as e:
                self.logger.error(f"Worker loop error: {e}")
                time.sleep(1)
    
    def _process_task(self, task: ProcessingTask) -> ProcessingTask:
        """Process individual task."""
        try:
            self.logger.debug(f"Processing task: {task.task_id}")
            
            if task.task_type == "preprocess_eeg":
                result = self.eeg_processor.preprocess_eeg(task.data)
            elif task.task_type == "extract_features":
                result = self.eeg_processor.extract_features(task.data)
            elif task.task_type == "predict_text":
                result = self.eeg_processor.predict_text(task.data)
            else:
                raise Exception(f"Unsupported task type: {task.task_type}")
            
            task.result = result
            self.task_queue.complete_task(task)
            self.completed_tasks += 1
            
            return task
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Task {task.task_id} failed: {error_msg}")
            self.task_queue.fail_task(task, error_msg)
            self.failed_tasks += 1
            return task
    
    def _task_completed(self, future):
        """Handle task completion."""
        self.current_tasks -= 1
        try:
            task = future.result()
            self.logger.debug(f"Task completed: {task.task_id}")
        except Exception as e:
            self.logger.error(f"Task completion error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "current_tasks": self.current_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "supported_tasks": self.supported_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks
        }

class DistributedOrchestrator:
    """Orchestrates distributed BCI processing across multiple workers."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.task_queue = TaskQueue(max_queue_size)
        self.workers = []
        self.processing_pipelines = {}
        
        self.logger = logging.getLogger(__name__)
    
    def add_worker(self, 
                   worker_id: str,
                   max_concurrent_tasks: int = 4,
                   supported_tasks: List[str] = None) -> DistributedWorker:
        """Add worker to the processing cluster."""
        
        worker = DistributedWorker(
            worker_id=worker_id,
            max_concurrent_tasks=max_concurrent_tasks,
            supported_tasks=supported_tasks
        )
        
        self.workers.append(worker)
        worker.start(self.task_queue)
        
        self.logger.info(f"Added worker: {worker_id}")
        return worker
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from cluster."""
        for worker in self.workers:
            if worker.worker_id == worker_id:
                worker.stop()
                self.workers.remove(worker)
                self.logger.info(f"Removed worker: {worker_id}")
                return True
        return False
    
    def process_eeg_pipeline(self, 
                            eeg_data: Dict[str, Any],
                            pipeline_id: Optional[str] = None) -> str:
        """Process EEG data through complete pipeline."""
        
        if not pipeline_id:
            pipeline_id = hashlib.md5(json.dumps(eeg_data, sort_keys=True).encode()).hexdigest()[:8]
        
        # Create pipeline tasks
        tasks = []
        
        # Step 1: Preprocessing
        preprocess_task = ProcessingTask(
            task_id=f"{pipeline_id}_preprocess",
            task_type="preprocess_eeg",
            data=eeg_data,
            priority=1
        )
        tasks.append(preprocess_task)
        
        # Submit initial task
        self.task_queue.submit_task(preprocess_task)
        
        # Store pipeline info
        self.processing_pipelines[pipeline_id] = {
            "created_at": datetime.now(),
            "tasks": [preprocess_task.task_id],
            "status": "processing",
            "final_result": None
        }
        
        self.logger.info(f"Started EEG pipeline: {pipeline_id}")
        return pipeline_id
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get status of processing pipeline."""
        
        if pipeline_id not in self.processing_pipelines:
            return None
        
        pipeline = self.processing_pipelines[pipeline_id]
        task_statuses = {}
        
        for task_id in pipeline["tasks"]:
            task = self.task_queue.get_task_status(task_id)
            if task:
                task_statuses[task_id] = {
                    "status": "completed" if task.completed_at else "failed" if task.error else "processing",
                    "duration": task.duration(),
                    "worker_id": task.worker_id,
                    "error": task.error
                }
        
        # Check if pipeline is complete
        all_tasks_done = all(
            task_statuses.get(task_id, {}).get("status") in ["completed", "failed"]
            for task_id in pipeline["tasks"]
        )
        
        if all_tasks_done and pipeline["status"] == "processing":
            pipeline["status"] = "completed"
            # Get final result from last task
            final_task_id = pipeline["tasks"][-1]
            final_task = self.task_queue.get_task_status(final_task_id)
            if final_task and final_task.result:
                pipeline["final_result"] = final_task.result
        
        return {
            "pipeline_id": pipeline_id,
            "status": pipeline["status"],
            "created_at": pipeline["created_at"].isoformat(),
            "task_count": len(pipeline["tasks"]),
            "task_statuses": task_statuses,
            "final_result": pipeline.get("final_result")
        }
    
    def process_eeg_batch(self, 
                         eeg_batch: List[Dict[str, Any]],
                         batch_id: Optional[str] = None) -> str:
        """Process batch of EEG signals."""
        
        if not batch_id:
            batch_id = f"batch_{int(time.time())}"
        
        pipeline_ids = []
        
        for i, eeg_data in enumerate(eeg_batch):
            pipeline_id = f"{batch_id}_{i}"
            self.process_eeg_pipeline(eeg_data, pipeline_id)
            pipeline_ids.append(pipeline_id)
        
        self.logger.info(f"Started batch processing: {batch_id} ({len(pipeline_ids)} pipelines)")
        return batch_id
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get distributed cluster statistics."""
        
        worker_stats = [worker.get_stats() for worker in self.workers]
        queue_stats = self.task_queue.get_stats()
        
        return {
            "total_workers": len(self.workers),
            "active_workers": sum(1 for w in worker_stats if w["is_running"]),
            "total_current_tasks": sum(w["current_tasks"] for w in worker_stats),
            "total_completed_tasks": sum(w["completed_tasks"] for w in worker_stats),
            "total_failed_tasks": sum(w["failed_tasks"] for w in worker_stats),
            "queue_stats": queue_stats,
            "active_pipelines": len([p for p in self.processing_pipelines.values() if p["status"] == "processing"]),
            "total_pipelines": len(self.processing_pipelines),
            "worker_details": worker_stats
        }

# Example usage and testing
if __name__ == "__main__":
    print("⚡ Testing Distributed Processing System...")
    
    # Create orchestrator
    orchestrator = DistributedOrchestrator()
    
    # Add workers
    orchestrator.add_worker("worker_1", max_concurrent_tasks=2)
    orchestrator.add_worker("worker_2", max_concurrent_tasks=2)
    orchestrator.add_worker("worker_3", max_concurrent_tasks=2)
    
    print(f"✅ Added {len(orchestrator.workers)} workers")
    
    # Test single EEG processing
    sample_eeg = {
        "data": [i * 0.1 for i in range(1000)],
        "sampling_rate": 1000,
        "channels": ["Fz", "Cz", "Pz"],
        "subject_id": "test_001"
    }
    
    pipeline_id = orchestrator.process_eeg_pipeline(sample_eeg)
    print(f"✅ Started pipeline: {pipeline_id}")
    
    # Test batch processing
    eeg_batch = [
        {
            "data": [i * 0.1 + j for i in range(500)],
            "sampling_rate": 1000,
            "subject_id": f"test_{j:03d}"
        }
        for j in range(3)
    ]
    
    batch_id = orchestrator.process_eeg_batch(eeg_batch)
    print(f"✅ Started batch: {batch_id}")
    
    # Monitor processing
    print("⏳ Monitoring processing...")
    for _ in range(10):
        time.sleep(1)
        
        cluster_stats = orchestrator.get_cluster_stats()
        print(f"📊 Cluster: {cluster_stats['total_current_tasks']} active, {cluster_stats['total_completed_tasks']} completed")
        
        # Check pipeline status
        pipeline_status = orchestrator.get_pipeline_status(pipeline_id)
        if pipeline_status:
            print(f"🔄 Pipeline {pipeline_id}: {pipeline_status['status']}")
            
            if pipeline_status['status'] == 'completed':
                final_result = pipeline_status.get('final_result')
                if final_result:
                    print(f"✅ Final result: {final_result.get('predicted_text', 'N/A')}")
                break
    
    # Final cluster stats
    final_stats = orchestrator.get_cluster_stats()
    print(f"\n📊 Final Stats:")
    print(f"   Workers: {final_stats['active_workers']}/{final_stats['total_workers']}")
    print(f"   Completed: {final_stats['total_completed_tasks']}")
    print(f"   Failed: {final_stats['total_failed_tasks']}")
    print(f"   Pipelines: {final_stats['total_pipelines']}")
    
    # Cleanup
    for worker in orchestrator.workers[:]:
        orchestrator.remove_worker(worker.worker_id)
    
    print("\n🚀 Distributed Processing System Ready!")
