"""Minimal distributed processing for BCI-GPT."""

import threading
import time
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime

from ..utils.logging_config import get_logger


class MinimalDistributedProcessor:
    """Minimal distributed processing system."""
    
    def __init__(self):
        """Initialize minimal distributed processor."""
        self.logger = get_logger(__name__)
        self.processing_active = False
        self.task_queue = deque()
        self.completed_tasks = {}
        self.worker_threads = []
        
    def start_processing(self) -> None:
        """Start processing."""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start worker thread
        worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_threads.append(worker_thread)
        worker_thread.start()
        
        self.logger.info("Minimal distributed processing started")
    
    def stop_processing(self) -> None:
        """Stop processing."""
        self.processing_active = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.logger.info("Minimal distributed processing stopped")
    
    def submit_task(self, task_type: str, payload: Dict[str, Any]) -> str:
        """Submit a task for processing."""
        task_id = f"{task_type}_{int(time.time())}"
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "payload": payload,
            "created_at": datetime.now()
        }
        
        self.task_queue.append(task)
        self.logger.info(f"Task submitted: {task_id}")
        return task_id
    
    def _worker_loop(self) -> None:
        """Worker processing loop."""
        while self.processing_active:
            try:
                if self.task_queue:
                    task = self.task_queue.popleft()
                    result = self._execute_task(task)
                    self.completed_tasks[task["task_id"]] = result
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                time.sleep(1.0)
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task."""
        self.logger.info(f"Executing task: {task['task_id']}")
        
        # Simulate task execution
        time.sleep(0.1)
        
        return {
            "task_id": task["task_id"],
            "result": f"processed_{task['task_type']}",
            "completed_at": datetime.now(),
            "status": "completed"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "processing_active": self.processing_active,
            "queue_size": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "worker_threads": len(self.worker_threads)
        }