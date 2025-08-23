"""
Fault Tolerance Module for BCI-GPT System

Provides fault-tolerant processing capabilities for critical
BCI operations with automatic recovery and graceful degradation.
"""

import logging
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of faults that can occur"""
    HARDWARE_FAULT = "hardware_fault"
    SOFTWARE_FAULT = "software_fault"
    NETWORK_FAULT = "network_fault"
    DATA_FAULT = "data_fault"
    TIMEOUT_FAULT = "timeout_fault"
    RESOURCE_FAULT = "resource_fault"


@dataclass
class FaultRecord:
    """Record of a fault occurrence"""
    fault_type: FaultType
    component: str
    error_message: str
    timestamp: float = field(default_factory=time.time)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'fault_type': self.fault_type.value,
            'component': self.component,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'stack_trace': self.stack_trace,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }


class FaultTolerantProcessor:
    """
    Fault-tolerant processor for BCI operations
    
    Provides automatic fault detection, recovery, and graceful
    degradation for critical BCI processing components.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.fault_history: List[FaultRecord] = []
        self.recovery_handlers: Dict[FaultType, List[Callable]] = {}
        self._lock = threading.Lock()
        self._is_healthy = True
        
        logger.info(f"Fault tolerant processor '{name}' initialized")
    
    def add_recovery_handler(
        self, 
        fault_type: FaultType, 
        handler: Callable[[], bool]
    ) -> None:
        """Add recovery handler for specific fault type"""
        if fault_type not in self.recovery_handlers:
            self.recovery_handlers[fault_type] = []
        self.recovery_handlers[fault_type].append(handler)
        
        logger.info(f"Added recovery handler for {fault_type.value}")
    
    def process_with_fault_tolerance(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with fault tolerance"""
        try:
            # Check if processor is healthy
            if not self._is_healthy:
                self._attempt_recovery()
            
            # Execute the operation
            result = operation(*args, **kwargs)
            
            # Mark as healthy on success
            self._is_healthy = True
            return result
            
        except Exception as e:
            fault_type = self._classify_fault(e)
            fault_record = FaultRecord(
                fault_type=fault_type,
                component=self.name,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
            with self._lock:
                self.fault_history.append(fault_record)
            
            logger.error(f"Fault detected in {self.name}: {e}")
            
            # Attempt recovery
            if self._attempt_fault_recovery(fault_type, fault_record):
                # Retry operation after successful recovery
                try:
                    result = operation(*args, **kwargs)
                    self._is_healthy = True
                    fault_record.recovery_successful = True
                    logger.info(f"Recovery successful for {self.name}")
                    return result
                except Exception as retry_error:
                    logger.error(f"Retry failed after recovery: {retry_error}")
                    self._is_healthy = False
                    raise
            else:
                self._is_healthy = False
                raise
    
    def _classify_fault(self, exception: Exception) -> FaultType:
        """Classify fault type based on exception"""
        error_type = type(exception).__name__
        error_message = str(exception).lower()
        
        if 'timeout' in error_message or error_type == 'TimeoutError':
            return FaultType.TIMEOUT_FAULT
        elif 'connection' in error_message or 'network' in error_message:
            return FaultType.NETWORK_FAULT
        elif 'memory' in error_message or 'resource' in error_message:
            return FaultType.RESOURCE_FAULT
        elif 'data' in error_message or 'value' in error_message:
            return FaultType.DATA_FAULT
        elif 'hardware' in error_message or 'device' in error_message:
            return FaultType.HARDWARE_FAULT
        else:
            return FaultType.SOFTWARE_FAULT
    
    def _attempt_fault_recovery(
        self, 
        fault_type: FaultType, 
        fault_record: FaultRecord
    ) -> bool:
        """Attempt recovery from specific fault type"""
        fault_record.recovery_attempted = True
        
        if fault_type not in self.recovery_handlers:
            logger.warning(f"No recovery handler for {fault_type.value}")
            return False
        
        for handler in self.recovery_handlers[fault_type]:
            try:
                logger.info(f"Attempting recovery for {fault_type.value}")
                success = handler()
                if success:
                    logger.info(f"Recovery handler succeeded for {fault_type.value}")
                    return True
                else:
                    logger.warning(f"Recovery handler failed for {fault_type.value}")
            except Exception as e:
                logger.error(f"Recovery handler error: {e}")
        
        return False
    
    def _attempt_recovery(self) -> bool:
        """General recovery attempt"""
        logger.info(f"Attempting general recovery for {self.name}")
        
        # Try all available recovery handlers
        for fault_type, handlers in self.recovery_handlers.items():
            for handler in handlers:
                try:
                    if handler():
                        self._is_healthy = True
                        logger.info(f"General recovery successful")
                        return True
                except Exception as e:
                    logger.error(f"General recovery handler error: {e}")
        
        return False
    
    def is_healthy(self) -> bool:
        """Check if processor is healthy"""
        return self._is_healthy
    
    def get_fault_history(self) -> List[Dict[str, Any]]:
        """Get fault history"""
        with self._lock:
            return [fault.to_dict() for fault in self.fault_history]
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """Get fault statistics"""
        with self._lock:
            if not self.fault_history:
                return {
                    'total_faults': 0,
                    'fault_types': {},
                    'recovery_rate': 0.0,
                    'average_faults_per_hour': 0.0
                }
            
            fault_type_counts = {}
            successful_recoveries = 0
            
            for fault in self.fault_history:
                fault_type = fault.fault_type.value
                fault_type_counts[fault_type] = fault_type_counts.get(fault_type, 0) + 1
                
                if fault.recovery_successful:
                    successful_recoveries += 1
            
            # Calculate time window for rate calculation
            oldest_fault = min(fault.timestamp for fault in self.fault_history)
            time_window_hours = (time.time() - oldest_fault) / 3600
            
            return {
                'total_faults': len(self.fault_history),
                'fault_types': fault_type_counts,
                'recovery_rate': successful_recoveries / len(self.fault_history),
                'average_faults_per_hour': len(self.fault_history) / max(1, time_window_hours),
                'is_healthy': self._is_healthy
            }
    
    def reset_fault_history(self) -> None:
        """Reset fault history"""
        with self._lock:
            self.fault_history.clear()
            self._is_healthy = True
        logger.info(f"Fault history reset for {self.name}")


def fault_tolerant(name: str):
    """Decorator for fault-tolerant function execution"""
    processor = FaultTolerantProcessor(name)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return processor.process_with_fault_tolerance(func, *args, **kwargs)
        
        # Attach processor to wrapper for access to fault tolerance features
        wrapper.fault_processor = processor
        return wrapper
    
    return decorator


# Default recovery handlers
def default_network_recovery() -> bool:
    """Default network recovery handler"""
    logger.info("Attempting network recovery")
    time.sleep(1.0)  # Simple wait for network recovery
    return True


def default_resource_recovery() -> bool:
    """Default resource recovery handler"""
    logger.info("Attempting resource cleanup")
    # In a real implementation, this would clean up resources
    return True


def default_timeout_recovery() -> bool:
    """Default timeout recovery handler"""
    logger.info("Attempting timeout recovery")
    time.sleep(0.5)  # Brief pause before retry
    return True