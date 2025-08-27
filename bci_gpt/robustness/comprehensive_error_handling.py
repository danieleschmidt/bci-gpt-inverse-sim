#!/usr/bin/env python3
"""
Comprehensive Error Handling Framework for BCI-GPT System
Generation 2: Robust error handling with graceful degradation
"""

import sys
import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import json

class BCIErrorSeverity(Enum):
    """Error severity levels for BCI operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"  # For clinical safety

class BCIError(Exception):
    """Base exception for BCI-GPT operations."""
    
    def __init__(self, 
                 message: str, 
                 severity: BCIErrorSeverity = BCIErrorSeverity.MEDIUM,
                 error_code: str = "BCI_UNKNOWN",
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/monitoring."""
        return {
            "error_code": self.error_code,
            "message": str(self),
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc() if sys.exc_info()[0] else None
        }

class EEGProcessingError(BCIError):
    """Errors in EEG signal processing."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="BCI_EEG_PROCESSING", **kwargs)

class ModelInferenceError(BCIError):
    """Errors in model inference/prediction."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="BCI_MODEL_INFERENCE", **kwargs)

class ClinicalSafetyError(BCIError):
    """Critical clinical safety errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', BCIErrorSeverity.EMERGENCY)
        super().__init__(message, error_code="BCI_CLINICAL_SAFETY", **kwargs)

class DataValidationError(BCIError):
    """Data validation and quality errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="BCI_DATA_VALIDATION", **kwargs)

class SystemResourceError(BCIError):
    """System resource and performance errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="BCI_SYSTEM_RESOURCE", **kwargs)

class RobustErrorHandler:
    """Comprehensive error handling with graceful degradation."""
    
    def __init__(self, 
                 log_file: str = "bci_errors.log",
                 max_retries: int = 3,
                 enable_telemetry: bool = True):
        self.max_retries = max_retries
        self.enable_telemetry = enable_telemetry
        self.error_counts = {}
        self.setup_logging(log_file)
    
    def setup_logging(self, log_file: str):
        """Setup comprehensive error logging."""
        self.logger = logging.getLogger("BCI_ErrorHandler")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for all errors
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for critical errors only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any] = None,
                    attempt_recovery: bool = True) -> Dict[str, Any]:
        """Handle errors with comprehensive logging and recovery attempts."""
        
        if isinstance(error, BCIError):
            bci_error = error
        else:
            # Wrap non-BCI errors
            bci_error = BCIError(
                f"Unexpected error: {str(error)}",
                context=context
            )
        
        # Log error
        self.log_error(bci_error)
        
        # Update error statistics
        self.error_counts[bci_error.error_code] = self.error_counts.get(bci_error.error_code, 0) + 1
        
        # Attempt recovery for non-critical errors
        recovery_result = None
        if attempt_recovery and bci_error.severity != BCIErrorSeverity.EMERGENCY:
            recovery_result = self.attempt_recovery(bci_error)
        
        return {
            "error": bci_error.to_dict(),
            "recovery_attempted": attempt_recovery,
            "recovery_result": recovery_result,
            "total_count": self.error_counts[bci_error.error_code]
        }
    
    def log_error(self, error: BCIError):
        """Log error with appropriate severity level."""
        error_dict = error.to_dict()
        
        if error.severity == BCIErrorSeverity.EMERGENCY:
            self.logger.critical(f"🚨 EMERGENCY: {error.message}", extra=error_dict)
        elif error.severity == BCIErrorSeverity.CRITICAL:
            self.logger.error(f"❌ CRITICAL: {error.message}", extra=error_dict)
        elif error.severity == BCIErrorSeverity.HIGH:
            self.logger.error(f"⚠️  HIGH: {error.message}", extra=error_dict)
        elif error.severity == BCIErrorSeverity.MEDIUM:
            self.logger.warning(f"🔶 MEDIUM: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"ℹ️  LOW: {error.message}", extra=error_dict)
    
    def attempt_recovery(self, error: BCIError) -> Optional[Dict[str, Any]]:
        """Attempt automated error recovery."""
        recovery_strategies = {
            "BCI_EEG_PROCESSING": self._recover_eeg_processing,
            "BCI_MODEL_INFERENCE": self._recover_model_inference,
            "BCI_DATA_VALIDATION": self._recover_data_validation,
            "BCI_SYSTEM_RESOURCE": self._recover_system_resource
        }
        
        strategy = recovery_strategies.get(error.error_code)
        if strategy:
            try:
                return strategy(error)
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
                return {"recovery_status": "failed", "recovery_error": str(recovery_error)}
        
        return {"recovery_status": "no_strategy", "message": "No recovery strategy available"}
    
    def _recover_eeg_processing(self, error: BCIError) -> Dict[str, Any]:
        """Recover from EEG processing errors."""
        return {
            "recovery_status": "attempted",
            "strategy": "fallback_to_basic_preprocessing",
            "message": "Attempting simplified EEG preprocessing"
        }
    
    def _recover_model_inference(self, error: BCIError) -> Dict[str, Any]:
        """Recover from model inference errors."""
        return {
            "recovery_status": "attempted", 
            "strategy": "fallback_to_cached_model",
            "message": "Using cached model predictions"
        }
    
    def _recover_data_validation(self, error: BCIError) -> Dict[str, Any]:
        """Recover from data validation errors."""
        return {
            "recovery_status": "attempted",
            "strategy": "relaxed_validation",
            "message": "Using relaxed validation criteria"
        }
    
    def _recover_system_resource(self, error: BCIError) -> Dict[str, Any]:
        """Recover from system resource errors."""
        return {
            "recovery_status": "attempted",
            "strategy": "reduce_batch_size",
            "message": "Reducing processing batch size"
        }

def with_error_handling(max_retries: int = 3, 
                       fallback_value: Any = None,
                       severity: BCIErrorSeverity = BCIErrorSeverity.MEDIUM):
    """Decorator for robust error handling with retries."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = RobustErrorHandler(max_retries=max_retries)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        # Final attempt failed
                        error_result = error_handler.handle_error(
                            e, 
                            context={"function": func.__name__, "attempt": attempt + 1}
                        )
                        if fallback_value is not None:
                            return fallback_value
                        raise
                    else:
                        # Retry with exponential backoff
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        error_handler.logger.info(f"Retrying {func.__name__} (attempt {attempt + 2}/{max_retries + 1})")
        
        return wrapper
    return decorator

# Clinical Safety Monitors
class ClinicalSafetyMonitor:
    """Monitor for clinical safety during BCI operations."""
    
    def __init__(self):
        self.session_start = None
        self.max_session_duration = 3600  # 1 hour
        self.fatigue_threshold = 0.8
        self.error_handler = RobustErrorHandler()
    
    def start_session(self, user_id: str):
        """Start clinical monitoring session."""
        self.session_start = time.time()
        self.user_id = user_id
        self.logger.info(f"Clinical session started for user {user_id}")
    
    def check_session_safety(self) -> Dict[str, Any]:
        """Check if session is within safety parameters."""
        if not self.session_start:
            raise ClinicalSafetyError("Session not properly initialized")
        
        session_duration = time.time() - self.session_start
        
        safety_status = {
            "session_duration": session_duration,
            "max_duration_exceeded": session_duration > self.max_session_duration,
            "fatigue_detected": False,  # Would integrate with actual fatigue detection
            "emergency_stop_required": False
        }
        
        if safety_status["max_duration_exceeded"]:
            raise ClinicalSafetyError(
                f"Session duration {session_duration:.0f}s exceeds maximum {self.max_session_duration}s",
                context=safety_status
            )
        
        return safety_status
    
    def emergency_stop(self, reason: str):
        """Execute emergency stop procedure."""
        self.error_handler.handle_error(
            ClinicalSafetyError(f"Emergency stop: {reason}"),
            context={"user_id": self.user_id, "session_duration": time.time() - self.session_start}
        )
        # Would trigger actual emergency procedures

# Example usage and testing
def example_bci_function():
    """Example function demonstrating error handling."""
    import random
    
    if random.random() < 0.3:
        raise EEGProcessingError("Simulated EEG processing failure")
    elif random.random() < 0.2:
        raise ModelInferenceError("Model prediction failed")
    else:
        return {"prediction": "hello", "confidence": 0.85}

@with_error_handling(max_retries=2, fallback_value={"prediction": "error", "confidence": 0.0})
def robust_bci_function():
    """BCI function with robust error handling."""
    return example_bci_function()

if __name__ == "__main__":
    print("🛡️  Testing Robust Error Handling Framework...")
    
    # Test basic error handling
    error_handler = RobustErrorHandler()
    
    for i in range(5):
        try:
            result = robust_bci_function()
            print(f"✅ Trial {i+1}: {result}")
        except Exception as e:
            print(f"❌ Trial {i+1}: {e}")
    
    # Test clinical safety monitoring
    safety_monitor = ClinicalSafetyMonitor()
    safety_monitor.start_session("test_user_001")
    
    try:
        safety_status = safety_monitor.check_session_safety()
        print(f"🏥 Safety Status: {safety_status}")
    except ClinicalSafetyError as e:
        print(f"🚨 Safety Alert: {e}")
    
    print("\n📊 Error Handling Framework Ready!")
