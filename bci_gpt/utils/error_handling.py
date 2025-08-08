"""Comprehensive error handling and validation for BCI-GPT."""

import functools
import traceback
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from .logging_config import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BCIError:
    """Structured error information."""
    error_type: str
    message: str
    severity: ErrorSeverity
    component: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    suggestion: Optional[str] = None
    error_code: Optional[str] = None


class BCIGPTException(Exception):
    """Base exception class for BCI-GPT."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.severity = severity
        self.timestamp = None
        
        # Auto-generate timestamp
        import time
        self.timestamp = time.time()


class ModelLoadError(BCIGPTException):
    """Exception raised when model loading fails."""
    
    def __init__(self, message: str, model_path: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_LOAD_ERROR", **kwargs)
        if model_path:
            self.details["model_path"] = model_path


class DataValidationError(BCIGPTException):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, data_shape: tuple = None, expected_shape: tuple = None, **kwargs):
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        if data_shape:
            self.details["data_shape"] = data_shape
        if expected_shape:
            self.details["expected_shape"] = expected_shape


class ProcessingError(BCIGPTException):
    """Exception raised during signal processing."""
    
    def __init__(self, message: str, processing_stage: str = None, **kwargs):
        super().__init__(message, error_code="PROCESSING_ERROR", **kwargs)
        if processing_stage:
            self.details["processing_stage"] = processing_stage


class ConfigurationError(BCIGPTException):
    """Exception raised for configuration issues."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        if config_key:
            self.details["config_key"] = config_key


class StreamingError(BCIGPTException):
    """Exception raised during real-time streaming."""
    
    def __init__(self, message: str, stream_type: str = None, **kwargs):
        super().__init__(message, error_code="STREAMING_ERROR", **kwargs)
        if stream_type:
            self.details["stream_type"] = stream_type


class InferenceError(BCIGPTException):
    """Exception raised during model inference."""
    
    def __init__(self, message: str, model_type: str = None, **kwargs):
        super().__init__(message, error_code="INFERENCE_ERROR", **kwargs)
        if model_type:
            self.details["model_type"] = model_type


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.error_handlers = {}
        self.recovery_strategies = {}
        self.error_count = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default error handlers."""
        
        @self.register_handler(ModelLoadError)
        def handle_model_load_error(error: ModelLoadError, context: Dict = None):
            """Handle model loading errors."""
            self.logger.log_error(f"Model loading failed: {error.message}", error, error.details)
            
            # Try alternative loading strategies
            if context and "alternative_path" in context:
                self.logger.log_info("Attempting alternative model loading")
                return {"action": "retry", "alternative_path": context["alternative_path"]}
            
            return {"action": "fail", "suggestion": "Check model path and file integrity"}
        
        @self.register_handler(DataValidationError)
        def handle_data_validation_error(error: DataValidationError, context: Dict = None):
            """Handle data validation errors."""
            self.logger.log_error(f"Data validation failed: {error.message}", error, error.details)
            
            # Suggest data preprocessing if shapes don't match
            if "data_shape" in error.details and "expected_shape" in error.details:
                return {
                    "action": "preprocess", 
                    "suggestion": "Reshape data to match expected format"
                }
            
            return {"action": "fail", "suggestion": "Validate input data format"}
        
        @self.register_handler(ProcessingError)
        def handle_processing_error(error: ProcessingError, context: Dict = None):
            """Handle processing errors."""
            self.logger.log_error(f"Processing failed: {error.message}", error, error.details)
            
            # Try fallback processing if available
            if context and "fallback_method" in context:
                self.logger.log_info("Attempting fallback processing method")
                return {"action": "retry", "method": context["fallback_method"]}
            
            return {"action": "fail", "suggestion": "Check input data quality"}
        
        @self.register_handler(StreamingError)
        def handle_streaming_error(error: StreamingError, context: Dict = None):
            """Handle streaming errors."""
            self.logger.log_error(f"Streaming failed: {error.message}", error, error.details)
            
            # Try reconnection for streaming issues
            return {"action": "reconnect", "suggestion": "Check stream source availability"}
        
        @self.register_handler(Exception)
        def handle_generic_error(error: Exception, context: Dict = None):
            """Handle generic exceptions."""
            self.logger.log_error(f"Unexpected error: {str(error)}", error)
            
            return {"action": "fail", "suggestion": "Contact support with error details"}
    
    def register_handler(self, error_type: Type[Exception]):
        """Decorator to register error handlers."""
        def decorator(handler_func: Callable):
            self.error_handlers[error_type] = handler_func
            return handler_func
        return decorator
    
    def register_recovery_strategy(self, error_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for specific error types."""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, error: Exception, context: Dict = None) -> Dict[str, Any]:
        """Handle an error using registered handlers."""
        error_type = type(error)
        
        # Update error count
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
        
        # Find appropriate handler
        handler = None
        
        # Look for exact type match first
        if error_type in self.error_handlers:
            handler = self.error_handlers[error_type]
        else:
            # Look for parent class handlers
            for registered_type, registered_handler in self.error_handlers.items():
                if issubclass(error_type, registered_type):
                    handler = registered_handler
                    break
        
        if handler:
            try:
                return handler(error, context)
            except Exception as handler_error:
                self.logger.log_error(f"Error handler failed: {str(handler_error)}", handler_error)
                return {"action": "fail", "error": "Handler execution failed"}
        
        # Fallback to generic handling
        self.logger.log_error(f"Unhandled error type: {error_type.__name__}", error)
        return {"action": "fail", "suggestion": "Unhandled error type"}
    
    def attempt_recovery(self, error: Exception, context: Dict = None) -> bool:
        """Attempt to recover from an error."""
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_type]
                return recovery_func(error, context)
            except Exception as recovery_error:
                self.logger.log_error(f"Recovery attempt failed: {str(recovery_error)}", recovery_error)
                return False
        
        return False
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return self.error_count.copy()


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    
    return _global_error_handler


def robust_function(retry_count: int = 3, 
                   exceptions: tuple = (Exception,),
                   delay: float = 1.0,
                   backoff_multiplier: float = 2.0):
    """Decorator to make functions robust with retry logic."""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            error_handler = get_error_handler()
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(retry_count):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Log the attempt
                    logger.log_warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{retry_count}: {str(e)}"
                    )
                    
                    # Handle the error
                    error_result = error_handler.handle_error(e, {"attempt": attempt + 1, "function": func.__name__})
                    
                    # Check if we should retry
                    if error_result.get("action") == "fail" or attempt == retry_count - 1:
                        break
                    
                    # Attempt recovery if available
                    if error_handler.attempt_recovery(e, {"attempt": attempt + 1}):
                        logger.log_info(f"Recovery successful for {func.__name__}")
                        continue
                    
                    # Wait before retry with exponential backoff
                    if attempt < retry_count - 1:
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_multiplier
            
            # If we get here, all retries failed
            logger.log_error(f"Function {func.__name__} failed after {retry_count} attempts", last_exception)
            raise last_exception
        
        return wrapper
    return decorator


def validate_input(validation_rules: Dict[str, Callable]):
    """Decorator for input validation."""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, validator in validation_rules.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    try:
                        if not validator(value):
                            raise DataValidationError(
                                f"Validation failed for parameter '{param_name}'",
                                details={"parameter": param_name, "value": str(value)[:100]}
                            )
                    except Exception as e:
                        if isinstance(e, DataValidationError):
                            raise
                        else:
                            raise DataValidationError(
                                f"Validation error for parameter '{param_name}': {str(e)}",
                                details={"parameter": param_name, "validator_error": str(e)}
                            )
            
            # Call the function if validation passes
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log function execution errors
                logger.log_error(f"Function {func.__name__} execution failed", e)
                raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                default_return: Any = None,
                log_errors: bool = True,
                raise_on_error: bool = False) -> Any:
    """Safely execute a function with error handling."""
    
    logger = get_logger()
    error_handler = get_error_handler()
    
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.log_error(f"Safe execution failed for {func.__name__ if hasattr(func, '__name__') else 'function'}", e)
        
        # Handle the error
        error_handler.handle_error(e, {"safe_execution": True})
        
        if raise_on_error:
            raise
        
        return default_return


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exceptions: tuple = (Exception,)):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds to wait before trying again
            expected_exceptions: Exception types that trigger the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exceptions = expected_exceptions
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = get_logger()
    
    def __call__(self, func: Callable):
        """Decorator implementation."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        import time
        
        current_time = time.time()
        
        # Check if circuit should be closed again
        if self.state == "OPEN":
            if current_time - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                self.logger.log_info(f"Circuit breaker for {func.__name__} moving to HALF_OPEN")
            else:
                raise BCIGPTException(
                    f"Circuit breaker OPEN for {func.__name__}. Cooling down.",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    severity=ErrorSeverity.HIGH
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count if we were in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.failure_count = 0
                self.state = "CLOSED"
                self.logger.log_info(f"Circuit breaker for {func.__name__} CLOSED after successful execution")
            
            return result
            
        except self.expected_exceptions as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.log_error(
                    f"Circuit breaker OPENED for {func.__name__} after {self.failure_count} failures",
                    e
                )
            
            raise


def critical_section(lock_name: str = "default"):
    """Decorator for critical sections with distributed locking."""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # For now, use a simple in-memory lock
            # In production, this would use distributed locking (Redis, etc.)
            import threading
            
            if not hasattr(critical_section, 'locks'):
                critical_section.locks = {}
            
            if lock_name not in critical_section.locks:
                critical_section.locks[lock_name] = threading.RLock()
            
            lock = critical_section.locks[lock_name]
            
            logger.log_info(f"Acquiring lock '{lock_name}' for {func.__name__}")
            
            with lock:
                logger.log_info(f"Lock '{lock_name}' acquired for {func.__name__}")
                try:
                    return func(*args, **kwargs)
                finally:
                    logger.log_info(f"Lock '{lock_name}' released for {func.__name__}")
        
        return wrapper
    return decorator


# Input validation helpers
def is_valid_eeg_data(data) -> bool:
    """Validate EEG data format."""
    try:
        import numpy as np
        
        if not isinstance(data, np.ndarray):
            return False
        
        if data.ndim != 2:
            return False
        
        if data.size == 0:
            return False
        
        if not np.all(np.isfinite(data)):
            return False
        
        # Check reasonable dimensions
        channels, samples = data.shape
        if channels > 256 or samples < 10:  # Reasonable limits
            return False
        
        return True
        
    except Exception:
        return False


def is_valid_sampling_rate(rate) -> bool:
    """Validate sampling rate."""
    try:
        return isinstance(rate, (int, float)) and 1 <= rate <= 10000
    except Exception:
        return False


def is_valid_model_path(path) -> bool:
    """Validate model path."""
    try:
        from pathlib import Path
        return Path(path).exists() and Path(path).is_file()
    except Exception:
        return False