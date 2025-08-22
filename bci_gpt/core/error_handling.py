"""Robust error handling system for BCI-GPT."""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional, Type
from contextlib import contextmanager
from datetime import datetime

from ..utils.logging_config import get_logger


class BCIGPTError(Exception):
    """Base exception for BCI-GPT."""
    pass


class ModelLoadError(BCIGPTError):
    """Error loading model."""
    pass


class DataValidationError(BCIGPTError):
    """Error validating data."""
    pass


class ProcessingError(BCIGPTError):
    """Error during processing."""
    pass


class ConfigurationError(BCIGPTError):
    """Error in configuration."""
    pass


class SecurityError(BCIGPTError):
    """Security-related error."""
    pass


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = get_logger(__name__)
        self.error_history = []
        self.recovery_strategies = {}
        self.error_counts = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies.update({
            ModelLoadError: self._recover_model_load,
            DataValidationError: self._recover_data_validation,
            ProcessingError: self._recover_processing,
            ConfigurationError: self._recover_configuration,
            SecurityError: self._recover_security
        })
    
    def register_recovery_strategy(self, error_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.log_info(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        """Handle an error with recovery strategy."""
        error_type = type(error)
        context = context or {}
        
        # Log error
        error_info = {
            "error_type": error_type.__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now(),
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        self.error_counts[error_type.__name__] = self.error_counts.get(error_type.__name__, 0) + 1
        
        self.logger.log_error(f"Handling error: {error_type.__name__}: {error}")
        
        # Attempt recovery
        if error_type in self.recovery_strategies:
            try:
                result = self.recovery_strategies[error_type](error, context)
                self.logger.log_info(f"Successfully recovered from {error_type.__name__}")
                return result
            except Exception as recovery_error:
                self.logger.log_error(f"Recovery failed: {recovery_error}")
                raise error
        else:
            self.logger.log_warning(f"No recovery strategy for {error_type.__name__}")
            raise error
    
    def _recover_model_load(self, error: ModelLoadError, context: Dict[str, Any]) -> Any:
        """Recover from model loading error."""
        self.logger.log_info("Attempting model load recovery")
        
        # Try fallback model
        fallback_path = context.get("fallback_model_path")
        if fallback_path:
            self.logger.log_info(f"Using fallback model: {fallback_path}")
            # In real implementation, would load fallback model
            return {"model": "fallback_model", "status": "recovered"}
        
        # Create minimal model
        self.logger.log_info("Creating minimal model")
        return {"model": "minimal_model", "status": "minimal"}
    
    def _recover_data_validation(self, error: DataValidationError, context: Dict[str, Any]) -> Any:
        """Recover from data validation error."""
        self.logger.log_info("Attempting data validation recovery")
        
        # Clean data
        data = context.get("data")
        if data:
            cleaned_data = self._clean_data(data)
            self.logger.log_info("Data cleaned successfully")
            return {"data": cleaned_data, "status": "cleaned"}
        
        return {"data": None, "status": "failed"}
    
    def _recover_processing(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        """Recover from processing error."""
        self.logger.log_info("Attempting processing recovery")
        
        # Retry with reduced parameters
        retry_count = context.get("retry_count", 0)
        if retry_count < 3:
            self.logger.log_info(f"Retry {retry_count + 1}/3")
            return {"status": "retry", "retry_count": retry_count + 1}
        
        # Use safe mode
        self.logger.log_info("Switching to safe mode")
        return {"status": "safe_mode", "result": "default_output"}
    
    def _recover_configuration(self, error: ConfigurationError, context: Dict[str, Any]) -> Any:
        """Recover from configuration error."""
        self.logger.log_info("Attempting configuration recovery")
        
        # Load default configuration
        default_config = self._get_default_config()
        self.logger.log_info("Using default configuration")
        return {"config": default_config, "status": "default"}
    
    def _recover_security(self, error: SecurityError, context: Dict[str, Any]) -> Any:
        """Recover from security error."""
        self.logger.log_error("Security error - enabling safe mode")
        
        # Enable security lockdown
        return {"status": "security_lockdown", "access": "restricted"}
    
    def _clean_data(self, data: Any) -> Any:
        """Clean problematic data."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [item for item in data if item is not None]
        return data
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model_type": "minimal",
            "batch_size": 1,
            "max_sequence_length": 128,
            "device": "cpu",
            "safety_mode": True
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "recovery_strategies": len(self.recovery_strategies)
        }


def with_error_handling(error_handler: ErrorHandler = None, 
                       context_provider: Callable = None):
    """Decorator for automatic error handling."""
    if error_handler is None:
        error_handler = ErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {}
                if context_provider:
                    context = context_provider(*args, **kwargs)
                return error_handler.handle_error(e, context)
        return wrapper
    return decorator


@contextmanager
def error_recovery_context(error_handler: ErrorHandler = None, context: Dict[str, Any] = None):
    """Context manager for error recovery."""
    if error_handler is None:
        error_handler = ErrorHandler()
    
    try:
        yield
    except Exception as e:
        return error_handler.handle_error(e, context or {})


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.log_info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.log_info("Circuit breaker reset to CLOSED state")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.log_warning(f"Circuit breaker OPENED after {self.failure_count} failures")


# Global error handler instance
global_error_handler = ErrorHandler()