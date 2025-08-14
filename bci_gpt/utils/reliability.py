"""
Enhanced reliability and error handling utilities for BCI-GPT system.
Provides comprehensive error management, circuit breakers, and retry mechanisms.
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
from enum import Enum
import threading


class FailureType(Enum):
    """Types of system failures."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error" 
    PROCESSING_ERROR = "processing_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    HARDWARE_ERROR = "hardware_error"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation for resilient error handling."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to count as failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker half-open, attempting recovery")
                else:
                    raise Exception("Circuit breaker OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return (datetime.now() - self._last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED
            if self._state == CircuitBreakerState.HALF_OPEN:
                self.logger.info("Circuit breaker recovered, state: CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self.logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} failures"
                )


class RetryManager:
    """Advanced retry mechanism with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry logic."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._retry_call(func, *args, **kwargs)
        return wrapper
    
    def _retry_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed. "
                        f"Final error: {str(e)}"
                    )
        
        # Re-raise the last exception if all retries failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            # Add jitter: delay ± 25%
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay


class HealthChecker:
    """Comprehensive system health monitoring and alerting."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, Dict] = {}
        self.failure_counts: Dict[str, int] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable, 
                      critical: bool = False, timeout: float = 30.0):
        """Register a health check function.
        
        Args:
            name: Unique name for the health check
            check_func: Function that returns True if healthy
            critical: Whether failure of this check is critical
            timeout: Maximum time to wait for check completion
        """
        self.checks[name] = {
            'func': check_func,
            'critical': critical,
            'timeout': timeout
        }
        self.failure_counts[name] = 0
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'critical_failures': [],
            'warnings': []
        }
        
        for name, config in self.checks.items():
            check_result = self._run_single_check(name, config)
            results['checks'][name] = check_result
            
            if not check_result['passed']:
                if config['critical']:
                    results['critical_failures'].append(name)
                    results['overall_status'] = 'critical'
                else:
                    results['warnings'].append(name)
                    if results['overall_status'] == 'healthy':
                        results['overall_status'] = 'degraded'
        
        self.last_check_results = results
        return results
    
    def _run_single_check(self, name: str, config: Dict) -> Dict[str, Any]:
        """Run a single health check with timeout protection."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Health check '{name}' timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(config['timeout']))
            
            try:
                passed = config['func']()
                signal.alarm(0)  # Cancel alarm
                
                if passed:
                    self.failure_counts[name] = 0
                else:
                    self.failure_counts[name] += 1
                
                return {
                    'passed': bool(passed),
                    'duration_ms': round((time.time() - start_time) * 1000, 2),
                    'failure_count': self.failure_counts[name],
                    'error': None
                }
                
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                raise e
                
        except Exception as e:
            self.failure_counts[name] += 1
            self.logger.error(f"Health check '{name}' failed: {str(e)}")
            
            return {
                'passed': False,
                'duration_ms': round((time.time() - start_time) * 1000, 2),
                'failure_count': self.failure_counts[name],
                'error': str(e)
            }
    
    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        if not self.last_check_results:
            return "No health checks have been run"
        
        results = self.last_check_results
        status = results['overall_status']
        
        summary = f"System Status: {status.upper()}\n"
        summary += f"Last Check: {results['timestamp']}\n"
        
        if results['critical_failures']:
            summary += f"Critical Failures: {', '.join(results['critical_failures'])}\n"
        
        if results['warnings']:
            summary += f"Warnings: {', '.join(results['warnings'])}\n"
        
        return summary


class ErrorReporter:
    """Centralized error reporting and analysis system."""
    
    def __init__(self, max_errors: int = 1000):
        """Initialize error reporter.
        
        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self.max_errors = max_errors
        self.errors: List[Dict] = []
        self.error_counts: Dict[str, int] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def report_error(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        severity: str = "error",
        component: str = "unknown"
    ):
        """Report an error with context information.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            severity: Error severity (debug, info, warning, error, critical)
            component: System component where error occurred
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'severity': severity,
            'component': component,
            'context': context or {}
        }
        
        # Add to errors list
        self.errors.append(error_info)
        
        # Maintain max_errors limit
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        
        # Update error counts
        error_key = f"{component}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error
        self.logger.log(
            getattr(logging, severity.upper(), logging.ERROR),
            f"[{component}] {type(error).__name__}: {str(error)}",
            extra={'context': context}
        )
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of errors in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            err for err in self.errors
            if datetime.fromisoformat(err['timestamp']) > cutoff_time
        ]
        
        # Count by type and component
        type_counts = {}
        component_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_type = error['type']
            component = error['component']
            severity = error['severity']
            
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
            component_counts[component] = component_counts.get(component, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'error_types': type_counts,
            'components': component_counts,
            'severity_distribution': severity_counts,
            'recent_errors': recent_errors[-10:]  # Last 10 errors
        }


# Global instances
_circuit_breaker = CircuitBreaker()
_retry_manager = RetryManager()
_health_checker = HealthChecker()
_error_reporter = ErrorReporter()


# Decorator functions for easy use
def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """Decorator to add circuit breaker protection to a function."""
    def decorator(func):
        cb = CircuitBreaker(failure_threshold, recovery_timeout)
        return cb(func)
    return decorator


def retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator to add retry logic to a function."""
    def decorator(func):
        rm = RetryManager(max_retries, base_delay)
        return rm(func)
    return decorator


@contextmanager
def error_context(component: str, context: Optional[Dict] = None):
    """Context manager for automatic error reporting."""
    try:
        yield
    except Exception as e:
        _error_reporter.report_error(e, context, component=component)
        raise


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker


def get_error_reporter() -> ErrorReporter:
    """Get the global error reporter instance."""
    return _error_reporter


# Standard health checks
def _check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Less than 90% memory usage
    except ImportError:
        return True  # Can't check, assume OK


def _check_disk_space() -> bool:
    """Check if disk space is sufficient."""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return disk.percent < 95  # Less than 95% disk usage
    except ImportError:
        return True  # Can't check, assume OK


def _check_python_version() -> bool:
    """Check if Python version is supported."""
    import sys
    return sys.version_info >= (3, 9)


# Register standard health checks
_health_checker.register_check("memory_usage", _check_memory_usage, critical=True)
_health_checker.register_check("disk_space", _check_disk_space, critical=True)
_health_checker.register_check("python_version", _check_python_version, critical=True)