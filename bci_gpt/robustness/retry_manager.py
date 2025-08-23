"""
Retry Management System for BCI-GPT

Provides intelligent retry strategies for transient failures
in EEG processing, model inference, and system operations.
"""

import time
import random
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Optional, List, Dict, Union, Type
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff" 
    RANDOM_JITTER = "random_jitter"


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    jitter_max: float = 0.1
    exceptions: Optional[List[Type[Exception]]] = None
    on_retry: Optional[Callable[[int, Exception], None]] = None
    
    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = [Exception]


class RetryStats:
    """Retry statistics"""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_retries = 0
        self.average_attempts_per_call = 0.0
        self.exception_counts: Dict[str, int] = {}
    
    def record_attempt(self, success: bool, exception_type: Optional[str] = None):
        """Record an attempt"""
        self.total_attempts += 1
        
        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            if exception_type:
                self.exception_counts[exception_type] = (
                    self.exception_counts.get(exception_type, 0) + 1
                )
    
    def record_retry(self):
        """Record a retry"""
        self.total_retries += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total_calls = self.successful_attempts
        self.average_attempts_per_call = (
            self.total_attempts / max(1, total_calls)
        )
        
        return {
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts, 
            'failed_attempts': self.failed_attempts,
            'total_retries': self.total_retries,
            'success_rate': self.successful_attempts / max(1, total_calls),
            'average_attempts_per_call': self.average_attempts_per_call,
            'exception_counts': self.exception_counts.copy()
        }


class RetryManager:
    """
    Intelligent retry management for BCI processing operations
    
    Handles transient failures in EEG processing, model inference,
    and network operations with configurable retry strategies.
    """
    
    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        self.name = name
        self.config = config or RetryConfig()
        self.stats = RetryStats()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry protection"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Retry manager '{self.name}' attempt {attempt}")
                result = func(*args, **kwargs)
                
                self.stats.record_attempt(success=True)
                
                if attempt > 1:
                    logger.info(
                        f"Retry manager '{self.name}' succeeded on attempt {attempt}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                exception_type = type(e).__name__
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.error(
                        f"Non-retryable exception in '{self.name}': {e}"
                    )
                    self.stats.record_attempt(False, exception_type)
                    raise
                
                self.stats.record_attempt(False, exception_type)
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts:
                    logger.error(
                        f"Retry manager '{self.name}' exhausted all {attempt} attempts"
                    )
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Retry manager '{self.name}' attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                # Call retry callback if provided
                if self.config.on_retry:
                    try:
                        self.config.on_retry(attempt, e)
                    except Exception as callback_error:
                        logger.error(f"Retry callback failed: {callback_error}")
                
                self.stats.record_retry()
                time.sleep(delay)
        
        # All attempts failed
        logger.error(
            f"Retry manager '{self.name}' failed after {self.config.max_attempts} attempts"
        )
        raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Async retry manager '{self.name}' attempt {attempt}")
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self.stats.record_attempt(success=True)
                
                if attempt > 1:
                    logger.info(
                        f"Async retry manager '{self.name}' succeeded on attempt {attempt}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                exception_type = type(e).__name__
                
                if not self._is_retryable_exception(e):
                    logger.error(f"Non-retryable async exception in '{self.name}': {e}")
                    self.stats.record_attempt(False, exception_type)
                    raise
                
                self.stats.record_attempt(False, exception_type)
                
                if attempt == self.config.max_attempts:
                    logger.error(
                        f"Async retry manager '{self.name}' exhausted all {attempt} attempts"
                    )
                    break
                
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Async retry manager '{self.name}' attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                if self.config.on_retry:
                    try:
                        self.config.on_retry(attempt, e)
                    except Exception as callback_error:
                        logger.error(f"Async retry callback failed: {callback_error}")
                
                self.stats.record_retry()
                await asyncio.sleep(delay)
        
        logger.error(
            f"Async retry manager '{self.name}' failed after {self.config.max_attempts} attempts"
        )
        raise last_exception
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return any(
            isinstance(exception, exc_type) 
            for exc_type in self.config.exceptions
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.RANDOM_JITTER:
            delay = self.config.base_delay + random.uniform(0, self.config.base_delay)
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled
        if self.config.jitter:
            jitter = random.uniform(-self.config.jitter_max, self.config.jitter_max)
            delay = delay * (1 + jitter)
        
        # Ensure delay is within bounds
        delay = max(0.1, min(delay, self.config.max_delay))
        
        return delay
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics"""
        stats = self.stats.get_stats()
        stats['name'] = self.name
        stats['config'] = {
            'max_attempts': self.config.max_attempts,
            'base_delay': self.config.base_delay,
            'max_delay': self.config.max_delay,
            'strategy': self.config.strategy.value,
            'jitter': self.config.jitter
        }
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = RetryStats()
        logger.info(f"Retry manager '{self.name}' statistics reset")


def retry(
    name: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """Decorator for adding retry functionality to functions"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        exceptions=exceptions or [Exception],
        on_retry=on_retry
    )
    
    def decorator(func: Callable) -> Callable:
        manager = RetryManager(name, config)
        return manager(func)
    
    return decorator


def async_retry(
    name: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """Decorator for adding retry functionality to async functions"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        exceptions=exceptions or [Exception],
        on_retry=on_retry
    )
    
    def decorator(func: Callable) -> Callable:
        manager = RetryManager(name, config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await manager.execute_async(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Specialized retry decorators for BCI operations
def retry_eeg_processing(name: str, max_attempts: int = 3):
    """Retry decorator for EEG processing operations"""
    return retry(
        name=f"eeg_processing_{name}",
        max_attempts=max_attempts,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exceptions=[IOError, RuntimeError, ValueError]
    )


def retry_model_inference(name: str, max_attempts: int = 2):
    """Retry decorator for model inference operations"""
    return retry(
        name=f"model_inference_{name}",
        max_attempts=max_attempts,
        base_delay=0.5,
        strategy=RetryStrategy.FIXED_DELAY,
        exceptions=[RuntimeError, ConnectionError]
    )


def retry_network_operation(name: str, max_attempts: int = 5):
    """Retry decorator for network operations"""
    return retry(
        name=f"network_{name}",
        max_attempts=max_attempts,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exceptions=[ConnectionError, TimeoutError, OSError]
    )