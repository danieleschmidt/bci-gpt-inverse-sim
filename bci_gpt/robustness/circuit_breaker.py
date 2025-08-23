"""
Circuit Breaker Pattern for BCI-GPT System

Prevents cascading failures and provides fault isolation
for critical BCI processing components.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0
    
    
@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_changes: List[Dict[str, Any]] = field(default_factory=list)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation for BCI processing components
    
    Provides automatic failure detection, isolation, and recovery
    for critical system components like EEG processing, model inference,
    and real-time decoding.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper.__name__ = f"protected_{func.__name__}"
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self._should_reject_request():
                logger.warning(f"Circuit breaker '{self.name}' rejecting request")
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is {self.state.value}"
                )
            
            self.stats.total_requests += 1
        
        start_time = time.time()
        
        try:
            # Execute the protected function
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Check for timeout
            if duration > self.config.timeout:
                logger.warning(
                    f"Function timeout ({duration:.2f}s > {self.config.timeout}s)"
                )
                self._record_failure()
                raise TimeoutError(f"Function timed out after {duration:.2f}s")
            
            self._record_success()
            return result
            
        except Exception as e:
            logger.error(f"Circuit breaker '{self.name}' caught exception: {e}")
            self._record_failure()
            raise
    
    def _should_reject_request(self) -> bool:
        """Determine if request should be rejected"""
        if self.state == CircuitState.CLOSED:
            return False
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.stats.last_failure_time >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return False
            return True
        
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return False
        
        return False
    
    def _record_success(self) -> None:
        """Record successful execution"""
        with self._lock:
            self.stats.successful_requests += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = time.time()
            
            # Transition from HALF_OPEN to CLOSED if enough successes
            if (self.state == CircuitState.HALF_OPEN and 
                self.stats.consecutive_successes >= self.config.success_threshold):
                self._transition_to_closed()
    
    def _record_failure(self) -> None:
        """Record failed execution"""
        with self._lock:
            self.stats.failed_requests += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()
            
            # Transition to OPEN if too many failures
            if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
                self.stats.consecutive_failures >= self.config.failure_threshold):
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self._log_state_change(old_state, self.state, "Failure threshold exceeded")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self._log_state_change(old_state, self.state, "Recovery timeout reached")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self._log_state_change(old_state, self.state, "Recovery successful")
    
    def _log_state_change(self, old_state: CircuitState, new_state: CircuitState, reason: str) -> None:
        """Log state change"""
        change_info = {
            'timestamp': time.time(),
            'old_state': old_state.value,
            'new_state': new_state.value,
            'reason': reason,
            'consecutive_failures': self.stats.consecutive_failures,
            'consecutive_successes': self.stats.consecutive_successes
        }
        
        self.stats.state_changes.append(change_info)
        
        logger.warning(
            f"Circuit breaker '{self.name}' state change: "
            f"{old_state.value} -> {new_state.value} ({reason})"
        )
    
    def force_open(self) -> None:
        """Manually open circuit breaker"""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.OPEN
            self._log_state_change(old_state, self.state, "Manually opened")
    
    def force_closed(self) -> None:
        """Manually close circuit breaker"""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.stats.consecutive_failures = 0
            self._log_state_change(old_state, self.state, "Manually closed")
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        with self._lock:
            self.stats = CircuitBreakerStats()
            logger.info(f"Circuit breaker '{self.name}' statistics reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': (
                    self.stats.successful_requests / max(1, self.stats.total_requests)
                ),
                'consecutive_failures': self.stats.consecutive_failures,
                'consecutive_successes': self.stats.consecutive_successes,
                'last_failure_time': self.stats.last_failure_time,
                'last_success_time': self.stats.last_success_time,
                'state_changes': len(self.stats.state_changes),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout
                }
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def remove_breaker(self, name: str) -> bool:
        """Remove circuit breaker"""
        with self._lock:
            return self._breakers.pop(name, None) is not None
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_stats() 
                for name, breaker in self._breakers.items()
            }
    
    def force_open_all(self) -> None:
        """Force open all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_open()
    
    def reset_all_stats(self) -> None:
        """Reset all statistics"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset_stats()


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get circuit breaker from global registry"""
    return _registry.get_breaker(name, config)


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0
):
    """Decorator for protecting functions with circuit breaker"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout
    )
    
    def decorator(func: Callable) -> Callable:
        breaker = get_circuit_breaker(name, config)
        return breaker(func)
    
    return decorator