"""Advanced logging configuration for BCI-GPT system."""

import logging
import logging.handlers
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class BCIGPTLogger:
    """Comprehensive logging system for BCI-GPT with security features."""
    
    def __init__(self, 
                 name: str = "bci_gpt",
                 log_level: str = "INFO",
                 log_dir: str = "./logs",
                 max_bytes: int = 10_000_000,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_performance: bool = True):
        """Initialize logging configuration.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            max_bytes: Maximum size per log file
            backup_count: Number of backup log files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_performance: Enable performance monitoring logs
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_performance = enable_performance
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.logger = self._setup_main_logger()
        self.security_logger = self._setup_security_logger()
        self.performance_logger = self._setup_performance_logger() if enable_performance else None
        self.audit_logger = self._setup_audit_logger()
        
        # Performance tracking
        self.performance_metrics = {}
        
    def _setup_main_logger(self) -> logging.Logger:
        """Setup main application logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}.log",
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_security_logger(self) -> logging.Logger:
        """Setup security-specific logger."""
        logger = logging.getLogger(f"{self.name}.security")
        logger.setLevel(logging.WARNING)  # Security logs at WARNING and above
        
        # Security formatter with additional context
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Separate security log file
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_security.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(security_formatter)
        logger.addHandler(security_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance monitoring logger."""
        logger = logging.getLogger(f"{self.name}.performance")
        logger.setLevel(logging.INFO)
        
        # JSON formatter for structured performance data
        performance_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        
        # Performance log file
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_performance.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(performance_formatter)
        logger.addHandler(performance_handler)
        
        return logger
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup audit trail logger for critical operations."""
        logger = logging.getLogger(f"{self.name}.audit")
        logger.setLevel(logging.INFO)
        
        # Audit formatter
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        
        # Audit log file
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_audit.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(audit_formatter)
        logger.addHandler(audit_handler)
        
        return logger
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Log info message with optional structured data."""
        if extra:
            message += f" | Extra: {json.dumps(extra, default=str)}"
        self.logger.info(message)
        
    def log_info(self, message: str, extra: Optional[Dict] = None):
        """Log info message with optional structured data."""
        return self.info(message, extra)
    
    def log_warning(self, message: str, extra: Optional[Dict] = None):
        """Log warning message with optional structured data."""
        if extra:
            message += f" | Extra: {json.dumps(extra, default=str)}"
        self.logger.warning(message)
    
    def log_error(self, message: str, exception: Optional[Exception] = None, extra: Optional[Dict] = None):
        """Log error message with exception and structured data."""
        if exception:
            message += f" | Exception: {str(exception)}"
        if extra:
            message += f" | Extra: {json.dumps(extra, default=str)}"
        self.logger.error(message, exc_info=exception is not None)
    
    def log_security(self, event: str, severity: str = "WARNING", details: Optional[Dict] = None):
        """Log security event."""
        message = f"Security Event: {event}"
        if details:
            message += f" | Details: {json.dumps(details, default=str)}"
        
        if severity.upper() == "CRITICAL":
            self.security_logger.critical(message)
        elif severity.upper() == "ERROR":
            self.security_logger.error(message)
        else:
            self.security_logger.warning(message)
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict] = None):
        """Log performance metrics."""
        if not self.performance_logger:
            return
        
        perf_data = {
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        self.performance_logger.info(json.dumps(perf_data))
        
        # Track performance metrics
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(duration)
    
    def log_audit(self, action: str, user: str = "system", details: Optional[Dict] = None):
        """Log audit event for critical operations."""
        audit_data = {
            "action": action,
            "user": user,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        self.audit_logger.info(json.dumps(audit_data))
    
    def start_performance_timer(self, operation: str) -> str:
        """Start performance timer for an operation."""
        timer_id = f"{operation}_{int(time.time() * 1000000)}"
        self.performance_metrics[timer_id] = time.time()
        return timer_id
    
    def end_performance_timer(self, timer_id: str, details: Optional[Dict] = None):
        """End performance timer and log results."""
        if timer_id not in self.performance_metrics:
            self.log_warning(f"Performance timer {timer_id} not found")
            return
        
        start_time = self.performance_metrics.pop(timer_id)
        duration = time.time() - start_time
        
        # Extract operation name from timer_id
        operation = "_".join(timer_id.split("_")[:-1])
        self.log_performance(operation, duration, details)
    
    def get_performance_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        if operation not in self.performance_metrics:
            return {}
        
        durations = self.performance_metrics[operation]
        if not durations:
            return {}
        
        import statistics
        
        return {
            "count": len(durations),
            "mean_ms": round(statistics.mean(durations) * 1000, 2),
            "median_ms": round(statistics.median(durations) * 1000, 2),
            "min_ms": round(min(durations) * 1000, 2),
            "max_ms": round(max(durations) * 1000, 2),
            "std_dev_ms": round(statistics.stdev(durations) * 1000, 2) if len(durations) > 1 else 0.0
        }
    
    def log_system_health(self):
        """Log system health metrics."""
        try:
            import psutil
            
            health_data = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.log_info("System health check", extra=health_data)
            
        except ImportError:
            self.log_warning("psutil not available for system health monitoring")
        except Exception as e:
            self.log_error("Failed to collect system health metrics", exception=e)


# Global logger instance
_global_logger: Optional[BCIGPTLogger] = None


def get_logger(name: Optional[str] = None, **config) -> BCIGPTLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = BCIGPTLogger(name=name or "bci_gpt", **config)
    
    return _global_logger


def setup_logging(log_level: str = "INFO", 
                 log_dir: str = "./logs",
                 enable_performance: bool = True) -> BCIGPTLogger:
    """Setup logging for BCI-GPT system."""
    global _global_logger
    
    _global_logger = BCIGPTLogger(
        name="bci_gpt",
        log_level=log_level,
        log_dir=log_dir,
        enable_performance=enable_performance
    )
    
    _global_logger.log_info("BCI-GPT logging system initialized")
    return _global_logger


class PerformanceContext:
    """Context manager for performance monitoring."""
    
    def __init__(self, operation: str, logger: Optional[BCIGPTLogger] = None, details: Optional[Dict] = None):
        self.operation = operation
        self.logger = logger or get_logger()
        self.details = details
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.logger.log_performance(self.operation, duration, self.details)
        
        if exc_type is not None:
            self.logger.log_error(
                f"Performance context {self.operation} failed",
                exception=exc_val
            )


def performance_monitor(operation: str, logger: Optional[BCIGPTLogger] = None):
    """Decorator for performance monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_instance = logger or get_logger()
            
            with PerformanceContext(operation, log_instance):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Security logging helpers
def log_security_event(event: str, severity: str = "WARNING", details: Optional[Dict] = None):
    """Global security logging function."""
    logger = get_logger()
    logger.log_security(event, severity, details)


def log_audit_event(action: str, user: str = "system", details: Optional[Dict] = None):
    """Global audit logging function."""
    logger = get_logger()
    logger.log_audit(action, user, details)