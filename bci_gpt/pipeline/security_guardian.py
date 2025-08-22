"""Minimal Security Guardian for BCI-GPT Pipeline."""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import deque

from ..utils.logging_config import get_logger


class SecurityGuardian:
    """Minimal security guardian for pipeline protection."""
    
    def __init__(self):
        """Initialize security guardian."""
        self.logger = get_logger(__name__)
        self.security_events = deque(maxlen=1000)
        self.blocked_requests = 0
        self.total_requests = 0
        
    def validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate incoming request."""
        self.total_requests += 1
        
        # Basic validation
        if not isinstance(request_data, dict):
            self._log_security_event("invalid_request_type", {"type": type(request_data)})
            self.blocked_requests += 1
            return False
        
        # Check for suspicious patterns
        if self._check_suspicious_patterns(request_data):
            self._log_security_event("suspicious_pattern", request_data)
            self.blocked_requests += 1
            return False
        
        return True
    
    def _check_suspicious_patterns(self, data: Dict[str, Any]) -> bool:
        """Check for suspicious patterns in data."""
        # Simple pattern checking
        suspicious_keys = ['eval', 'exec', '__import__', 'subprocess']
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern in suspicious_keys:
                    if pattern.lower() in value.lower():
                        return True
        
        return False
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        self.logger.warning(f"Security event: {event_type}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status."""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "security_events": len(self.security_events),
            "block_rate": self.blocked_requests / max(self.total_requests, 1),
            "status": "active"
        }