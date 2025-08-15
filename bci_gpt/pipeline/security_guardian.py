"""Security Guardian for BCI-GPT Self-Healing System.

Provides comprehensive security monitoring, threat detection, access control,
and security incident response for the self-healing pipeline system.
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json
import ipaddress

from ..utils.monitoring import HealthStatus
from ..utils.error_handling import BCI_GPTError
from ..utils.enhanced_security import SecurityManager


class ThreatLevel(Enum):
    """Security threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS_VIOLATION = "data_access_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SYSTEM_INTRUSION = "system_intrusion"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    severity: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    component: str
    description: str
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


@dataclass
class AccessAttempt:
    """Access attempt record for monitoring."""
    user_id: str
    source_ip: str
    component: str
    action: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # seconds
    session_timeout: int = 3600  # seconds
    require_mfa: bool = True
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    min_password_strength: int = 8
    audit_retention_days: int = 90
    encryption_required: bool = True


class SecurityGuardian:
    """Comprehensive security guardian for the BCI-GPT self-healing system.
    
    Monitors security events, detects threats, enforces policies,
    and responds to security incidents automatically.
    """
    
    def __init__(self, security_policy: Optional[SecurityPolicy] = None):
        self.logger = logging.getLogger(__name__)
        self.security_policy = security_policy or SecurityPolicy()
        
        # Security monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Event tracking
        self.security_events: deque = deque(maxlen=1000)
        self.access_attempts: deque = deque(maxlen=5000)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Threat detection
        self.threat_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.failed_attempts: Dict[str, Dict[str, Any]] = {}  # IP/User -> attempt info
        self.locked_accounts: Dict[str, datetime] = {}
        self.blocked_ips: Set[str] = set(self.security_policy.blocked_ips)
        
        # Security callbacks
        self.security_callbacks: List[Callable] = []
        self.incident_callbacks: List[Callable] = []
        
        # Encryption and authentication
        self.security_manager = SecurityManager()
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Audit logging
        self.audit_log: deque = deque(maxlen=10000)
        
        # Statistics
        self.total_events = 0
        self.total_threats_blocked = 0
        self.total_incidents = 0
        self.guardian_start_time = datetime.now()
        
        # Initialize threat detection patterns
        self._initialize_threat_patterns()
    
    def _initialize_threat_patterns(self) -> None:
        """Initialize threat detection patterns."""
        self.threat_patterns = {
            "brute_force": [
                {"pattern": "multiple_failed_logins", "threshold": 5, "timeframe": 300},
                {"pattern": "rapid_requests", "threshold": 100, "timeframe": 60},
                {"pattern": "dictionary_attack", "threshold": 10, "timeframe": 600}
            ],
            "injection": [
                {"pattern": "sql_keywords", "keywords": ["SELECT", "DROP", "INSERT", "DELETE", "UNION"]},
                {"pattern": "script_tags", "keywords": ["<script>", "javascript:", "onload="]},
                {"pattern": "command_injection", "keywords": ["; rm -rf", "| cat", "&& wget"]}
            ],
            "anomaly": [
                {"pattern": "unusual_access_time", "threshold": 0.1},
                {"pattern": "geographic_anomaly", "threshold": 1000},  # km from usual location
                {"pattern": "access_pattern_change", "threshold": 0.3}
            ]
        }
    
    def start_monitoring(self) -> None:
        """Start security monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Security guardian monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Security guardian monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main security monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for threat patterns
                self._detect_threat_patterns()
                
                # Monitor active sessions
                self._monitor_active_sessions()
                
                # Clean up expired data
                self._cleanup_expired_data()
                
                # Check for anomalous behavior
                self._detect_anomalies()
                
                # Update threat intelligence
                self._update_threat_intelligence()
                
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
            
            threading.Event().wait(10.0)  # Check every 10 seconds
    
    def authenticate_request(self, user_id: str, password: str, source_ip: str, 
                           component: str) -> Dict[str, Any]:
        """Authenticate a user request."""
        # Check if IP is blocked
        if self._is_ip_blocked(source_ip):
            self._record_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ThreatLevel.HIGH,
                source_ip,
                user_id,
                component,
                "Authentication attempt from blocked IP",
                {"reason": "blocked_ip"}
            )
            return {"success": False, "reason": "blocked_ip"}
        
        # Check if account is locked
        if self._is_account_locked(user_id):
            self._record_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ThreatLevel.MEDIUM,
                source_ip,
                user_id,
                component,
                "Authentication attempt on locked account",
                {"reason": "account_locked"}
            )
            return {"success": False, "reason": "account_locked"}
        
        # Perform authentication (simplified - would integrate with actual auth system)
        auth_success = self._verify_credentials(user_id, password)
        
        # Record access attempt
        self._record_access_attempt(
            user_id, source_ip, component, "authenticate", auth_success
        )
        
        if auth_success:
            # Reset failed attempts for this user/IP
            self._reset_failed_attempts(user_id, source_ip)
            
            # Create session
            session_token = self._create_session(user_id, source_ip, component)
            
            return {
                "success": True,
                "session_token": session_token,
                "expires_at": (datetime.now() + timedelta(seconds=self.security_policy.session_timeout)).isoformat()
            }
        else:
            # Record failed attempt
            self._record_failed_attempt(user_id, source_ip)
            
            # Check if we should lock account or block IP
            if self._should_lock_account(user_id):
                self._lock_account(user_id)
            
            if self._should_block_ip(source_ip):
                self._block_ip_address(source_ip)
            
            return {"success": False, "reason": "invalid_credentials"}
    
    def authorize_action(self, session_token: str, component: str, action: str, 
                        resource: str = None) -> Dict[str, Any]:
        """Authorize an action for a session."""
        session = self._validate_session(session_token)
        if not session:
            return {"authorized": False, "reason": "invalid_session"}
        
        user_id = session["user_id"]
        source_ip = session["source_ip"]
        
        # Check permissions (simplified - would integrate with RBAC system)
        authorized = self._check_permissions(user_id, component, action, resource)
        
        # Record access attempt
        self._record_access_attempt(
            user_id, source_ip, component, f"{action}:{resource}", authorized
        )
        
        if not authorized:
            self._record_security_event(
                SecurityEventType.AUTHORIZATION_VIOLATION,
                ThreatLevel.MEDIUM,
                source_ip,
                user_id,
                component,
                f"Unauthorized action: {action} on {resource}",
                {"action": action, "resource": resource}
            )
        
        return {"authorized": authorized, "user_id": user_id}
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP address is blocked."""
        return ip in self.blocked_ips
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked."""
        if user_id not in self.locked_accounts:
            return False
        
        lock_time = self.locked_accounts[user_id]
        unlock_time = lock_time + timedelta(seconds=self.security_policy.lockout_duration)
        
        if datetime.now() > unlock_time:
            del self.locked_accounts[user_id]
            return False
        
        return True
    
    def _verify_credentials(self, user_id: str, password: str) -> bool:
        """Verify user credentials (simplified implementation)."""
        # In a real implementation, this would check against a secure user database
        # For now, just perform basic validation
        if not user_id or not password:
            return False
        
        # Check password strength
        if len(password) < self.security_policy.min_password_strength:
            return False
        
        # Simulate credential verification
        return True  # Placeholder - would perform actual verification
    
    def _record_access_attempt(self, user_id: str, source_ip: str, component: str, 
                              action: str, success: bool, details: Dict[str, Any] = None) -> None:
        """Record an access attempt."""
        attempt = AccessAttempt(
            user_id=user_id,
            source_ip=source_ip,
            component=component,
            action=action,
            success=success,
            details=details or {}
        )
        
        self.access_attempts.append(attempt)
        
        # Add to audit log
        self._add_audit_entry("access_attempt", {
            "user_id": user_id,
            "source_ip": source_ip,
            "component": component,
            "action": action,
            "success": success,
            "timestamp": attempt.timestamp.isoformat()
        })
    
    def _record_failed_attempt(self, user_id: str, source_ip: str) -> None:
        """Record a failed authentication attempt."""
        key = f"{user_id}:{source_ip}"
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = {
                "count": 0,
                "first_attempt": datetime.now(),
                "last_attempt": datetime.now()
            }
        
        self.failed_attempts[key]["count"] += 1
        self.failed_attempts[key]["last_attempt"] = datetime.now()
    
    def _reset_failed_attempts(self, user_id: str, source_ip: str) -> None:
        """Reset failed attempts for user/IP."""
        key = f"{user_id}:{source_ip}"
        if key in self.failed_attempts:
            del self.failed_attempts[key]
    
    def _should_lock_account(self, user_id: str) -> bool:
        """Check if account should be locked based on failed attempts."""
        total_failures = 0
        cutoff_time = datetime.now() - timedelta(seconds=300)  # 5 minutes
        
        for key, attempt_info in self.failed_attempts.items():
            if key.startswith(f"{user_id}:") and attempt_info["last_attempt"] > cutoff_time:
                total_failures += attempt_info["count"]
        
        return total_failures >= self.security_policy.max_failed_attempts
    
    def _should_block_ip(self, source_ip: str) -> bool:
        """Check if IP should be blocked based on failed attempts."""
        total_failures = 0
        cutoff_time = datetime.now() - timedelta(seconds=300)  # 5 minutes
        
        for key, attempt_info in self.failed_attempts.items():
            if key.endswith(f":{source_ip}") and attempt_info["last_attempt"] > cutoff_time:
                total_failures += attempt_info["count"]
        
        return total_failures >= self.security_policy.max_failed_attempts * 2  # Higher threshold for IP blocking
    
    def _lock_account(self, user_id: str) -> None:
        """Lock a user account."""
        self.locked_accounts[user_id] = datetime.now()
        
        self._record_security_event(
            SecurityEventType.BRUTE_FORCE_ATTACK,
            ThreatLevel.HIGH,
            "multiple",
            user_id,
            "authentication",
            f"Account {user_id} locked due to excessive failed attempts",
            {"action": "account_locked"}
        )
        
        self.logger.warning(f"Account locked: {user_id}")
    
    def _block_ip_address(self, ip: str) -> None:
        \"\"\"Block an IP address.\"\"\"\n        self.blocked_ips.add(ip)\n        \n        self._record_security_event(\n            SecurityEventType.BRUTE_FORCE_ATTACK,\n            ThreatLevel.HIGH,\n            ip,\n            None,\n            \"authentication\",\n            f\"IP {ip} blocked due to excessive failed attempts\",\n            {\"action\": \"ip_blocked\"}\n        )\n        \n        self.logger.warning(f\"IP blocked: {ip}\")\n    \n    def _create_session(self, user_id: str, source_ip: str, component: str) -> str:\n        \"\"\"Create a new session for authenticated user.\"\"\"\n        session_token = secrets.token_urlsafe(32)\n        \n        session_data = {\n            \"user_id\": user_id,\n            \"source_ip\": source_ip,\n            \"component\": component,\n            \"created_at\": datetime.now(),\n            \"last_activity\": datetime.now(),\n            \"expires_at\": datetime.now() + timedelta(seconds=self.security_policy.session_timeout)\n        }\n        \n        self.active_sessions[session_token] = session_data\n        \n        self._add_audit_entry(\"session_created\", {\n            \"user_id\": user_id,\n            \"source_ip\": source_ip,\n            \"session_token\": session_token[:8] + \"...\",  # Partial token for audit\n            \"timestamp\": datetime.now().isoformat()\n        })\n        \n        return session_token\n    \n    def _validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Validate a session token.\"\"\"\n        if session_token not in self.active_sessions:\n            return None\n        \n        session = self.active_sessions[session_token]\n        \n        # Check if session is expired\n        if datetime.now() > session[\"expires_at\"]:\n            del self.active_sessions[session_token]\n            return None\n        \n        # Update last activity\n        session[\"last_activity\"] = datetime.now()\n        \n        return session\n    \n    def _check_permissions(self, user_id: str, component: str, action: str, resource: str = None) -> bool:\n        \"\"\"Check if user has permission for action (simplified RBAC).\"\"\"\n        # Simplified permission check - would integrate with actual RBAC system\n        \n        # Admin users have all permissions\n        if user_id.startswith(\"admin_\"):\n            return True\n        \n        # Component-specific permissions\n        if component == \"pipeline\":\n            return action in [\"status\", \"monitor\"]\n        elif component == \"model\":\n            return action in [\"inference\", \"status\"]\n        elif component == \"data\":\n            return action in [\"read\", \"status\"]\n        elif component == \"realtime\":\n            return action in [\"process\", \"status\"]\n        \n        # Default deny\n        return False\n    \n    def _record_security_event(self, event_type: SecurityEventType, severity: ThreatLevel,\n                              source_ip: str, user_id: Optional[str], component: str,\n                              description: str, evidence: Dict[str, Any]) -> None:\n        \"\"\"Record a security event.\"\"\"\n        event = SecurityEvent(\n            event_type=event_type,\n            severity=severity,\n            source_ip=source_ip,\n            user_id=user_id,\n            component=component,\n            description=description,\n            evidence=evidence\n        )\n        \n        self.security_events.append(event)\n        self.total_events += 1\n        \n        # Add to audit log\n        self._add_audit_entry(\"security_event\", {\n            \"event_type\": event_type.value,\n            \"severity\": severity.value,\n            \"source_ip\": source_ip,\n            \"user_id\": user_id,\n            \"component\": component,\n            \"description\": description,\n            \"timestamp\": event.timestamp.isoformat()\n        })\n        \n        # Trigger callbacks for high/critical events\n        if severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:\n            self._trigger_security_callbacks(event)\n            \n            # Auto-respond to critical threats\n            if severity == ThreatLevel.CRITICAL:\n                asyncio.run(self._auto_respond_to_threat(event))\n        \n        self.logger.warning(f\"Security event: {event_type.value} - {description}\")\n    \n    def _detect_threat_patterns(self) -> None:\n        \"\"\"Detect threat patterns in recent events.\"\"\"\n        current_time = datetime.now()\n        \n        # Brute force detection\n        self._detect_brute_force_patterns(current_time)\n        \n        # Injection attempt detection\n        self._detect_injection_patterns(current_time)\n        \n        # Anomaly detection\n        self._detect_behavioral_anomalies(current_time)\n    \n    def _detect_brute_force_patterns(self, current_time: datetime) -> None:\n        \"\"\"Detect brute force attack patterns.\"\"\"\n        # Check for rapid failed attempts from same IP\n        ip_attempts = {}\n        cutoff_time = current_time - timedelta(minutes=5)\n        \n        for attempt in self.access_attempts:\n            if attempt.timestamp > cutoff_time and not attempt.success:\n                ip = attempt.source_ip\n                if ip not in ip_attempts:\n                    ip_attempts[ip] = []\n                ip_attempts[ip].append(attempt)\n        \n        for ip, attempts in ip_attempts.items():\n            if len(attempts) >= 10:  # 10 failed attempts in 5 minutes\n                self._record_security_event(\n                    SecurityEventType.BRUTE_FORCE_ATTACK,\n                    ThreatLevel.HIGH,\n                    ip,\n                    None,\n                    \"authentication\",\n                    f\"Brute force attack detected from IP {ip}\",\n                    {\"failed_attempts\": len(attempts), \"timeframe\": \"5_minutes\"}\n                )\n    \n    def _detect_injection_patterns(self, current_time: datetime) -> None:\n        \"\"\"Detect injection attack patterns.\"\"\"\n        # This would analyze request content for injection patterns\n        # Simplified implementation - would need actual request data\n        pass\n    \n    def _detect_behavioral_anomalies(self, current_time: datetime) -> None:\n        \"\"\"Detect anomalous user behavior patterns.\"\"\"\n        # Check for unusual access times\n        for session_token, session in self.active_sessions.items():\n            user_id = session[\"user_id\"]\n            \n            # Check if accessing at unusual times (simplified)\n            current_hour = current_time.hour\n            if current_hour < 6 or current_hour > 22:  # Outside business hours\n                # Check if this is unusual for this user\n                if not self._is_usual_access_time(user_id, current_hour):\n                    self._record_security_event(\n                        SecurityEventType.ANOMALOUS_BEHAVIOR,\n                        ThreatLevel.MEDIUM,\n                        session[\"source_ip\"],\n                        user_id,\n                        \"session\",\n                        f\"Unusual access time for user {user_id}\",\n                        {\"access_hour\": current_hour, \"usual_hours\": \"6-22\"}\n                    )\n    \n    def _is_usual_access_time(self, user_id: str, hour: int) -> bool:\n        \"\"\"Check if access time is usual for user.\"\"\"\n        # Simplified - would analyze historical access patterns\n        return 6 <= hour <= 22  # Business hours\n    \n    def _monitor_active_sessions(self) -> None:\n        \"\"\"Monitor active sessions for security issues.\"\"\"\n        current_time = datetime.now()\n        expired_sessions = []\n        \n        for session_token, session in self.active_sessions.items():\n            # Check for expired sessions\n            if current_time > session[\"expires_at\"]:\n                expired_sessions.append(session_token)\n                continue\n            \n            # Check for session anomalies\n            session_duration = (current_time - session[\"created_at\"]).total_seconds()\n            if session_duration > 86400:  # 24 hours\n                self._record_security_event(\n                    SecurityEventType.SUSPICIOUS_ACTIVITY,\n                    ThreatLevel.MEDIUM,\n                    session[\"source_ip\"],\n                    session[\"user_id\"],\n                    \"session\",\n                    \"Unusually long session duration\",\n                    {\"duration_hours\": session_duration / 3600}\n                )\n        \n        # Clean up expired sessions\n        for session_token in expired_sessions:\n            del self.active_sessions[session_token]\n    \n    def _cleanup_expired_data(self) -> None:\n        \"\"\"Clean up expired security data.\"\"\"\n        current_time = datetime.now()\n        cutoff_time = current_time - timedelta(hours=24)\n        \n        # Clean up old failed attempts\n        expired_attempts = []\n        for key, attempt_info in self.failed_attempts.items():\n            if attempt_info[\"last_attempt\"] < cutoff_time:\n                expired_attempts.append(key)\n        \n        for key in expired_attempts:\n            del self.failed_attempts[key]\n        \n        # Clean up old audit logs\n        audit_cutoff = current_time - timedelta(days=self.security_policy.audit_retention_days)\n        while self.audit_log and self.audit_log[0][\"timestamp\"] < audit_cutoff:\n            self.audit_log.popleft()\n    \n    def _detect_anomalies(self) -> None:\n        \"\"\"Detect security anomalies using pattern analysis.\"\"\"\n        # This would implement more sophisticated anomaly detection\n        # using machine learning or statistical analysis\n        pass\n    \n    def _update_threat_intelligence(self) -> None:\n        \"\"\"Update threat intelligence and patterns.\"\"\"\n        # This would update threat patterns based on external threat feeds\n        # and internal learning\n        pass\n    \n    async def _auto_respond_to_threat(self, event: SecurityEvent) -> None:\n        \"\"\"Automatically respond to critical security threats.\"\"\"\n        self.logger.critical(f\"Auto-responding to critical threat: {event.description}\")\n        \n        response_actions = []\n        \n        # Block IP for critical threats\n        if event.source_ip and event.source_ip != \"unknown\":\n            self.blocked_ips.add(event.source_ip)\n            response_actions.append(f\"blocked_ip_{event.source_ip}\")\n        \n        # Lock user account for authentication-related threats\n        if event.user_id and event.event_type in [\n            SecurityEventType.BRUTE_FORCE_ATTACK,\n            SecurityEventType.PRIVILEGE_ESCALATION\n        ]:\n            self.locked_accounts[event.user_id] = datetime.now()\n            response_actions.append(f\"locked_account_{event.user_id}\")\n        \n        # Terminate suspicious sessions\n        if event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY:\n            terminated_sessions = self._terminate_suspicious_sessions(event.user_id, event.source_ip)\n            response_actions.extend(terminated_sessions)\n        \n        # Update event with response actions\n        event.response_actions = response_actions\n        \n        self.total_threats_blocked += 1\n        \n        # Notify incident callbacks\n        for callback in self.incident_callbacks:\n            try:\n                callback(event)\n            except Exception as e:\n                self.logger.error(f\"Incident callback error: {e}\")\n    \n    def _terminate_suspicious_sessions(self, user_id: str = None, source_ip: str = None) -> List[str]:\n        \"\"\"Terminate suspicious sessions.\"\"\"\n        terminated = []\n        sessions_to_remove = []\n        \n        for session_token, session in self.active_sessions.items():\n            should_terminate = False\n            \n            if user_id and session[\"user_id\"] == user_id:\n                should_terminate = True\n            elif source_ip and session[\"source_ip\"] == source_ip:\n                should_terminate = True\n            \n            if should_terminate:\n                sessions_to_remove.append(session_token)\n                terminated.append(f\"terminated_session_{session_token[:8]}\")\n        \n        for session_token in sessions_to_remove:\n            del self.active_sessions[session_token]\n        \n        return terminated\n    \n    def _trigger_security_callbacks(self, event: SecurityEvent) -> None:\n        \"\"\"Trigger security event callbacks.\"\"\"\n        for callback in self.security_callbacks:\n            try:\n                callback(event)\n            except Exception as e:\n                self.logger.error(f\"Security callback error: {e}\")\n    \n    def _add_audit_entry(self, event_type: str, details: Dict[str, Any]) -> None:\n        \"\"\"Add entry to audit log.\"\"\"\n        audit_entry = {\n            \"timestamp\": datetime.now(),\n            \"event_type\": event_type,\n            \"details\": details\n        }\n        \n        self.audit_log.append(audit_entry)\n    \n    def encrypt_data(self, data: str, context: str = None) -> str:\n        \"\"\"Encrypt sensitive data.\"\"\"\n        return self.security_manager.encrypt_data(data, context)\n    \n    def decrypt_data(self, encrypted_data: str, context: str = None) -> str:\n        \"\"\"Decrypt sensitive data.\"\"\"\n        return self.security_manager.decrypt_data(encrypted_data, context)\n    \n    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:\n        \"\"\"Generate API key for user.\"\"\"\n        api_key = secrets.token_urlsafe(32)\n        \n        self.api_keys[api_key] = {\n            \"user_id\": user_id,\n            \"permissions\": permissions,\n            \"created_at\": datetime.now(),\n            \"last_used\": None,\n            \"usage_count\": 0\n        }\n        \n        return api_key\n    \n    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Validate API key and return key info.\"\"\"\n        if api_key not in self.api_keys:\n            return None\n        \n        key_info = self.api_keys[api_key]\n        key_info[\"last_used\"] = datetime.now()\n        key_info[\"usage_count\"] += 1\n        \n        return key_info\n    \n    def register_security_callback(self, callback: Callable) -> None:\n        \"\"\"Register callback for security events.\"\"\"\n        self.security_callbacks.append(callback)\n    \n    def register_incident_callback(self, callback: Callable) -> None:\n        \"\"\"Register callback for security incidents.\"\"\"\n        self.incident_callbacks.append(callback)\n    \n    def get_security_status(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive security status.\"\"\"\n        current_time = datetime.now()\n        uptime = (current_time - self.guardian_start_time).total_seconds()\n        \n        # Calculate threat statistics\n        recent_events = [e for e in self.security_events if (current_time - e.timestamp).total_seconds() < 3600]\n        threat_counts = {}\n        for event in recent_events:\n            threat_type = event.event_type.value\n            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1\n        \n        return {\n            \"monitoring_active\": self.monitoring_active,\n            \"uptime_seconds\": uptime,\n            \"total_events\": self.total_events,\n            \"total_threats_blocked\": self.total_threats_blocked,\n            \"total_incidents\": self.total_incidents,\n            \"active_sessions\": len(self.active_sessions),\n            \"blocked_ips\": len(self.blocked_ips),\n            \"locked_accounts\": len(self.locked_accounts),\n            \"recent_threats\": threat_counts,\n            \"security_policy\": {\n                \"max_failed_attempts\": self.security_policy.max_failed_attempts,\n                \"lockout_duration\": self.security_policy.lockout_duration,\n                \"session_timeout\": self.security_policy.session_timeout,\n                \"require_mfa\": self.security_policy.require_mfa,\n                \"encryption_required\": self.security_policy.encryption_required\n            },\n            \"audit_log_size\": len(self.audit_log),\n            \"threat_detection_active\": True,\n            \"auto_response_enabled\": True,\n            \"timestamp\": current_time.isoformat()\n        }\n    \n    def get_security_events(self, limit: int = 100, severity: ThreatLevel = None) -> List[Dict[str, Any]]:\n        \"\"\"Get recent security events.\"\"\"\n        events = list(self.security_events)\n        \n        # Filter by severity if specified\n        if severity:\n            events = [e for e in events if e.severity == severity]\n        \n        # Sort by timestamp (most recent first)\n        events.sort(key=lambda x: x.timestamp, reverse=True)\n        \n        # Limit results\n        events = events[:limit]\n        \n        # Convert to dict format\n        return [\n            {\n                \"event_type\": e.event_type.value,\n                \"severity\": e.severity.value,\n                \"source_ip\": e.source_ip,\n                \"user_id\": e.user_id,\n                \"component\": e.component,\n                \"description\": e.description,\n                \"evidence\": e.evidence,\n                \"timestamp\": e.timestamp.isoformat(),\n                \"resolved\": e.resolved,\n                \"response_actions\": e.response_actions\n            }\n            for e in events\n        ]\n    \n    def get_audit_log(self, limit: int = 100, event_type: str = None) -> List[Dict[str, Any]]:\n        \"\"\"Get audit log entries.\"\"\"\n        entries = list(self.audit_log)\n        \n        # Filter by event type if specified\n        if event_type:\n            entries = [e for e in entries if e[\"event_type\"] == event_type]\n        \n        # Sort by timestamp (most recent first)\n        entries.sort(key=lambda x: x[\"timestamp\"], reverse=True)\n        \n        # Limit results and convert timestamps\n        return [\n            {\n                \"timestamp\": e[\"timestamp\"].isoformat(),\n                \"event_type\": e[\"event_type\"],\n                \"details\": e[\"details\"]\n            }\n            for e in entries[:limit]\n        ]"