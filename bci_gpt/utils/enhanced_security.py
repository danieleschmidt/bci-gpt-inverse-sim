"""
Enhanced security utilities for BCI-GPT system.
Provides comprehensive security measures, encryption, and compliance validation.
"""

import hashlib
import hmac
import secrets
import base64
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import threading


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    severity: str
    source: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AccessToken:
    """Secure access token."""
    token: str
    user_id: str
    scopes: List[str]
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


class SecureKeyManager:
    """Secure key management and rotation."""
    
    def __init__(self):
        """Initialize secure key manager."""
        self._keys: Dict[str, bytes] = {}
        self._key_metadata: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def generate_key(self, key_id: str, key_size: int = 32) -> bytes:
        """Generate a new cryptographic key.
        
        Args:
            key_id: Unique identifier for the key
            key_size: Size of key in bytes (default 32 for AES-256)
            
        Returns:
            Generated key bytes
        """
        with self._lock:
            key = secrets.token_bytes(key_size)
            self._keys[key_id] = key
            self._key_metadata[key_id] = {
                'created_at': datetime.now(),
                'size': key_size,
                'algorithm': 'AES-256' if key_size == 32 else f'Custom-{key_size*8}',
                'usage_count': 0
            }
            
            self.logger.info(f"Generated new key: {key_id}")
            return key
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a key by ID."""
        with self._lock:
            if key_id in self._keys:
                self._key_metadata[key_id]['usage_count'] += 1
                self._key_metadata[key_id]['last_used'] = datetime.now()
                return self._keys[key_id]
            return None
    
    def rotate_key(self, key_id: str) -> bytes:
        """Rotate an existing key."""
        with self._lock:
            if key_id not in self._keys:
                raise ValueError(f"Key {key_id} not found")
            
            old_metadata = self._key_metadata[key_id]
            new_key = self.generate_key(key_id, old_metadata['size'])
            
            self.logger.info(f"Rotated key: {key_id}")
            return new_key
    
    def list_keys(self) -> Dict[str, Dict]:
        """List all keys with metadata (without exposing key material)."""
        with self._lock:
            return {
                key_id: {
                    **metadata,
                    'key_hash': hashlib.sha256(key).hexdigest()[:16]  # Only show hash prefix
                }
                for key_id, metadata in self._key_metadata.items()
                for key in [self._keys[key_id]]
            }


class DataEncryption:
    """Advanced data encryption and decryption utilities."""
    
    def __init__(self, key_manager: SecureKeyManager):
        """Initialize data encryption.
        
        Args:
            key_manager: Key manager instance
        """
        self.key_manager = key_manager
        self.logger = logging.getLogger(__name__)
    
    def encrypt_data(
        self,
        data: Union[str, bytes, Dict],
        key_id: str = "default_encryption_key"
    ) -> str:
        """Encrypt data using AES encryption.
        
        Args:
            data: Data to encrypt (string, bytes, or JSON-serializable dict)
            key_id: Key identifier to use for encryption
            
        Returns:
            Base64-encoded encrypted data with metadata
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError:
            raise ImportError("cryptography package required for encryption")
        
        # Get or generate key
        key = self.key_manager.get_key(key_id)
        if key is None:
            key = self.key_manager.generate_key(key_id)
        
        # Prepare data
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Create Fernet instance
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        fernet_key = base64.urlsafe_b64encode(kdf.derive(key))
        fernet = Fernet(fernet_key)
        
        # Encrypt data
        encrypted_data = fernet.encrypt(data_bytes)
        
        # Create envelope with metadata
        envelope = {
            'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
            'salt': base64.b64encode(salt).decode('utf-8'),
            'key_id': key_id,
            'algorithm': 'Fernet-AES256',
            'timestamp': datetime.now().isoformat()
        }
        
        return base64.b64encode(json.dumps(envelope).encode('utf-8')).decode('utf-8')
    
    def decrypt_data(self, encrypted_envelope: str) -> Union[str, Dict]:
        """Decrypt data encrypted with encrypt_data.
        
        Args:
            encrypted_envelope: Base64-encoded encrypted envelope
            
        Returns:
            Decrypted data (string or dict)
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError:
            raise ImportError("cryptography package required for decryption")
        
        # Parse envelope
        envelope_data = json.loads(base64.b64decode(encrypted_envelope).decode('utf-8'))
        
        # Get key
        key = self.key_manager.get_key(envelope_data['key_id'])
        if key is None:
            raise ValueError(f"Key {envelope_data['key_id']} not found")
        
        # Reconstruct Fernet instance
        salt = base64.b64decode(envelope_data['salt'])
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        fernet_key = base64.urlsafe_b64encode(kdf.derive(key))
        fernet = Fernet(fernet_key)
        
        # Decrypt data
        encrypted_data = base64.b64decode(envelope_data['encrypted_data'])
        decrypted_bytes = fernet.decrypt(encrypted_data)
        decrypted_str = decrypted_bytes.decode('utf-8')
        
        # Try to parse as JSON, otherwise return string
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError:
            return decrypted_str


class SecurityAuditor:
    """Security event monitoring and auditing."""
    
    def __init__(self, max_events: int = 10000):
        """Initialize security auditor.
        
        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.max_events = max_events
        self.events: List[SecurityEvent] = []
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log a security event.
        
        Args:
            event_type: Type of security event (login, access_denied, etc.)
            severity: Event severity (low, medium, high, critical)
            source: Source of the event (component, IP, etc.)
            details: Additional event details
            user_id: User involved in the event
            session_id: Session ID associated with the event
        """
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source=source,
            details=details,
            user_id=user_id,
            session_id=session_id
        )
        
        with self._lock:
            self.events.append(event)
            
            # Maintain max_events limit
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
        
        # Log to standard logging
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(
            log_level,
            f"Security Event [{event_type}]: {details.get('message', 'No message')} "
            f"(Source: {source}, User: {user_id})"
        )
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.events
            if event.timestamp > cutoff_time
        ]
        
        # Aggregate statistics
        event_types = {}
        severity_counts = {}
        sources = {}
        users = {}
        
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            sources[event.source] = sources.get(event.source, 0) + 1
            if event.user_id:
                users[event.user_id] = users.get(event.user_id, 0) + 1
        
        # Identify potential threats
        threats = self._identify_threats(recent_events)
        
        return {
            'period_hours': hours,
            'total_events': len(recent_events),
            'event_types': event_types,
            'severity_distribution': severity_counts,
            'top_sources': dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]),
            'active_users': len(users),
            'potential_threats': threats,
            'high_severity_events': len([e for e in recent_events if e.severity in ['high', 'critical']])
        }
    
    def _identify_threats(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Identify potential security threats from events."""
        threats = []
        
        # Failed login attempts
        failed_logins = [e for e in events if e.event_type == 'login_failed']
        if len(failed_logins) > 10:  # More than 10 failed logins
            source_counts = {}
            for event in failed_logins:
                source = event.source
                source_counts[source] = source_counts.get(source, 0) + 1
            
            for source, count in source_counts.items():
                if count > 5:  # Same source with many failures
                    threats.append({
                        'type': 'brute_force_attack',
                        'severity': 'high',
                        'source': source,
                        'count': count,
                        'description': f'Potential brute force attack from {source} ({count} failed attempts)'
                    })
        
        # Unusual access patterns
        access_events = [e for e in events if e.event_type in ['data_access', 'api_call']]
        if len(access_events) > 100:  # High activity
            threats.append({
                'type': 'unusual_activity',
                'severity': 'medium',
                'count': len(access_events),
                'description': f'Unusually high activity detected ({len(access_events)} events)'
            })
        
        return threats


class TokenManager:
    """Secure token management for API authentication."""
    
    def __init__(self, key_manager: SecureKeyManager):
        """Initialize token manager.
        
        Args:
            key_manager: Key manager for token signing
        """
        self.key_manager = key_manager
        self.active_tokens: Dict[str, AccessToken] = {}
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def create_token(
        self,
        user_id: str,
        scopes: List[str],
        expires_in_hours: int = 24
    ) -> str:
        """Create a new access token.
        
        Args:
            user_id: User identifier
            scopes: List of permission scopes
            expires_in_hours: Token expiration time in hours
            
        Returns:
            Signed token string
        """
        # Generate token data
        token_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        token_data = {
            'token_id': token_id,
            'user_id': user_id,
            'scopes': scopes,
            'expires_at': expires_at.isoformat(),
            'issued_at': datetime.now().isoformat()
        }
        
        # Sign token
        token_string = self._sign_token(token_data)
        
        # Store token
        access_token = AccessToken(
            token=token_string,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at
        )
        
        with self._lock:
            self.active_tokens[token_id] = access_token
        
        self.logger.info(f"Created token for user {user_id} with scopes {scopes}")
        return token_string
    
    def validate_token(self, token: str) -> Optional[AccessToken]:
        """Validate and return token information.
        
        Args:
            token: Token string to validate
            
        Returns:
            AccessToken if valid, None otherwise
        """
        try:
            token_data = self._verify_token(token)
            token_id = token_data['token_id']
            
            with self._lock:
                if token_id not in self.active_tokens:
                    return None
                
                access_token = self.active_tokens[token_id]
                
                # Check expiration
                if datetime.now() > access_token.expires_at:
                    del self.active_tokens[token_id]
                    return None
                
                # Update last used
                access_token.last_used = datetime.now()
                return access_token
                
        except Exception as e:
            self.logger.warning(f"Token validation failed: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if token was revoked, False if not found
        """
        try:
            token_data = self._verify_token(token)
            token_id = token_data['token_id']
            
            with self._lock:
                if token_id in self.active_tokens:
                    del self.active_tokens[token_id]
                    self.logger.info(f"Revoked token {token_id}")
                    return True
                return False
                
        except Exception:
            return False
    
    def _sign_token(self, token_data: Dict[str, Any]) -> str:
        """Sign token data with HMAC."""
        # Get or generate signing key
        signing_key = self.key_manager.get_key("token_signing_key")
        if signing_key is None:
            signing_key = self.key_manager.generate_key("token_signing_key")
        
        # Create payload
        payload = base64.b64encode(json.dumps(token_data).encode('utf-8')).decode('utf-8')
        
        # Create signature
        signature = hmac.new(
            signing_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload}.{signature}"
    
    def _verify_token(self, token: str) -> Dict[str, Any]:
        """Verify token signature and return data."""
        try:
            payload, signature = token.split('.', 1)
        except ValueError:
            raise ValueError("Invalid token format")
        
        # Get signing key
        signing_key = self.key_manager.get_key("token_signing_key")
        if signing_key is None:
            raise ValueError("Signing key not found")
        
        # Verify signature
        expected_signature = hmac.new(
            signing_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid token signature")
        
        # Decode payload
        token_data = json.loads(base64.b64decode(payload).decode('utf-8'))
        return token_data
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens from active tokens."""
        current_time = datetime.now()
        
        with self._lock:
            expired_tokens = [
                token_id for token_id, token in self.active_tokens.items()
                if current_time > token.expires_at
            ]
            
            for token_id in expired_tokens:
                del self.active_tokens[token_id]
            
            if expired_tokens:
                self.logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")


class ComplianceValidator:
    """Enhanced compliance validation for healthcare and privacy regulations."""
    
    def __init__(self):
        """Initialize compliance validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_hipaa_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HIPAA compliance requirements.
        
        Args:
            config: System configuration to validate
            
        Returns:
            Compliance report
        """
        violations = []
        recommendations = []
        
        # Required HIPAA safeguards
        required_safeguards = {
            'encryption_at_rest': 'Data must be encrypted when stored',
            'encryption_in_transit': 'Data must be encrypted during transmission',
            'access_controls': 'Role-based access controls must be implemented',
            'audit_logging': 'All access to PHI must be logged',
            'data_backup': 'Regular data backups must be performed',
            'incident_response': 'Incident response procedures must be in place',
            'employee_training': 'Staff must be trained on HIPAA compliance',
            'business_associate_agreements': 'BAAs must be in place with vendors'
        }
        
        for safeguard, description in required_safeguards.items():
            if not config.get(safeguard, False):
                violations.append(f"Missing {safeguard}: {description}")
        
        # Technical safeguards
        if config.get('password_complexity', False) != True:
            recommendations.append("Implement strong password complexity requirements")
        
        if config.get('automatic_logoff', False) != True:
            recommendations.append("Implement automatic logoff for inactive sessions")
        
        if config.get('unique_user_identification', False) != True:
            violations.append("Each user must have unique identification")
        
        # Administrative safeguards
        if not config.get('security_officer_assigned', False):
            violations.append("HIPAA Security Officer must be assigned")
        
        if not config.get('workforce_training', False):
            violations.append("Workforce security training must be completed")
        
        # Physical safeguards
        if not config.get('facility_access_controls', False):
            recommendations.append("Implement facility access controls")
        
        if not config.get('workstation_controls', False):
            recommendations.append("Implement workstation access controls")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'score': max(0, 100 - (len(violations) * 20) - (len(recommendations) * 5))
        }
    
    def validate_gdpr_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GDPR compliance requirements."""
        violations = []
        recommendations = []
        
        # GDPR requirements
        gdpr_requirements = {
            'consent_management': 'Explicit consent mechanism must be implemented',
            'data_portability': 'Users must be able to export their data',
            'right_to_erasure': 'Users must be able to delete their data',
            'data_minimization': 'Only necessary data should be collected',
            'purpose_limitation': 'Data must only be used for stated purposes',
            'data_protection_officer': 'DPO must be appointed if required',
            'privacy_by_design': 'Privacy must be built into system design',
            'breach_notification': 'Data breach notification procedures must exist'
        }
        
        for requirement, description in gdpr_requirements.items():
            if not config.get(requirement, False):
                violations.append(f"Missing {requirement}: {description}")
        
        # Additional checks
        if not config.get('data_retention_policy', False):
            violations.append("Data retention policy must be defined and implemented")
        
        if not config.get('third_party_agreements', False):
            recommendations.append("Ensure data processing agreements with third parties")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'score': max(0, 100 - (len(violations) * 15) - (len(recommendations) * 5))
        }


# Global instances
_key_manager = SecureKeyManager()
_encryption = DataEncryption(_key_manager)
_auditor = SecurityAuditor()
_token_manager = TokenManager(_key_manager)
_compliance_validator = ComplianceValidator()


def get_key_manager() -> SecureKeyManager:
    """Get the global key manager instance."""
    return _key_manager


def get_encryption() -> DataEncryption:
    """Get the global encryption instance."""
    return _encryption


def get_security_auditor() -> SecurityAuditor:
    """Get the global security auditor instance."""
    return _auditor


def get_token_manager() -> TokenManager:
    """Get the global token manager instance."""
    return _token_manager


def get_compliance_validator() -> ComplianceValidator:
    """Get the global compliance validator instance."""
    return _compliance_validator


@contextmanager
def security_context(operation: str, user_id: Optional[str] = None):
    """Context manager for security event logging."""
    start_time = time.time()
    
    try:
        yield
        
        # Log successful operation
        _auditor.log_security_event(
            event_type=f"{operation}_success",
            severity="low",
            source="system",
            details={
                'operation': operation,
                'duration_ms': (time.time() - start_time) * 1000,
                'status': 'success'
            },
            user_id=user_id
        )
        
    except Exception as e:
        # Log failed operation
        _auditor.log_security_event(
            event_type=f"{operation}_failed",
            severity="medium",
            source="system",
            details={
                'operation': operation,
                'duration_ms': (time.time() - start_time) * 1000,
                'status': 'failed',
                'error': str(e)
            },
            user_id=user_id
        )
        raise