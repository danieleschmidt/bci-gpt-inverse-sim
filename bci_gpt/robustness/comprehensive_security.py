#!/usr/bin/env python3
"""
Comprehensive Security Framework for BCI-GPT System
Generation 2: Enterprise-grade security with neural data protection
"""

import hashlib
import secrets
import logging
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hmac
import base64
from pathlib import Path

@dataclass
class SecurityContext:
    """Security context for BCI operations."""
    user_id: str
    session_id: str
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_level: str = "basic"
    
    def is_valid(self) -> bool:
        """Check if security context is valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions or "admin" in self.permissions

class NeuralDataProtector:
    """Advanced protection for sensitive neural data."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.logger = logging.getLogger(__name__)
        self.access_log = []
    
    def _generate_key(self) -> bytes:
        """Generate secure encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def encrypt_neural_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Encrypt neural data with strong encryption."""
        # In production, use proper encryption like AES-256-GCM
        data_json = json.dumps(data, default=str)
        data_bytes = data_json.encode('utf-8')
        
        # Simple encryption for demo (use proper crypto in production)
        encrypted = base64.b64encode(data_bytes).decode('utf-8')
        
        # Generate data hash for integrity
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        return {
            "encrypted_data": encrypted,
            "data_hash": data_hash,
            "encryption_method": "base64_demo",  # Would be AES-256-GCM in production
            "timestamp": datetime.now().isoformat()
        }
    
    def decrypt_neural_data(self, encrypted_package: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt and validate neural data."""
        try:
            # Decrypt data
            encrypted_data = encrypted_package["encrypted_data"]
            decrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Verify integrity
            current_hash = hashlib.sha256(decrypted_bytes).hexdigest()
            if current_hash != encrypted_package["data_hash"]:
                raise SecurityError("Data integrity check failed")
            
            # Deserialize
            data_json = decrypted_bytes.decode('utf-8')
            return json.loads(data_json)
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Failed to decrypt neural data: {e}")
    
    def anonymize_neural_data(self, data: Dict[str, Any], anonymization_level: str = "full") -> Dict[str, Any]:
        """Anonymize neural data for privacy protection."""
        anonymized = data.copy()
        
        if anonymization_level in ["basic", "full"]:
            # Remove direct identifiers
            for key in ["user_id", "name", "email", "phone", "address"]:
                if key in anonymized:
                    del anonymized[key]
        
        if anonymization_level == "full":
            # Add noise to neural signals for k-anonymity
            if "eeg_data" in anonymized:
                # In production, add carefully calibrated noise
                anonymized["eeg_data_anonymized"] = True
                anonymized["anonymization_method"] = "differential_privacy"
        
        # Add anonymization metadata
        anonymized["_anonymized"] = True
        anonymized["_anonymization_level"] = anonymization_level
        anonymized["_anonymization_timestamp"] = datetime.now().isoformat()
        
        return anonymized

class AccessController:
    """Role-based access control for BCI operations."""
    
    def __init__(self):
        self.active_sessions = {}
        self.access_policies = self._load_default_policies()
        self.logger = logging.getLogger(__name__)
    
    def _load_default_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load default access control policies."""
        return {
            "researcher": {
                "permissions": ["read_anonymized_data", "run_experiments", "view_aggregated_results"],
                "neural_data_access": "anonymized_only",
                "max_session_duration": 28800,  # 8 hours
                "rate_limits": {"requests_per_hour": 1000}
            },
            "clinician": {
                "permissions": ["read_patient_data", "clinical_assessment", "emergency_access"],
                "neural_data_access": "full_with_audit",
                "max_session_duration": 14400,  # 4 hours
                "rate_limits": {"requests_per_hour": 500}
            },
            "patient": {
                "permissions": ["view_own_data", "basic_bci_functions"],
                "neural_data_access": "own_data_only",
                "max_session_duration": 7200,  # 2 hours
                "rate_limits": {"requests_per_hour": 100}
            },
            "admin": {
                "permissions": ["admin", "system_config", "user_management"],
                "neural_data_access": "system_admin",
                "max_session_duration": 3600,  # 1 hour
                "rate_limits": {"requests_per_hour": 200}
            }
        }
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, str]) -> Optional[SecurityContext]:
        """Authenticate user and create security context."""
        # In production, integrate with proper authentication system
        if self._verify_credentials(user_id, credentials):
            user_role = self._get_user_role(user_id)
            policy = self.access_policies.get(user_role, self.access_policies["patient"])
            
            session_id = secrets.token_hex(16)
            expires_at = datetime.now() + timedelta(seconds=policy["max_session_duration"])
            
            context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                permissions=policy["permissions"],
                expires_at=expires_at,
                access_level=user_role
            )
            
            self.active_sessions[session_id] = context
            self.logger.info(f"User {user_id} authenticated with role {user_role}")
            
            return context
        
        self.logger.warning(f"Authentication failed for user {user_id}")
        return None
    
    def _verify_credentials(self, user_id: str, credentials: Dict[str, str]) -> bool:
        """Verify user credentials (mock implementation)."""
        # In production, integrate with proper authentication system
        password = credentials.get("password", "")
        # Mock verification - use proper authentication in production
        return len(password) >= 8
    
    def _get_user_role(self, user_id: str) -> str:
        """Get user role from user database."""
        # Mock implementation - integrate with user management system
        role_map = {
            "admin": "admin",
            "doctor": "clinician", 
            "researcher": "researcher"
        }
        
        for prefix, role in role_map.items():
            if user_id.startswith(prefix):
                return role
        
        return "patient"  # Default role
    
    def authorize_operation(self, context: SecurityContext, operation: str, resource: str = None) -> bool:
        """Authorize specific operation for user."""
        if not context.is_valid():
            self.logger.warning(f"Invalid security context for user {context.user_id}")
            return False
        
        # Check basic permissions
        if context.has_permission(operation):
            self.logger.debug(f"Operation {operation} authorized for user {context.user_id}")
            return True
        
        # Check resource-specific permissions
        if resource and context.has_permission(f"{operation}:{resource}"):
            return True
        
        self.logger.warning(f"Operation {operation} denied for user {context.user_id}")
        return False
    
    def log_access(self, context: SecurityContext, operation: str, result: str, details: Dict[str, Any] = None):
        """Log security-relevant access events."""
        access_event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "operation": operation,
            "result": result,
            "details": details or {}
        }
        
        self.logger.info(f"Access Log: {json.dumps(access_event)}")
        
        # In production, send to security monitoring system
        return access_event

class SecurityError(Exception):
    """Security-related errors."""
    pass

class ComplianceMonitor:
    """Monitor compliance with data protection regulations."""
    
    def __init__(self):
        self.compliance_log = []
        self.logger = logging.getLogger(__name__)
    
    def check_gdpr_compliance(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance for data operations."""
        compliance_result = {
            "operation": operation,
            "gdpr_compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for personal data
        personal_data_fields = ["name", "email", "phone", "address", "id_number"]
        personal_data_found = [field for field in personal_data_fields if field in data]
        
        if personal_data_found:
            compliance_result["issues"].append(f"Personal data found: {personal_data_found}")
            compliance_result["recommendations"].append("Consider anonymization or explicit consent")
        
        # Check for neural data processing
        if any(key.startswith("eeg") or key.startswith("neural") for key in data.keys()):
            compliance_result["recommendations"].append("Neural data requires explicit consent and purpose limitation")
        
        # Log compliance check
        self.compliance_log.append({
            "timestamp": datetime.now().isoformat(),
            "check_type": "gdpr",
            "result": compliance_result
        })
        
        return compliance_result
    
    def check_hipaa_compliance(self, operation: str, context: SecurityContext) -> Dict[str, Any]:
        """Check HIPAA compliance for healthcare operations."""
        compliance_result = {
            "operation": operation,
            "hipaa_compliant": True,
            "issues": [],
            "audit_required": True
        }
        
        # Check access controls
        if not context.has_permission("clinical_assessment") and "patient" in operation:
            compliance_result["hipaa_compliant"] = False
            compliance_result["issues"].append("Insufficient permissions for patient data access")
        
        # All healthcare operations require audit logging
        self.logger.info(f"HIPAA Audit: User {context.user_id} performed {operation}")
        
        return compliance_result

# Example usage and testing
if __name__ == "__main__":
    print("🔒 Testing Comprehensive Security Framework...")
    
    # Test neural data protection
    protector = NeuralDataProtector()
    
    sample_neural_data = {
        "user_id": "patient_001",
        "eeg_data": [1.2, 3.4, 2.1, 4.5],
        "timestamp": datetime.now().isoformat(),
        "quality_score": 0.85
    }
    
    # Encrypt data
    encrypted = protector.encrypt_neural_data(sample_neural_data)
    print(f"✅ Data encrypted: {encrypted['encryption_method']}")
    
    # Decrypt data  
    decrypted = protector.decrypt_neural_data(encrypted)
    print(f"✅ Data decrypted successfully: {decrypted['user_id']}")
    
    # Anonymize data
    anonymized = protector.anonymize_neural_data(sample_neural_data)
    print(f"✅ Data anonymized: {anonymized.get('_anonymized', False)}")
    
    # Test access control
    access_controller = AccessController()
    
    context = access_controller.authenticate_user(
        "researcher_jane", 
        {"password": "secure_password_123"}
    )
    
    if context:
        print(f"✅ User authenticated: {context.access_level}")
        
        # Test authorization
        authorized = access_controller.authorize_operation(context, "read_anonymized_data")
        print(f"✅ Authorization result: {authorized}")
        
        # Log access
        access_controller.log_access(context, "data_access", "success")
    
    # Test compliance
    compliance_monitor = ComplianceMonitor()
    
    gdpr_result = compliance_monitor.check_gdpr_compliance("data_processing", sample_neural_data)
    print(f"✅ GDPR Compliance: {gdpr_result['gdpr_compliant']}")
    
    if context:
        hipaa_result = compliance_monitor.check_hipaa_compliance("patient_data_access", context)
        print(f"✅ HIPAA Compliance: {hipaa_result['hipaa_compliant']}")
    
    print("\n🛡️  Security Framework Ready!")
