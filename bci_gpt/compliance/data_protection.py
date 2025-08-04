"""Data protection and privacy framework for BCI-GPT."""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
import base64
import warnings

logger = logging.getLogger(__name__)

try:
    import cryptography
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    warnings.warn("Cryptography library not available - encryption features disabled")

class DataProtectionManager:
    """Manages data protection and privacy for BCI-GPT."""
    
    def __init__(self, 
                 key_storage_path: Optional[Path] = None,
                 encryption_enabled: bool = True):
        """Initialize data protection manager.
        
        Args:
            key_storage_path: Path to store encryption keys
            encryption_enabled: Whether to enable encryption features
        """
        self.encryption_enabled = encryption_enabled and HAS_CRYPTOGRAPHY
        
        if not self.encryption_enabled:
            logger.warning("Encryption disabled - data will not be encrypted")
        
        # Set up key storage
        self.key_storage_path = key_storage_path or Path("./keys")
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption keys if needed
        self._master_key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None
        
        if self.encryption_enabled:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption system."""
        try:
            master_key_file = self.key_storage_path / "master.key"
            
            # Load or generate master key
            if master_key_file.exists():
                with open(master_key_file, 'rb') as f:
                    self._master_key = f.read()
            else:
                # Generate new master key
                self._master_key = secrets.token_bytes(32)
                with open(master_key_file, 'wb') as f:
                    f.write(self._master_key)
                # Restrict permissions
                master_key_file.chmod(0o600)
            
            # Initialize Fernet cipher
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'bci_gpt_salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._master_key))
            self._fernet = Fernet(key)
            
            logger.info("Encryption system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.encryption_enabled = False
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> Optional[bytes]:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt (string, bytes, or dictionary)
            
        Returns:
            Encrypted data as bytes, or None if encryption failed
        """
        if not self.encryption_enabled:
            logger.warning("Encryption not available")
            return None
        
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt
            encrypted_data = self._fernet.encrypt(data_bytes)
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes, 
                    return_type: str = 'bytes') -> Optional[Union[str, bytes, Dict[str, Any]]]:
        """Decrypt data.
        
        Args:
            encrypted_data: Encrypted data as bytes
            return_type: Type to return ('bytes', 'str', 'json')
            
        Returns:
            Decrypted data in requested format, or None if decryption failed
        """
        if not self.encryption_enabled:
            logger.warning("Encryption not available")
            return None
        
        try:
            # Decrypt
            decrypted_bytes = self._fernet.decrypt(encrypted_data)
            
            # Convert to requested type
            if return_type == 'str':
                return decrypted_bytes.decode('utf-8')
            elif return_type == 'json':
                return json.loads(decrypted_bytes.decode('utf-8'))
            else:
                return decrypted_bytes
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def hash_identifier(self, identifier: str, salt: Optional[str] = None) -> str:
        """Create a one-way hash of an identifier for pseudonymization.
        
        Args:
            identifier: Original identifier
            salt: Optional salt for hashing
            
        Returns:
            Hashed identifier as hex string
        """
        if salt is None:
            salt = "bci_gpt_default_salt"  # In production, use random salt per user
        
        # Create hash
        hash_input = f"{identifier}{salt}".encode('utf-8')
        hash_object = hashlib.sha256(hash_input)
        return hash_object.hexdigest()
    
    def pseudonymize_subject_id(self, original_id: str) -> str:
        """Create a pseudonymized subject ID.
        
        Args:
            original_id: Original subject identifier
            
        Returns:
            Pseudonymized identifier
        """
        # Use HMAC for better security
        if not hasattr(self, '_pseudonym_key'):
            self._pseudonym_key = secrets.token_bytes(32)
        
        mac = hmac.new(
            self._pseudonym_key,
            original_id.encode('utf-8'),
            hashlib.sha256
        )
        return f"subj_{mac.hexdigest()[:16]}"
    
    def anonymize_data(self, data: Dict[str, Any], 
                      fields_to_remove: Optional[List[str]] = None) -> Dict[str, Any]:
        """Anonymize data by removing or hashing identifying fields.
        
        Args:
            data: Data dictionary to anonymize
            fields_to_remove: List of field names to remove completely
            
        Returns:
            Anonymized data dictionary
        """
        if fields_to_remove is None:
            fields_to_remove = ['name', 'email', 'phone', 'address', 'birth_date']
        
        anonymized = data.copy()
        
        # Remove identifying fields
        for field in fields_to_remove:
            if field in anonymized:
                del anonymized[field]
        
        # Hash certain fields instead of removing
        hash_fields = ['subject_id', 'session_id']
        for field in hash_fields:
            if field in anonymized:
                anonymized[field] = self.hash_identifier(str(anonymized[field]))
        
        # Add anonymization timestamp
        from datetime import datetime
        anonymized['_anonymized_at'] = datetime.now().isoformat()
        
        return anonymized
    
    def generate_data_usage_token(self, subject_id: str, 
                                 purposes: List[str],
                                 expiry_days: int = 30) -> str:
        """Generate a token for authorized data usage.
        
        Args:
            subject_id: Data subject identifier
            purposes: List of purposes for data usage
            expiry_days: Token validity period in days
            
        Returns:
            Usage token string
        """
        from datetime import datetime, timedelta
        
        token_data = {
            'subject_id': subject_id,
            'purposes': purposes,
            'issued_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=expiry_days)).isoformat(),
            'token_id': secrets.token_hex(16)
        }
        
        # Create signed token
        token_json = json.dumps(token_data, sort_keys=True)
        signature = hmac.new(
            self._master_key if self._master_key else b'default_key',
            token_json.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        token = base64.b64encode(f"{token_json}:{signature}".encode('utf-8')).decode('utf-8')
        return token
    
    def validate_data_usage_token(self, token: str) -> Dict[str, Any]:
        """Validate and parse a data usage token.
        
        Args:
            token: Token string to validate
            
        Returns:
            Token data if valid, empty dict if invalid
        """
        try:
            # Decode token
            decoded = base64.b64decode(token.encode('utf-8')).decode('utf-8')
            token_json, signature = decoded.rsplit(':', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self._master_key if self._master_key else b'default_key',
                token_json.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Invalid token signature")
                return {}
            
            # Parse token data
            token_data = json.loads(token_json)
            
            # Check expiry
            from datetime import datetime
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expires_at:
                logger.warning("Token has expired")
                return {}
            
            return token_data
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return {}
    
    def secure_delete_file(self, file_path: Path) -> bool:
        """Securely delete a file by overwriting it.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deletion successful
        """
        try:
            if not file_path.exists():
                return True
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Overwrite file with random data (3 passes)
            with open(file_path, 'r+b') as f:
                for _ in range(3):
                    f.seek(0)
                    f.write(secrets.token_bytes(file_size))
                    f.flush()
            
            # Delete the file
            file_path.unlink()
            
            logger.info(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Secure file deletion failed: {e}")
            return False
    
    def audit_data_access(self, 
                         subject_id: str,
                         access_type: str,
                         purpose: str,
                         user_id: Optional[str] = None) -> bool:
        """Log data access for audit purposes.
        
        Args:
            subject_id: Data subject identifier
            access_type: Type of access (read, write, delete, etc.)
            purpose: Purpose of access
            user_id: User performing the access
            
        Returns:
            True if logging successful
        """
        try:
            from datetime import datetime
            
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'subject_id': self.hash_identifier(subject_id),  # Hash for privacy
                'access_type': access_type,
                'purpose': purpose,
                'user_id': user_id,
                'session_id': secrets.token_hex(8)
            }
            
            # Log to audit file
            audit_file = self.key_storage_path / "data_access_audit.log"
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            return False
    
    def get_data_categories_for_subject(self, subject_id: str) -> List[str]:
        """Get data categories processed for a subject.
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            List of data category names
        """
        # This would typically query a database or registry
        # For now, return common EEG data categories
        return [
            "biometric",  # EEG signals
            "health",     # Medical information
            "demographic" # Age, gender, etc.
        ]
    
    def create_privacy_policy_summary(self) -> Dict[str, Any]:
        """Create a summary of privacy practices for transparency.
        
        Returns:
            Privacy policy summary
        """
        return {
            "data_controller": "BCI-GPT Research Team",
            "data_categories": [
                {
                    "category": "Biometric Data",
                    "description": "EEG brain signals and neural activity recordings",
                    "purpose": "Brain-computer interface research and model training",
                    "retention_period": "7 years"
                },
                {
                    "category": "Health Data", 
                    "description": "Medical history and health status information",
                    "purpose": "Safety assessment and research context",
                    "retention_period": "7 years"
                },
                {
                    "category": "Demographic Data",
                    "description": "Age, gender, education level",
                    "purpose": "Research analysis and model personalization", 
                    "retention_period": "3 years"
                }
            ],
            "legal_basis": "Informed consent for research purposes",
            "rights": [
                "Right to access your data",
                "Right to rectify incorrect data",
                "Right to erase your data",
                "Right to restrict processing",
                "Right to data portability",
                "Right to object to processing",
                "Right to withdraw consent"
            ],
            "security_measures": [
                "End-to-end encryption",
                "Access logging and monitoring",
                "Pseudonymization of identifiers",
                "Secure key management",
                "Regular security audits"
            ],
            "contact": {
                "data_protection_officer": "privacy@bci-gpt.org",
                "supervisory_authority": "Local Data Protection Authority"
            }
        }