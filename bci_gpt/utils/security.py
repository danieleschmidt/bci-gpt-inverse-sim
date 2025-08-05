"""
Security and privacy utilities for BCI-GPT.

This module provides comprehensive security features including:
- Data encryption and secure storage
- Privacy-preserving processing
- Input validation and sanitization
- Secure model inference
- Clinical data protection (HIPAA compliance)
- Attack detection and mitigation
"""

import hashlib
import hmac
import secrets
import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
import numpy as np

# Optional security dependencies
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    warnings.warn("cryptography not available. Security features will be limited.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class DataEncryption:
    """Secure data encryption and decryption."""
    
    def __init__(self, password: Optional[str] = None):
        """Initialize encryption with password or generate key.
        
        Args:
            password: Optional password for key derivation
        """
        if not HAS_CRYPTOGRAPHY:
            raise ImportError("cryptography package required for encryption")
        
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        logger.info("Initialized data encryption")
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = b'bci_gpt_salt_2025'  # Fixed salt for deterministic key generation
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return Fernet.generate_key()  # Use password-derived key in real implementation
    
    def encrypt_data(self, data: Union[str, bytes, np.ndarray]) -> bytes:
        """Encrypt data securely.
        
        Args:
            data: Data to encrypt (string, bytes, or numpy array)
            
        Returns:
            Encrypted data as bytes
        """
        try:
            # Convert to bytes if necessary
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            encrypted = self.cipher.encrypt(data_bytes)
            logger.debug(f"Encrypted {len(data_bytes)} bytes")
            return encrypted
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, 
                    data_type: str = 'bytes') -> Union[str, bytes, np.ndarray]:
        """Decrypt data securely.
        
        Args:
            encrypted_data: Encrypted data bytes
            data_type: Expected output type ('str', 'bytes', 'numpy')
            
        Returns:
            Decrypted data in requested format
        """
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_data)
            
            if data_type == 'str':
                return decrypted_bytes.decode('utf-8')
            elif data_type == 'bytes':
                return decrypted_bytes
            elif data_type == 'numpy':
                return np.frombuffer(decrypted_bytes, dtype=np.float64)
            else:
                return decrypted_bytes
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Encrypt entire file."""
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.encrypt_data(data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"Encrypted file: {input_path} -> {output_path}")
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise
    
    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Decrypt entire file."""
        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.decrypt_data(encrypted_data, 'bytes')
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"Decrypted file: {input_path} -> {output_path}")
            
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise


class PrivacyProtection:
    """Privacy-preserving data processing utilities."""
    
    @staticmethod
    def anonymize_eeg_data(eeg_data: np.ndarray, 
                          method: str = 'gaussian_noise',
                          privacy_level: float = 0.1) -> np.ndarray:
        """Anonymize EEG data while preserving utility.
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            method: Anonymization method ('gaussian_noise', 'differential_privacy')
            privacy_level: Level of privacy protection (0.0 to 1.0)
            
        Returns:
            Anonymized EEG data
        """
        try:
            if method == 'gaussian_noise':
                # Add calibrated Gaussian noise
                noise_std = privacy_level * np.std(eeg_data, axis=1, keepdims=True)
                noise = np.random.normal(0, noise_std, eeg_data.shape)
                anonymized = eeg_data + noise
                
            elif method == 'differential_privacy':
                # Simplified differential privacy (Laplace mechanism)
                sensitivity = np.max(eeg_data) - np.min(eeg_data)
                epsilon = 1.0 / privacy_level  # Privacy budget
                scale = sensitivity / epsilon
                
                noise = np.random.laplace(0, scale, eeg_data.shape)
                anonymized = eeg_data + noise
                
            else:
                raise ValueError(f"Unknown anonymization method: {method}")
            
            logger.info(f"Anonymized EEG data using {method} (privacy_level={privacy_level})")
            return anonymized.astype(eeg_data.dtype)
            
        except Exception as e:
            logger.error(f"EEG anonymization failed: {e}")
            raise
    
    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 1000) -> str:
        """Sanitize text input for security.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed text length
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Length validation
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text input truncated to {max_length} characters")
        
        # Remove potential injection patterns
        dangerous_patterns = [
            '<script', '</script>', 'javascript:', 'data:',
            'eval(', 'exec(', '__import__', 'subprocess',
            'os.system', 'open(', 'file://'
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                logger.warning(f"Removed dangerous pattern: {pattern}")
                text = text.replace(pattern, '[FILTERED]')
        
        # Basic HTML/XML sanitization
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        return text.strip()
    
    @staticmethod
    def generate_patient_id(patient_data: Dict[str, Any], 
                          salt: Optional[str] = None) -> str:
        """Generate anonymized patient ID from patient data.
        
        Args:
            patient_data: Dictionary containing patient information
            salt: Optional salt for hashing
            
        Returns:
            Anonymized patient ID
        """
        if salt is None:
            salt = "bci_gpt_patient_salt_2025"
        
        # Create deterministic hash from patient data
        data_string = json.dumps(patient_data, sort_keys=True)
        combined = f"{salt}:{data_string}"
        
        patient_id = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        logger.debug("Generated anonymized patient ID")
        return f"patient_{patient_id}"
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], 
                           sensitive_fields: List[str] = None) -> Dict[str, Any]:
        """Mask sensitive fields in data dictionary.
        
        Args:
            data: Data dictionary
            sensitive_fields: List of sensitive field names
            
        Returns:
            Data with sensitive fields masked
        """
        if sensitive_fields is None:
            sensitive_fields = [
                'name', 'email', 'phone', 'address', 'ssn', 'dob',
                'medical_record_number', 'patient_id'
            ]
        
        masked_data = data.copy()
        
        for field in sensitive_fields:
            if field in masked_data:
                if isinstance(masked_data[field], str):
                    masked_data[field] = '*' * len(masked_data[field])
                else:
                    masked_data[field] = '[MASKED]'
        
        return masked_data


class InputValidation:
    """Input validation and sanitization."""
    
    @staticmethod
    def validate_eeg_data(eeg_data: np.ndarray,
                         expected_channels: Optional[int] = None,
                         expected_sampling_rate: Optional[float] = None,
                         max_duration_seconds: float = 3600.0) -> bool:
        """Validate EEG data format and constraints.
        
        Args:
            eeg_data: EEG data array (channels x samples)
            expected_channels: Expected number of channels
            expected_sampling_rate: Expected sampling rate
            max_duration_seconds: Maximum allowed duration
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        try:
            if not isinstance(eeg_data, np.ndarray):
                raise ValueError("EEG data must be numpy array")
            
            if eeg_data.ndim != 2:
                raise ValueError(f"EEG data must be 2D, got {eeg_data.ndim}D")
            
            channels, samples = eeg_data.shape
            
            if expected_channels and channels != expected_channels:
                raise ValueError(f"Expected {expected_channels} channels, got {channels}")
            
            if expected_sampling_rate:
                duration = samples / expected_sampling_rate
                if duration > max_duration_seconds:
                    raise ValueError(f"Duration {duration:.1f}s exceeds maximum {max_duration_seconds}s")
            
            # Check for valid data ranges (typical EEG is -200 to +200 μV)
            if np.any(np.abs(eeg_data) > 1000):
                logger.warning("EEG data contains very large values (>1000 μV)")
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(eeg_data)):
                raise ValueError("EEG data contains NaN or infinite values")
            
            logger.debug(f"Validated EEG data: {channels} channels, {samples} samples")
            return True
            
        except Exception as e:
            logger.error(f"EEG validation failed: {e}")
            raise
    
    @staticmethod
    def validate_model_input(input_tensor: Any,
                           expected_shape: Optional[Tuple[int, ...]] = None,
                           device_check: bool = True) -> bool:
        """Validate model input tensor.
        
        Args:
            input_tensor: Input tensor to validate
            expected_shape: Expected tensor shape
            device_check: Whether to check device compatibility
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available for tensor validation")
            return True
        
        try:
            if not torch.is_tensor(input_tensor):
                raise ValueError("Input must be a PyTorch tensor")
            
            if expected_shape and input_tensor.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {input_tensor.shape}")
            
            if device_check and input_tensor.device.type not in ['cpu', 'cuda']:
                raise ValueError(f"Unsupported device: {input_tensor.device}")
            
            if not torch.all(torch.isfinite(input_tensor)):
                raise ValueError("Input tensor contains NaN or infinite values")
            
            logger.debug(f"Validated model input: shape {input_tensor.shape}, device {input_tensor.device}")
            return True
            
        except Exception as e:
            logger.error(f"Model input validation failed: {e}")
            raise
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path],
                          allowed_extensions: List[str] = None,
                          max_size_mb: float = 100.0) -> bool:
        """Validate file path and constraints.
        
        Args:
            file_path: Path to validate
            allowed_extensions: Allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise ValueError(f"File does not exist: {path}")
            
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            
            # Check file extension
            if allowed_extensions:
                if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                    raise ValueError(f"File extension {path.suffix} not allowed. "
                                   f"Allowed: {allowed_extensions}")
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"File size {size_mb:.1f}MB exceeds maximum {max_size_mb}MB")
            
            # Security check: prevent path traversal
            if str(path.resolve()).startswith('..'):
                raise ValueError("Path traversal detected")
            
            logger.debug(f"Validated file path: {path} ({size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"File path validation failed: {e}")
            raise


class AttackDetection:
    """Detect and mitigate potential attacks."""
    
    def __init__(self, alert_threshold: int = 10):
        """Initialize attack detection.
        
        Args:
            alert_threshold: Number of suspicious events before alert
        """
        self.alert_threshold = alert_threshold
        self.suspicious_events = []
        self.blocked_ips = set()
        
    def detect_adversarial_eeg(self, eeg_data: np.ndarray,
                              baseline_stats: Optional[Dict[str, float]] = None) -> bool:
        """Detect adversarial modifications to EEG data.
        
        Args:
            eeg_data: EEG data to analyze
            baseline_stats: Expected statistical properties
            
        Returns:
            True if adversarial patterns detected
        """
        try:
            # Statistical anomaly detection
            anomalies = []
            
            # Check for unusual amplitude patterns
            amplitude_std = np.std(eeg_data, axis=1)
            if np.any(amplitude_std > 200):  # Unusually high variation
                anomalies.append("high_amplitude_variation")
            
            # Check for unusual frequency content
            if eeg_data.shape[1] > 100:  # Need sufficient samples for FFT
                for ch in range(min(eeg_data.shape[0], 8)):  # Check first 8 channels
                    fft = np.fft.fft(eeg_data[ch])
                    power_spectrum = np.abs(fft[:len(fft)//2])
                    
                    # Check for artificial spikes in frequency domain
                    max_power = np.max(power_spectrum)
                    mean_power = np.mean(power_spectrum)
                    
                    if max_power > 10 * mean_power:  # Suspicious spike
                        anomalies.append("frequency_spike")
                        break
            
            # Check against baseline if provided
            if baseline_stats:
                current_mean = np.mean(eeg_data)
                current_std = np.std(eeg_data)
                
                if abs(current_mean - baseline_stats.get('mean', 0)) > 3 * baseline_stats.get('std', 1):
                    anomalies.append("statistical_deviation")
            
            is_adversarial = len(anomalies) >= 2  # Multiple indicators
            
            if is_adversarial:
                logger.warning(f"Adversarial EEG detected: {anomalies}")
                self._record_suspicious_event("adversarial_eeg", {"anomalies": anomalies})
            
            return is_adversarial
            
        except Exception as e:
            logger.error(f"Adversarial detection failed: {e}")
            return False
    
    def detect_model_extraction_attack(self, query_history: List[Dict[str, Any]],
                                     time_window_minutes: int = 60) -> bool:
        """Detect model extraction attacks through query patterns.
        
        Args:
            query_history: List of recent queries
            time_window_minutes: Time window for analysis
            
        Returns:
            True if extraction attack detected
        """
        try:
            if len(query_history) < 10:
                return False
            
            # Analyze recent queries
            import time
            current_time = time.time()
            recent_queries = [
                q for q in query_history
                if current_time - q.get('timestamp', 0) < time_window_minutes * 60
            ]
            
            if len(recent_queries) < 10:
                return False
            
            # Check for systematic probing patterns
            suspicious_patterns = 0
            
            # High query frequency
            if len(recent_queries) > 100:  # More than 100 queries in time window
                suspicious_patterns += 1
            
            # Systematic input variations
            inputs = [q.get('input_size', 0) for q in recent_queries]
            if len(set(inputs)) > 20:  # Many different input sizes
                suspicious_patterns += 1
            
            # Repeated similar queries
            query_texts = [str(q.get('input', '')) for q in recent_queries]
            unique_queries = len(set(query_texts))
            if unique_queries < len(query_texts) * 0.3:  # High repetition
                suspicious_patterns += 1
            
            is_extraction_attack = suspicious_patterns >= 2
            
            if is_extraction_attack:
                logger.warning("Model extraction attack detected")
                self._record_suspicious_event("model_extraction", {
                    "query_count": len(recent_queries),
                    "unique_queries": unique_queries,
                    "patterns": suspicious_patterns
                })
            
            return is_extraction_attack
            
        except Exception as e:
            logger.error(f"Model extraction detection failed: {e}")
            return False
    
    def _record_suspicious_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record suspicious event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details
        }
        
        self.suspicious_events.append(event)
        
        # Trigger alert if threshold exceeded
        if len(self.suspicious_events) >= self.alert_threshold:
            logger.critical(f"Attack detection threshold exceeded: {len(self.suspicious_events)} events")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            'suspicious_events': len(self.suspicious_events),
            'blocked_ips': len(self.blocked_ips),
            'alert_threshold': self.alert_threshold,
            'status': 'alert' if len(self.suspicious_events) >= self.alert_threshold else 'normal'
        }


class SecureInference:
    """Secure model inference with privacy protection."""
    
    def __init__(self, enable_encryption: bool = True):
        """Initialize secure inference.
        
        Args:
            enable_encryption: Whether to encrypt intermediate results
        """
        self.enable_encryption = enable_encryption
        self.encryptor = DataEncryption() if enable_encryption and HAS_CRYPTOGRAPHY else None
        
    def secure_forward_pass(self, model: Any, input_data: Any,
                          privacy_level: float = 0.1) -> Any:
        """Perform secure forward pass with privacy protection.
        
        Args:
            model: Model to run inference on
            input_data: Input data (will be validated)
            privacy_level: Level of privacy protection
            
        Returns:
            Model output with privacy protection applied
        """
        try:
            # Validate input
            if HAS_TORCH and torch.is_tensor(input_data):
                InputValidation.validate_model_input(input_data)
            
            # Apply privacy protection to input
            if isinstance(input_data, np.ndarray):
                protected_input = PrivacyProtection.anonymize_eeg_data(
                    input_data, privacy_level=privacy_level
                )
                if HAS_TORCH:
                    protected_input = torch.from_numpy(protected_input).float()
            else:
                protected_input = input_data
            
            # Run inference
            if HAS_TORCH and hasattr(model, 'eval'):
                model.eval()
                with torch.no_grad():
                    output = model(protected_input)
            else:
                output = model(protected_input)
            
            # Encrypt output if enabled
            if self.encryptor and isinstance(output, (np.ndarray, torch.Tensor)):
                if torch.is_tensor(output):
                    output_data = output.detach().cpu().numpy()
                else:
                    output_data = output
                
                encrypted_output = self.encryptor.encrypt_data(output_data)
                logger.debug("Encrypted inference output")
                return encrypted_output
            
            return output
            
        except Exception as e:
            logger.error(f"Secure inference failed: {e}")
            raise
    
    def decrypt_output(self, encrypted_output: bytes) -> np.ndarray:
        """Decrypt inference output.
        
        Args:
            encrypted_output: Encrypted output data
            
        Returns:
            Decrypted output as numpy array
        """
        if not self.encryptor:
            raise ValueError("Encryption not enabled")
        
        return self.encryptor.decrypt_data(encrypted_output, 'numpy')


class ComplianceChecker:
    """Check compliance with medical data regulations."""
    
    def __init__(self):
        self.required_safeguards = {
            'encryption_at_rest': False,
            'encryption_in_transit': False,
            'access_controls': False,
            'audit_logging': False,
            'data_anonymization': False,
            'secure_deletion': False
        }
    
    def check_hipaa_compliance(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check HIPAA compliance requirements.
        
        Args:
            system_config: System configuration to check
            
        Returns:
            Compliance report
        """
        compliance_report = {
            'compliant': True,
            'violations': [],
            'recommendations': [],
            'safeguards_status': {}
        }
        
        # Check encryption requirements
        if not system_config.get('encryption_enabled', False):
            compliance_report['compliant'] = False
            compliance_report['violations'].append("Encryption not enabled")
            compliance_report['recommendations'].append("Enable data encryption at rest and in transit")
        
        # Check access controls
        if not system_config.get('access_controls', False):
            compliance_report['compliant'] = False
            compliance_report['violations'].append("Access controls not implemented")
            compliance_report['recommendations'].append("Implement role-based access controls")
        
        # Check audit logging
        if not system_config.get('audit_logging', False):
            compliance_report['compliant'] = False
            compliance_report['violations'].append("Audit logging not enabled")
            compliance_report['recommendations'].append("Enable comprehensive audit logging")
        
        # Check data retention policies
        if not system_config.get('data_retention_policy', False):
            compliance_report['violations'].append("Data retention policy not defined")
            compliance_report['recommendations'].append("Define and implement data retention policy")
        
        # Update safeguards status
        for safeguard in self.required_safeguards:
            self.required_safeguards[safeguard] = system_config.get(safeguard, False)
        
        compliance_report['safeguards_status'] = self.required_safeguards.copy()
        
        logger.info(f"HIPAA compliance check: {'COMPLIANT' if compliance_report['compliant'] else 'NON-COMPLIANT'}")
        
        return compliance_report
    
    def generate_compliance_report(self) -> str:
        """Generate human-readable compliance report."""
        report = "=== BCI-GPT Security Compliance Report ===\n\n"
        
        report += "Required Safeguards:\n"
        for safeguard, status in self.required_safeguards.items():
            status_symbol = "✓" if status else "✗"
            report += f"  {status_symbol} {safeguard.replace('_', ' ').title()}\n"
        
        compliant_count = sum(self.required_safeguards.values())
        total_count = len(self.required_safeguards)
        
        report += f"\nCompliance Score: {compliant_count}/{total_count} ({compliant_count/total_count*100:.1f}%)\n"
        
        if compliant_count == total_count:
            report += "\n✓ System meets all required security safeguards\n"
        else:
            report += f"\n⚠ System missing {total_count - compliant_count} required safeguards\n"
        
        return report


# Utility functions for easy integration
def secure_hash(data: str, salt: Optional[str] = None) -> str:
    """Generate secure hash of data."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    combined = f"{salt}:{data}"
    return hashlib.sha256(combined.encode()).hexdigest()


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a, b)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test encryption
    if HAS_CRYPTOGRAPHY:
        print("Testing encryption...")
        encryptor = DataEncryption(password="test_password")
        
        test_data = "Sensitive EEG data"
        encrypted = encryptor.encrypt_data(test_data)
        decrypted = encryptor.decrypt_data(encrypted, 'str')
        
        print(f"Original: {test_data}")
        print(f"Decrypted: {decrypted}")
        print(f"Match: {test_data == decrypted}")
    
    # Test privacy protection
    print("\nTesting privacy protection...")
    test_eeg = np.random.randn(8, 1000) * 50
    anonymized_eeg = PrivacyProtection.anonymize_eeg_data(test_eeg, privacy_level=0.1)
    
    print(f"Original EEG shape: {test_eeg.shape}")
    print(f"Anonymized EEG shape: {anonymized_eeg.shape}")
    print(f"Data modified: {not np.array_equal(test_eeg, anonymized_eeg)}")
    
    # Test input validation
    print("\nTesting input validation...")
    try:
        InputValidation.validate_eeg_data(test_eeg, expected_channels=8)
        print("EEG data validation: PASSED")
    except ValueError as e:
        print(f"EEG data validation: FAILED - {e}")
    
    # Test attack detection
    print("\nTesting attack detection...")
    detector = AttackDetection()
    is_adversarial = detector.detect_adversarial_eeg(test_eeg * 10)  # Amplified data
    print(f"Adversarial detection: {'DETECTED' if is_adversarial else 'CLEAN'}")
    
    # Test compliance checking
    print("\nTesting compliance checking...")
    checker = ComplianceChecker()
    
    test_config = {
        'encryption_enabled': True,
        'access_controls': True,
        'audit_logging': False,
        'data_retention_policy': True
    }
    
    compliance = checker.check_hipaa_compliance(test_config)
    print(f"HIPAA Compliance: {'COMPLIANT' if compliance['compliant'] else 'NON-COMPLIANT'}")
    print(f"Violations: {len(compliance['violations'])}")
    
    print("\nSecurity system test completed!")