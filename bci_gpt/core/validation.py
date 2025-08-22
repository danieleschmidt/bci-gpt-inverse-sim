"""Data validation and sanitization for BCI-GPT."""

import re
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..utils.logging_config import get_logger
from .error_handling import DataValidationError


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Any = None
    metadata: Dict[str, Any] = None


class DataValidator:
    """Comprehensive data validation system."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = get_logger(__name__)
        self.rules = {}
        self.sanitizers = {}
        
        # Register default rules
        self._register_default_rules()
        self._register_default_sanitizers()
    
    def _register_default_rules(self):
        """Register default validation rules."""
        self.rules.update({
            "eeg_data": [
                ValidationRule(
                    "shape_check",
                    lambda x: self._validate_eeg_shape(x),
                    "EEG data must have shape (channels, samples)"
                ),
                ValidationRule(
                    "value_range",
                    lambda x: self._validate_eeg_range(x),
                    "EEG values must be in valid microvolts range (-1000 to 1000)"
                ),
                ValidationRule(
                    "no_nan",
                    lambda x: not self._has_nan(x),
                    "EEG data cannot contain NaN values"
                )
            ],
            "text_input": [
                ValidationRule(
                    "length_check",
                    lambda x: isinstance(x, str) and 1 <= len(x) <= 1000,
                    "Text must be 1-1000 characters"
                ),
                ValidationRule(
                    "encoding_check",
                    lambda x: self._validate_encoding(x),
                    "Text must be valid UTF-8"
                ),
                ValidationRule(
                    "content_safety",
                    lambda x: self._validate_content_safety(x),
                    "Text contains potentially harmful content"
                )
            ],
            "model_config": [
                ValidationRule(
                    "required_fields",
                    lambda x: self._validate_config_fields(x),
                    "Missing required configuration fields"
                ),
                ValidationRule(
                    "value_types",
                    lambda x: self._validate_config_types(x),
                    "Invalid configuration value types"
                )
            ]
        })
    
    def _register_default_sanitizers(self):
        """Register default data sanitizers."""
        self.sanitizers.update({
            "eeg_data": self._sanitize_eeg_data,
            "text_input": self._sanitize_text_input,
            "model_config": self._sanitize_config
        })
    
    def validate(self, data: Any, data_type: str, sanitize: bool = True) -> ValidationResult:
        """Validate data according to rules."""
        errors = []
        warnings = []
        sanitized_data = data
        
        # Get validation rules for data type
        rules = self.rules.get(data_type, [])
        
        # Apply validation rules
        for rule in rules:
            try:
                is_valid = rule.validator(data)
                if not is_valid:
                    if rule.severity == "error":
                        errors.append(f"{rule.name}: {rule.error_message}")
                    elif rule.severity == "warning":
                        warnings.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{rule.name}: Validation failed - {str(e)}")
        
        # Sanitize data if requested and no critical errors
        if sanitize and not errors and data_type in self.sanitizers:
            try:
                sanitized_data = self.sanitizers[data_type](data)
                self.logger.log_info(f"Data sanitized for type: {data_type}")
            except Exception as e:
                warnings.append(f"Sanitization failed: {str(e)}")
                sanitized_data = data
        
        # Log results
        if errors:
            self.logger.log_error(f"Validation failed for {data_type}: {errors}")
        if warnings:
            self.logger.log_warning(f"Validation warnings for {data_type}: {warnings}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data,
            metadata={
                "data_type": data_type,
                "validation_timestamp": datetime.now(),
                "rules_applied": len(rules)
            }
        )
    
    def register_rule(self, data_type: str, rule: ValidationRule):
        """Register a new validation rule."""
        if data_type not in self.rules:
            self.rules[data_type] = []
        
        self.rules[data_type].append(rule)
        self.logger.log_info(f"Registered validation rule '{rule.name}' for {data_type}")
    
    def register_sanitizer(self, data_type: str, sanitizer: Callable):
        """Register a data sanitizer."""
        self.sanitizers[data_type] = sanitizer
        self.logger.log_info(f"Registered sanitizer for {data_type}")
    
    # EEG validation methods
    def _validate_eeg_shape(self, data: Any) -> bool:
        """Validate EEG data shape."""
        if not HAS_NUMPY:
            return isinstance(data, (list, tuple)) and len(data) > 0
        
        if isinstance(data, np.ndarray):
            return len(data.shape) == 2 and data.shape[0] > 0 and data.shape[1] > 0
        return False
    
    def _validate_eeg_range(self, data: Any) -> bool:
        """Validate EEG value range."""
        if not HAS_NUMPY:
            if isinstance(data, (list, tuple)):
                flat_data = self._flatten_list(data)
                return all(-1000 <= val <= 1000 for val in flat_data if isinstance(val, (int, float)))
            return True
        
        if isinstance(data, np.ndarray):
            return np.all((-1000 <= data) & (data <= 1000))
        return True
    
    def _has_nan(self, data: Any) -> bool:
        """Check for NaN values."""
        if not HAS_NUMPY:
            if isinstance(data, (list, tuple)):
                flat_data = self._flatten_list(data)
                return any(val != val for val in flat_data if isinstance(val, float))  # NaN check
            return False
        
        if isinstance(data, np.ndarray):
            return np.any(np.isnan(data))
        return False
    
    def _flatten_list(self, data: Any) -> List:
        """Flatten nested list structure."""
        result = []
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, (list, tuple)):
                    result.extend(self._flatten_list(item))
                else:
                    result.append(item)
        else:
            result.append(data)
        return result
    
    # Text validation methods
    def _validate_encoding(self, text: str) -> bool:
        """Validate text encoding."""
        try:
            text.encode('utf-8').decode('utf-8')
            return True
        except UnicodeError:
            return False
    
    def _validate_content_safety(self, text: str) -> bool:
        """Validate content safety."""
        # Basic safety checks
        dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'eval\s*\(',  # Eval calls
            r'exec\s*\(',  # Exec calls
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    # Config validation methods
    def _validate_config_fields(self, config: Dict[str, Any]) -> bool:
        """Validate required configuration fields."""
        required_fields = ['model_type', 'device']
        return all(field in config for field in required_fields)
    
    def _validate_config_types(self, config: Dict[str, Any]) -> bool:
        """Validate configuration value types."""
        type_expectations = {
            'batch_size': int,
            'learning_rate': (int, float),
            'epochs': int,
            'model_type': str,
            'device': str
        }
        
        for field, expected_type in type_expectations.items():
            if field in config:
                if not isinstance(config[field], expected_type):
                    return False
        
        return True
    
    # Sanitization methods
    def _sanitize_eeg_data(self, data: Any) -> Any:
        """Sanitize EEG data."""
        if not HAS_NUMPY:
            return data
        
        if isinstance(data, np.ndarray):
            # Remove NaN values
            data = np.nan_to_num(data, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            # Clip to valid range
            data = np.clip(data, -1000, 1000)
            
            return data
        
        return data
    
    def _sanitize_text_input(self, text: str) -> str:
        """Sanitize text input."""
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Limit length
        text = text[:1000]
        
        # Remove potentially dangerous content
        dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration."""
        sanitized = config.copy()
        
        # Ensure safe defaults
        if 'batch_size' in sanitized:
            sanitized['batch_size'] = max(1, min(1024, sanitized['batch_size']))
        
        if 'learning_rate' in sanitized:
            sanitized['learning_rate'] = max(1e-6, min(1.0, sanitized['learning_rate']))
        
        if 'epochs' in sanitized:
            sanitized['epochs'] = max(1, min(1000, sanitized['epochs']))
        
        # Remove unsafe fields
        unsafe_fields = ['__import__', 'eval', 'exec']
        for field in unsafe_fields:
            sanitized.pop(field, None)
        
        return sanitized


class InputSanitizer:
    """Specialized input sanitization."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.logger = get_logger(__name__)
    
    def sanitize_file_path(self, path: str) -> str:
        """Sanitize file path."""
        # Remove dangerous path components
        path = re.sub(r'\.\./', '', path)  # Remove path traversal
        path = re.sub(r'[<>:"|?*]', '', path)  # Remove invalid characters
        
        # Ensure path doesn't start with /
        if path.startswith('/'):
            path = path[1:]
        
        return path
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename."""
        # Remove path separators and dangerous characters
        filename = re.sub(r'[/\\<>:"|?*]', '_', filename)
        
        # Limit length
        filename = filename[:255]
        
        return filename
    
    def validate_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """Validate data checksum."""
        actual_checksum = hashlib.sha256(data).hexdigest()
        return actual_checksum == expected_checksum


# Global validator instance
global_validator = DataValidator()
global_sanitizer = InputSanitizer()