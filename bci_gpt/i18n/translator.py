"""Translation system for BCI-GPT internationalization."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
from functools import lru_cache
import warnings

from .locales import (
    DEFAULT_LOCALE, 
    SUPPORTED_LOCALES, 
    get_locale_dir, 
    detect_system_locale,
    get_available_locales
)

logger = logging.getLogger(__name__)

class Translator:
    """Translation manager for BCI-GPT."""
    
    def __init__(self, locale: Optional[str] = None, fallback_locale: str = DEFAULT_LOCALE):
        """Initialize translator.
        
        Args:
            locale: Target locale code (e.g., 'en_US', 'zh_CN')
            fallback_locale: Fallback locale when translation is missing
        """
        self.fallback_locale = fallback_locale
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._current_locale = locale or detect_system_locale()
        
        # Load default locale first
        self._load_locale(self.fallback_locale)
        
        # Load target locale if different
        if self._current_locale != self.fallback_locale:
            self._load_locale(self._current_locale)
    
    @property
    def current_locale(self) -> str:
        """Get current locale."""
        return self._current_locale
    
    @current_locale.setter 
    def current_locale(self, locale: str):
        """Set current locale and load translations."""
        if locale not in SUPPORTED_LOCALES:
            warnings.warn(f"Unsupported locale '{locale}', using {DEFAULT_LOCALE}")
            locale = DEFAULT_LOCALE
        
        self._current_locale = locale
        if locale not in self._translations:
            self._load_locale(locale)
    
    def _load_locale(self, locale: str) -> bool:
        """Load translations for a specific locale.
        
        Args:
            locale: Locale code to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            locale_file = get_locale_dir() / f"{locale}.json"
            
            if not locale_file.exists():
                if locale == DEFAULT_LOCALE:
                    # Create default English translations
                    self._create_default_translations(locale_file)
                else:
                    logger.warning(f"Translation file not found: {locale_file}")
                    return False
            
            with open(locale_file, 'r', encoding='utf-8') as f:
                self._translations[locale] = json.load(f)
            
            logger.info(f"Loaded translations for locale: {locale}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load locale {locale}: {e}")
            return False
    
    def _create_default_translations(self, locale_file: Path):
        """Create default English translation file."""
        default_translations = {
            "common": {
                "yes": "Yes",
                "no": "No", 
                "ok": "OK",
                "cancel": "Cancel",
                "error": "Error",
                "warning": "Warning",
                "info": "Information",
                "success": "Success",
                "loading": "Loading...",
                "please_wait": "Please wait...",
                "retry": "Retry",
                "close": "Close"
            },
            "bci": {
                "eeg_signal": "EEG Signal",
                "channels": "Channels", 
                "sampling_rate": "Sampling Rate",
                "signal_quality": "Signal Quality",
                "recording": "Recording",
                "processing": "Processing",
                "decoding": "Decoding",
                "imagined_speech": "Imagined Speech",
                "brain_computer_interface": "Brain-Computer Interface",
                "neural_signal": "Neural Signal",
                "electrode": "Electrode",
                "artifact": "Artifact",
                "filter": "Filter",
                "calibration": "Calibration"
            },
            "model": {
                "training": "Training",
                "validation": "Validation", 
                "testing": "Testing",
                "epoch": "Epoch",
                "batch": "Batch",
                "loss": "Loss",
                "accuracy": "Accuracy",
                "learning_rate": "Learning Rate",
                "model_loading": "Loading model...",
                "model_saving": "Saving model...",
                "inference": "Inference",
                "prediction": "Prediction",
                "confidence": "Confidence"
            },
            "errors": {
                "file_not_found": "File not found: {filename}",
                "invalid_format": "Invalid file format",
                "processing_failed": "Processing failed: {error}",
                "model_load_failed": "Failed to load model: {error}",
                "insufficient_data": "Insufficient data for processing",
                "connection_failed": "Connection failed",
                "timeout": "Operation timed out",
                "permission_denied": "Permission denied",
                "out_of_memory": "Out of memory",
                "gpu_not_available": "GPU not available"
            },
            "cli": {
                "train_command": "Train BCI-GPT model",
                "decode_command": "Decode EEG signals to text",
                "generate_command": "Generate synthetic EEG from text",
                "serve_command": "Start BCI-GPT API server",
                "info_command": "Show system information",
                "config_file": "Configuration file path",
                "model_path": "Path to model file",
                "data_path": "Path to data directory",
                "output_path": "Output file path",
                "verbose": "Enable verbose output",
                "quiet": "Suppress output messages"
            },
            "api": {
                "server_starting": "Starting BCI-GPT server...",
                "server_ready": "Server ready on {host}:{port}",
                "server_stopping": "Stopping server...",
                "invalid_request": "Invalid request format",
                "processing_request": "Processing request...",
                "request_completed": "Request completed successfully",
                "health_check": "Health check",
                "status_healthy": "System is healthy",
                "status_unhealthy": "System is unhealthy"
            }
        }
        
        # Create locale directory if it doesn't exist
        locale_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write default translations
        with open(locale_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, ensure_ascii=False, indent=2)
    
    def t(self, key: str, **kwargs) -> str:
        """Translate a key with optional formatting.
        
        Args:
            key: Translation key (e.g., 'common.error', 'bci.eeg_signal')
            **kwargs: Format arguments for string interpolation
            
        Returns:
            Translated string
        """
        return self.translate(key, **kwargs)
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a key with optional formatting.
        
        Args:
            key: Translation key (dot-separated, e.g., 'common.error')
            locale: Override locale for this translation
            **kwargs: Format arguments for string interpolation
            
        Returns:
            Translated string
        """
        target_locale = locale or self._current_locale
        
        # Try to get translation from target locale
        text = self._get_translation(key, target_locale)
        
        # Fall back to default locale if not found
        if text is None and target_locale != self.fallback_locale:
            text = self._get_translation(key, self.fallback_locale)
        
        # Fall back to key itself if still not found
        if text is None:
            logger.warning(f"Missing translation for key: {key}")
            text = key.split('.')[-1].replace('_', ' ').title()
        
        # Apply formatting if provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting failed for key {key}: {e}")
        
        return text
    
    def _get_translation(self, key: str, locale: str) -> Optional[str]:
        """Get translation for a key from specific locale.
        
        Args:
            key: Translation key (dot-separated)
            locale: Locale code
            
        Returns:
            Translation string or None if not found
        """
        if locale not in self._translations:
            return None
        
        # Navigate nested dictionary using dot notation
        value = self._translations[locale]
        
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value if isinstance(value, str) else None
    
    def has_translation(self, key: str, locale: Optional[str] = None) -> bool:
        """Check if translation exists for a key.
        
        Args:
            key: Translation key
            locale: Locale to check (default: current locale)
            
        Returns:
            True if translation exists
        """
        target_locale = locale or self._current_locale
        return self._get_translation(key, target_locale) is not None
    
    def get_available_keys(self, locale: Optional[str] = None) -> list:
        """Get all available translation keys for a locale.
        
        Args:
            locale: Locale to check (default: current locale)
            
        Returns:
            List of available keys
        """
        target_locale = locale or self._current_locale
        
        if target_locale not in self._translations:
            return []
        
        return self._flatten_keys(self._translations[target_locale])
    
    def _flatten_keys(self, data: dict, prefix: str = "") -> list:
        """Flatten nested dictionary keys with dot notation."""
        keys = []
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
            else:
                keys.append(full_key)
        
        return keys
    
    def reload_translations(self):
        """Reload all translations from files."""
        self._translations.clear()
        self._load_locale(self.fallback_locale)
        
        if self._current_locale != self.fallback_locale:
            self._load_locale(self._current_locale)

# Global translator instance
_global_translator: Optional[Translator] = None

def get_translator(locale: Optional[str] = None) -> Translator:
    """Get global translator instance.
    
    Args:
        locale: Locale to use (default: auto-detect)
        
    Returns:
        Translator instance
    """
    global _global_translator
    
    if _global_translator is None or (locale and locale != _global_translator.current_locale):
        _global_translator = Translator(locale)
    
    return _global_translator

@lru_cache(maxsize=1000)
def _(key: str, **kwargs) -> str:
    """Convenience function for translation (cached).
    
    Args:
        key: Translation key
        **kwargs: Format arguments
        
    Returns:
        Translated string
    """
    return get_translator().translate(key, **kwargs)