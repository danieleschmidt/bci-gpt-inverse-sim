"""Lightweight configuration manager without YAML dependency."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class LightweightConfigManager:
    """Lightweight configuration manager using JSON instead of YAML."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config_data = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Try several locations
        possible_paths = [
            "./bci_gpt_config.json",
            "~/.bci_gpt/config.json",
            "/etc/bci_gpt/config.json"
        ]
        
        for path in possible_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                return str(expanded_path)
        
        # Return first option as default
        return possible_paths[0]
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        config_path = Path(self.config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.config_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # Use default configuration on error
                self.config_data = self._get_default_config()
        else:
            # Use default configuration
            self.config_data = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "model": {
                "eeg_channels": 9,
                "sampling_rate": 1000,
                "sequence_length": 1000,
                "hidden_dim": 512,
                "n_layers": 6,
                "n_heads": 8,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "epochs": 100,
                "early_stopping_patience": 10
            },
            "preprocessing": {
                "bandpass_low": 0.5,
                "bandpass_high": 40.0,
                "artifact_removal": "ica",
                "reference": "average"
            },
            "deployment": {
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "workers": 1,
                "timeout": 30
            },
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console_logging": True
            },
            "security": {
                "enable_encryption": True,
                "api_key_required": True,
                "rate_limit": 100
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.hidden_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self.config_data
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.hidden_dim')
            value: Value to set
        """
        keys = key.split('.')
        current = self.config_data
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to current config path)
        """
        save_path = path or self.config_path
        config_dir = Path(save_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'model', 'training')
            
        Returns:
            Dictionary containing section configuration
        """
        return self.config_data.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section with new values.
        
        Args:
            section: Section name
            updates: Dictionary of updates to apply
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section].update(updates)


# Global instance
_config_manager = None


def get_config_manager() -> LightweightConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = LightweightConfigManager()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Convenience function to get configuration value."""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """Convenience function to set configuration value."""
    get_config_manager().set(key, value)