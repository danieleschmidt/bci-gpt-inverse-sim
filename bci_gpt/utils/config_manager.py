"""Configuration management system for BCI-GPT with validation and hot-reloading."""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field, asdict
import threading
import time
from datetime import datetime

from .logging_config import get_logger
from .error_handling import ConfigurationError, robust_function
from .security import sanitize_text_input, validate_file_path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str = "bci-gpt"
    model_type: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_sequence_length: int = 512
    vocab_size: int = 50000
    dropout_prob: float = 0.1
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler: str = "linear"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    dataloader_num_workers: int = 4
    gradient_accumulation_steps: int = 1
    use_tensorboard: bool = True
    use_wandb: bool = False


@dataclass
class EEGConfig:
    """EEG processing configuration."""
    sampling_rate: int = 1000
    num_channels: int = 64
    window_size: float = 2.0
    overlap: float = 0.5
    filter_low: float = 0.5
    filter_high: float = 100.0
    notch_freq: float = 60.0
    reference_method: str = "average"
    artifact_removal: bool = True
    normalization: str = "zscore"
    channels: List[str] = field(default_factory=lambda: [])


@dataclass
class StreamingConfig:
    """Real-time streaming configuration."""
    backend: str = "simulated"
    buffer_size: int = 1000
    chunk_size: int = 32
    timeout: float = 1.0
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    data_validation: bool = True
    quality_threshold: float = 0.8


@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    encryption_enabled: bool = True
    hash_algorithm: str = "sha256"
    key_derivation: str = "pbkdf2"
    max_file_size_mb: float = 100.0
    allowed_file_extensions: List[str] = field(default_factory=lambda: [".npy", ".mat", ".edf"])
    input_sanitization: bool = True
    audit_logging: bool = True
    session_timeout: int = 3600
    max_concurrent_sessions: int = 10


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    metrics_enabled: bool = True
    metrics_interval: float = 60.0
    performance_profiling: bool = True
    health_checks: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_threshold": 80.0,
        "memory_threshold": 85.0,
        "error_rate_threshold": 0.05
    })


@dataclass
class BCIGPTConfig:
    """Main configuration container."""
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eeg: EEGConfig = field(default_factory=EEGConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    version: str = "1.0.0"
    debug: bool = False
    data_dir: str = "./data"
    model_dir: str = "./models"
    cache_dir: str = "./cache"
    temp_dir: str = "./tmp"
    
    # Environment-specific overrides
    environment: str = "development"  # development, staging, production
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigValidator:
    """Configuration validation system."""
    
    def __init__(self):
        self.logger = get_logger()
        self.validation_rules = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules."""
        
        # Model validation
        self.validation_rules["model.hidden_size"] = lambda x: isinstance(x, int) and x > 0
        self.validation_rules["model.num_layers"] = lambda x: isinstance(x, int) and 1 <= x <= 50
        self.validation_rules["model.dropout_prob"] = lambda x: isinstance(x, float) and 0.0 <= x <= 1.0
        self.validation_rules["model.device"] = lambda x: x in ["cpu", "cuda", "auto"]
        
        # Training validation
        self.validation_rules["training.learning_rate"] = lambda x: isinstance(x, float) and x > 0
        self.validation_rules["training.batch_size"] = lambda x: isinstance(x, int) and x > 0
        self.validation_rules["training.num_epochs"] = lambda x: isinstance(x, int) and x > 0
        self.validation_rules["training.max_grad_norm"] = lambda x: isinstance(x, float) and x > 0
        
        # EEG validation
        self.validation_rules["eeg.sampling_rate"] = lambda x: isinstance(x, int) and x > 0
        self.validation_rules["eeg.num_channels"] = lambda x: isinstance(x, int) and x > 0
        self.validation_rules["eeg.window_size"] = lambda x: isinstance(x, float) and x > 0
        self.validation_rules["eeg.filter_low"] = lambda x: isinstance(x, float) and x >= 0
        self.validation_rules["eeg.filter_high"] = lambda x: isinstance(x, float) and x > 0
        
        # Security validation
        self.validation_rules["security.max_file_size_mb"] = lambda x: isinstance(x, float) and x > 0
        self.validation_rules["security.session_timeout"] = lambda x: isinstance(x, int) and x > 0
        self.validation_rules["security.max_concurrent_sessions"] = lambda x: isinstance(x, int) and x > 0
    
    def validate_config(self, config: BCIGPTConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        config_dict = self._flatten_dict(asdict(config))
        
        for path, rule in self.validation_rules.items():
            if path in config_dict:
                value = config_dict[path]
                try:
                    if not rule(value):
                        errors.append(f"Validation failed for {path}: {value}")
                except Exception as e:
                    errors.append(f"Validation error for {path}: {str(e)}")
        
        # Custom validations
        errors.extend(self._validate_relationships(config))
        
        return errors
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary with dot notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _validate_relationships(self, config: BCIGPTConfig) -> List[str]:
        """Validate relationships between configuration parameters."""
        errors = []
        
        # EEG filter validation
        if config.eeg.filter_low >= config.eeg.filter_high:
            errors.append("EEG filter_low must be less than filter_high")
        
        # Nyquist frequency validation
        nyquist = config.eeg.sampling_rate / 2
        if config.eeg.filter_high >= nyquist:
            errors.append(f"EEG filter_high ({config.eeg.filter_high}) must be less than Nyquist frequency ({nyquist})")
        
        # Training batch size vs gradient accumulation
        effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
        if effective_batch_size > 128:
            errors.append(f"Effective batch size ({effective_batch_size}) may be too large")
        
        # Model size validation
        if config.model.hidden_size % config.model.num_attention_heads != 0:
            errors.append("Model hidden_size must be divisible by num_attention_heads")
        
        # Directory validation
        for dir_attr in ['data_dir', 'model_dir', 'cache_dir', 'temp_dir']:
            dir_path = getattr(config, dir_attr)
            if not isinstance(dir_path, str) or not dir_path.strip():
                errors.append(f"Invalid directory path for {dir_attr}")
        
        return errors


class ConfigManager:
    """Configuration management system with hot-reloading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[BCIGPTConfig] = None
        self.validator = ConfigValidator()
        self.logger = get_logger()
        
        # Hot-reload functionality
        self.hot_reload_enabled = False
        self.reload_thread: Optional[threading.Thread] = None
        self.reload_lock = threading.RLock()
        self.last_modified: Optional[float] = None
        self.reload_callbacks: List[callable] = []
        
        # Load initial configuration
        self.load_config()
    
    @robust_function(retry_count=3, exceptions=(ConfigurationError, FileNotFoundError))
    def load_config(self) -> BCIGPTConfig:
        """Load configuration from file or create default."""
        with self.reload_lock:
            try:
                if self.config_path and os.path.exists(self.config_path):
                    config_dict = self._load_config_file(self.config_path)
                    self.config = self._dict_to_config(config_dict)
                    self.last_modified = os.path.getmtime(self.config_path)
                    self.logger.log_info(f"Loaded configuration from {self.config_path}")
                else:
                    # Create default configuration
                    self.config = BCIGPTConfig()
                    self.logger.log_info("Using default configuration")
                
                # Apply environment variables
                self._apply_env_overrides()
                
                # Validate configuration
                errors = self.validator.validate_config(self.config)
                if errors:
                    error_msg = f"Configuration validation errors: {errors}"
                    raise ConfigurationError(error_msg)
                
                # Ensure directories exist
                self._create_directories()
                
                return self.config
                
            except Exception as e:
                self.logger.log_error("Failed to load configuration", e)
                if isinstance(e, ConfigurationError):
                    raise
                else:
                    raise ConfigurationError(f"Configuration loading failed: {str(e)}")
    
    def _load_config_file(self, file_path: str) -> dict:
        """Load configuration from file."""
        file_path = validate_file_path(file_path)
        path = Path(file_path)
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    raise ConfigurationError("PyYAML required for YAML configuration files")
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {path.suffix}")
    
    def _dict_to_config(self, config_dict: dict) -> BCIGPTConfig:
        """Convert dictionary to configuration object."""
        try:
            # Create nested configuration objects
            model_config = ModelConfig(**config_dict.get('model', {}))
            training_config = TrainingConfig(**config_dict.get('training', {}))
            eeg_config = EEGConfig(**config_dict.get('eeg', {}))
            streaming_config = StreamingConfig(**config_dict.get('streaming', {}))
            security_config = SecurityConfig(**config_dict.get('security', {}))
            monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
            
            # Extract global settings
            global_settings = {k: v for k, v in config_dict.items() 
                             if k not in ['model', 'training', 'eeg', 'streaming', 'security', 'monitoring']}
            
            return BCIGPTConfig(
                model=model_config,
                training=training_config,
                eeg=eeg_config,
                streaming=streaming_config,
                security=security_config,
                monitoring=monitoring_config,
                **global_settings
            )
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration format: {str(e)}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_prefix = "BCI_GPT_"
        
        env_mappings = {
            f"{env_prefix}DEBUG": ("debug", bool),
            f"{env_prefix}LOG_LEVEL": ("monitoring.log_level", str),
            f"{env_prefix}MODEL_DEVICE": ("model.device", str),
            f"{env_prefix}BATCH_SIZE": ("training.batch_size", int),
            f"{env_prefix}LEARNING_RATE": ("training.learning_rate", float),
            f"{env_prefix}DATA_DIR": ("data_dir", str),
            f"{env_prefix}MODEL_DIR": ("model_dir", str),
        }
        
        for env_var, (config_path, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = type_func(os.environ[env_var])
                    self._set_config_value(config_path, value)
                    self.logger.log_info(f"Applied environment override: {config_path} = {value}")
                except (ValueError, TypeError) as e:
                    self.logger.log_warning(f"Invalid environment variable {env_var}: {e}")
    
    def _set_config_value(self, path: str, value: Any):
        """Set configuration value using dot notation path."""
        parts = path.split('.')
        obj = self.config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.data_dir,
            self.config.model_dir,
            self.config.cache_dir,
            self.config.temp_dir,
            self.config.monitoring.log_dir
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self.logger.log_warning(f"Failed to create directory {directory}: {e}")
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if not file_path:
            file_path = self.config_path
        
        if not file_path:
            raise ConfigurationError("No file path specified for saving configuration")
        
        try:
            config_dict = asdict(self.config)
            path = Path(file_path)
            
            with open(path, 'w') as f:
                if path.suffix.lower() in ['.yml', '.yaml']:
                    try:
                        import yaml
                        yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
                    except ImportError:
                        raise ConfigurationError("PyYAML required for YAML configuration files")
                elif path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {path.suffix}")
            
            self.logger.log_info(f"Saved configuration to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def get_config(self) -> BCIGPTConfig:
        """Get current configuration."""
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        with self.reload_lock:
            try:
                # Apply updates
                for path, value in updates.items():
                    # Sanitize string inputs
                    if isinstance(value, str):
                        value = sanitize_text_input(value)
                    
                    self._set_config_value(path, value)
                
                # Validate updated configuration
                errors = self.validator.validate_config(self.config)
                if errors:
                    self.logger.log_error(f"Configuration validation errors after update: {errors}")
                    # Could optionally reload from file to revert
                    raise ConfigurationError(f"Invalid configuration after update: {errors}")
                
                self.logger.log_info(f"Updated configuration: {updates}")
                
                # Notify callbacks
                self._notify_reload_callbacks()
                
            except Exception as e:
                self.logger.log_error("Configuration update failed", e)
                raise ConfigurationError(f"Configuration update failed: {str(e)}")
    
    def enable_hot_reload(self, check_interval: float = 1.0):
        """Enable hot-reloading of configuration file."""
        if not self.config_path:
            self.logger.log_warning("Cannot enable hot-reload: no configuration file path")
            return
        
        if self.hot_reload_enabled:
            self.logger.log_warning("Hot-reload already enabled")
            return
        
        self.hot_reload_enabled = True
        self.reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            args=(check_interval,),
            daemon=True
        )
        self.reload_thread.start()
        
        self.logger.log_info(f"Enabled configuration hot-reload (interval: {check_interval}s)")
    
    def disable_hot_reload(self):
        """Disable hot-reloading."""
        self.hot_reload_enabled = False
        if self.reload_thread:
            self.reload_thread.join(timeout=2.0)
        self.logger.log_info("Disabled configuration hot-reload")
    
    def add_reload_callback(self, callback: callable):
        """Add callback to be called when configuration reloads."""
        self.reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: callable):
        """Remove reload callback."""
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
    
    def _hot_reload_loop(self, check_interval: float):
        """Hot-reload monitoring loop."""
        while self.hot_reload_enabled:
            try:
                if os.path.exists(self.config_path):
                    current_modified = os.path.getmtime(self.config_path)
                    
                    if self.last_modified and current_modified > self.last_modified:
                        self.logger.log_info("Configuration file changed, reloading...")
                        self.load_config()
                        self._notify_reload_callbacks()
                
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.log_error("Hot-reload check failed", e)
                time.sleep(check_interval)
    
    def _notify_reload_callbacks(self):
        """Notify all reload callbacks."""
        for callback in self.reload_callbacks:
            try:
                callback(self.config)
            except Exception as e:
                self.logger.log_error("Reload callback failed", e)
    
    def export_config_schema(self, file_path: str):
        """Export configuration schema for documentation."""
        try:
            schema = {
                "title": "BCI-GPT Configuration Schema",
                "version": self.config.version if self.config else "1.0.0",
                "generated": datetime.utcnow().isoformat(),
                "sections": {
                    "model": {
                        "description": "Model architecture configuration",
                        "fields": {
                            field.name: {
                                "type": str(field.type),
                                "default": field.default if field.default != dataclass.MISSING else None,
                                "description": f"Model {field.name} parameter"
                            }
                            for field in ModelConfig.__dataclass_fields__.values()
                        }
                    },
                    "training": {
                        "description": "Training configuration",
                        "fields": {
                            field.name: {
                                "type": str(field.type),
                                "default": field.default if field.default != dataclass.MISSING else None,
                                "description": f"Training {field.name} parameter"
                            }
                            for field in TrainingConfig.__dataclass_fields__.values()
                        }
                    },
                    # Add other sections...
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(schema, f, indent=2, default=str)
            
            self.logger.log_info(f"Exported configuration schema to {file_path}")
            
        except Exception as e:
            self.logger.log_error("Failed to export configuration schema", e)
            raise ConfigurationError(f"Schema export failed: {str(e)}")


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager is None:
        config_file = config_path or os.getenv('BCI_GPT_CONFIG', './config.yaml')
        _global_config_manager = ConfigManager(config_file)
    
    return _global_config_manager


def get_config() -> BCIGPTConfig:
    """Get current configuration."""
    return get_config_manager().get_config()


def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values."""
    get_config_manager().update_config(updates)