"""Core BCI-GPT models and architectures."""

# Safe imports with fallbacks
__all__ = []

try:
    from .models import BCIGPTModel
    __all__.append("BCIGPTModel")
except ImportError:
    pass

try:
    from .inverse_gan import InverseSimulator, Generator, Discriminator
    __all__.extend(["InverseSimulator", "Generator", "Discriminator"])
except ImportError:
    pass

try:
    from .fusion_layers import CrossAttentionFusion, MultiModalFusion
    __all__.extend(["CrossAttentionFusion", "MultiModalFusion"])
except ImportError:
    pass

# Always available modules
try:
    from .error_handling import ErrorHandler, BCIGPTError, global_error_handler
    __all__.extend(["ErrorHandler", "BCIGPTError", "global_error_handler"])
except ImportError:
    pass

try:
    from .validation import DataValidator, ValidationResult, global_validator
    __all__.extend(["DataValidator", "ValidationResult", "global_validator"])
except ImportError:
    pass