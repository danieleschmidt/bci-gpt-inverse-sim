"""Utility modules for BCI-GPT."""

# Safe imports for graceful degradation
__all__ = []

try:
    from .logging_config import setup_logging, get_logger
    __all__.extend(["setup_logging", "get_logger"])
except ImportError:
    pass

try:
    from .config_manager import get_config_manager
    __all__.append("get_config_manager")
except ImportError:
    pass

try:
    from .streaming import StreamingEEG
    __all__.append("StreamingEEG")
except ImportError:
    pass

try:
    from .metrics import BCIMetrics
    __all__.append("BCIMetrics")
except ImportError:
    pass

try:
    from .visualization import EEGVisualizer
    __all__.append("EEGVisualizer")
except ImportError:
    pass