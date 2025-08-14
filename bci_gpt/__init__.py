"""BCI-GPT Inverse Simulator for Imagined Speech.

A comprehensive brain-computer interface toolkit that enables direct conversion
of imagined speech EEG signals to text using GPT-based language models and
GAN-based inverse mapping for synthetic EEG generation.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Optional imports for graceful degradation without heavy dependencies
__all__ = []

# Core utilities (always available)
try:
    from .utils.logging_config import setup_logging, get_logger
    from .utils.error_handling import BCI_GPTError
    from .utils.config_manager import get_config_manager
    __all__.extend(["setup_logging", "get_logger", "BCI_GPTError", "get_config_manager"])
except ImportError:
    pass

# Core models (require PyTorch)
try:
    from .core.models import BCIGPTModel
    from .core.inverse_gan import InverseSimulator
    __all__.extend(["BCIGPTModel", "InverseSimulator"])
except ImportError:
    # Heavy dependencies not available
    pass

# Preprocessing (require MNE, NumPy)
try:
    from .preprocessing.eeg_processor import EEGProcessor, SignalQuality
    __all__.extend(["EEGProcessor", "SignalQuality"])
except ImportError:
    pass

# Decoding (require PyTorch)
try:
    from .decoding.realtime_decoder import RealtimeDecoder
    from .decoding.token_decoder import TokenDecoder
    __all__.extend(["RealtimeDecoder", "TokenDecoder"])
except ImportError:
    pass

# Training (require PyTorch)
try:
    from .training.trainer import BCIGPTTrainer
    __all__.extend(["BCIGPTTrainer"])
except ImportError:
    pass

# Inverse mapping (require PyTorch)
try:
    from .inverse.text_to_eeg import TextToEEG
    __all__.extend(["TextToEEG"])
except ImportError:
    pass

# Utilities (lightweight)
try:
    from .utils.streaming import StreamingEEG
    from .utils.metrics import BCIMetrics
    __all__.extend(["StreamingEEG", "BCIMetrics"])
except ImportError:
    pass