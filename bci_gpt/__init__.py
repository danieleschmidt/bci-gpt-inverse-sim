"""BCI-GPT Inverse Simulator for Imagined Speech.

A comprehensive brain-computer interface toolkit that enables direct conversion
of imagined speech EEG signals to text using GPT-based language models and
GAN-based inverse mapping for synthetic EEG generation.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core.models import BCIGPTModel
from .core.inverse_gan import InverseSimulator
from .preprocessing.eeg_processor import EEGProcessor, SignalQuality
from .decoding.realtime_decoder import RealtimeDecoder
from .decoding.token_decoder import TokenDecoder
from .training.trainer import BCIGPTTrainer
from .inverse.text_to_eeg import TextToEEG
from .utils.streaming import StreamingEEG
from .utils.metrics import BCIMetrics

__all__ = [
    # Core models
    "BCIGPTModel",
    "InverseSimulator",
    # Preprocessing
    "EEGProcessor", 
    "SignalQuality",
    # Decoding
    "RealtimeDecoder",
    "TokenDecoder",
    # Training
    "BCIGPTTrainer",
    # Inverse mapping
    "TextToEEG",
    # Utilities
    "StreamingEEG",
    "BCIMetrics",
]