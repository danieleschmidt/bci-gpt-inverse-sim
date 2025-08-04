"""Inverse mapping modules for text-to-EEG generation."""

from .text_to_eeg import TextToEEG
from .style_transfer import EEGStyleTransfer
from .validation import SyntheticEEGValidator

__all__ = [
    "TextToEEG",
    "EEGStyleTransfer",
    "SyntheticEEGValidator",
]