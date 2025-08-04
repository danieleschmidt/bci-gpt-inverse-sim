"""EEG preprocessing modules for BCI-GPT."""

from .eeg_processor import EEGProcessor, SignalQuality
from .artifact_removal import ArtifactRemover
from .feature_extraction import FeatureExtractor

__all__ = [
    "EEGProcessor",
    "SignalQuality", 
    "ArtifactRemover",
    "FeatureExtractor",
]