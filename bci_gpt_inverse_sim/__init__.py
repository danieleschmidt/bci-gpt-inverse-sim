"""
bci_gpt_inverse_sim — BCI-GPT Inverse Simulation Package.

Modules
-------
eeg_sim       : EEGSignalSim — synthetic EEG signal generation
band_power    : BandPowerExtractor — frequency band power extraction
inverse_mapper: InverseMapper — feature-to-logit linear mapping
decoder       : TopKDecoder + softmax — top-k word decoding
pipeline      : BCIPipeline — end-to-end demo pipeline
"""

from .eeg_sim import EEGSignalSim
from .band_power import BandPowerExtractor
from .inverse_mapper import InverseMapper
from .decoder import TopKDecoder, softmax
from .pipeline import BCIPipeline, demo

__all__ = [
    "EEGSignalSim",
    "BandPowerExtractor",
    "InverseMapper",
    "TopKDecoder",
    "softmax",
    "BCIPipeline",
    "demo",
]

__version__ = "0.1.0"
