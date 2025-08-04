"""Training modules for BCI-GPT models."""

from .trainer import BCIGPTTrainer
from .gan_trainer import GANTrainer
from .losses import BCILoss, ReconstructionLoss, AdversarialLoss
from .augmentation import EEGAugmenter

__all__ = [
    "BCIGPTTrainer",
    "GANTrainer",
    "BCILoss",
    "ReconstructionLoss", 
    "AdversarialLoss",
    "EEGAugmenter",
]