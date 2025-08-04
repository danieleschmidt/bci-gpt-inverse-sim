"""Custom loss functions for BCI-GPT training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


class BCILoss(nn.Module):
    """Main BCI-GPT loss combining multiple objectives."""
    
    def __init__(self,
                 language_weight: float = 1.0,
                 reconstruction_weight: float = 0.1,
                 consistency_weight: float = 0.05,
                 smoothness_weight: float = 0.01):
        super().__init__()
        
        self.language_weight = language_weight
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        
        # Component losses
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute total BCI loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target dictionary with labels and EEG data
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Language modeling loss
        if 'logits' in outputs and 'labels' in targets:
            language_loss = self.compute_language_loss(outputs['logits'], targets['labels'])
            losses['language_loss'] = language_loss
        
        # EEG reconstruction loss
        if 'eeg_reconstruction' in outputs and 'eeg_data' in targets:
            recon_loss = self.compute_reconstruction_loss(
                outputs['eeg_reconstruction'], targets['eeg_data']
            )
            losses['reconstruction_loss'] = recon_loss
        
        # Consistency loss between modalities
        if 'eeg_features' in outputs and 'text_features' in outputs:
            consistency_loss = self.compute_consistency_loss(
                outputs['eeg_features'], outputs['text_features']
            )
            losses['consistency_loss'] = consistency_loss
        
        # Temporal smoothness loss for EEG features
        if 'eeg_features' in outputs:
            smoothness_loss = self.compute_smoothness_loss(outputs['eeg_features'])
            losses['smoothness_loss'] = smoothness_loss
        
        # Compute total loss
        total_loss = 0.0
        if 'language_loss' in losses:
            total_loss += self.language_weight * losses['language_loss']
        if 'reconstruction_loss' in losses:
            total_loss += self.reconstruction_weight * losses['reconstruction_loss']
        if 'consistency_loss' in losses:
            total_loss += self.consistency_weight * losses['consistency_loss']
        if 'smoothness_loss' in losses:
            total_loss += self.smoothness_weight * losses['smoothness_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_language_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss.
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len)
            
        Returns:
            Cross-entropy loss
        """
        # Shift labels for causal language modeling
        if logits.dim() == 3:
            # Sequence prediction
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            
            # Flatten for cross entropy
            loss = self.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1)
            )
        else:
            # Single token prediction
            loss = self.cross_entropy(logits, labels)
        
        return loss
    
    def compute_reconstruction_loss(self, 
                                  reconstructed: torch.Tensor,
                                  original: torch.Tensor) -> torch.Tensor:
        """Compute EEG reconstruction loss.
        
        Args:
            reconstructed: Reconstructed EEG (batch_size, channels, seq_len)
            original: Original EEG (batch_size, channels, seq_len)
            
        Returns:
            Reconstruction loss combining MSE and spectral loss
        """
        # Time domain reconstruction
        mse_loss = self.mse_loss(reconstructed, original)
        
        # Spectral domain reconstruction
        spectral_loss = self._spectral_reconstruction_loss(reconstructed, original)
        
        return mse_loss + 0.1 * spectral_loss
    
    def compute_consistency_loss(self,
                               eeg_features: torch.Tensor,
                               text_features: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between EEG and text features.
        
        Args:
            eeg_features: EEG features (batch_size, eeg_seq_len, hidden_dim)
            text_features: Text features (batch_size, text_seq_len, hidden_dim)
            
        Returns:
            Consistency loss
        """
        # Pool features to same dimension
        eeg_pooled = torch.mean(eeg_features, dim=1)  # (batch_size, hidden_dim)
        text_pooled = torch.mean(text_features, dim=1)  # (batch_size, hidden_dim)
        
        # Cosine similarity loss
        cosine_sim = F.cosine_similarity(eeg_pooled, text_pooled, dim=1)
        consistency_loss = 1.0 - cosine_sim.mean()
        
        return consistency_loss
    
    def compute_smoothness_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss.
        
        Args:
            features: Feature tensor (batch_size, seq_len, hidden_dim)
            
        Returns:
            Smoothness loss
        """
        # Total variation loss
        diff = features[:, 1:, :] - features[:, :-1, :]
        smoothness_loss = torch.mean(torch.abs(diff))
        
        return smoothness_loss
    
    def _spectral_reconstruction_loss(self,
                                    reconstructed: torch.Tensor,
                                    original: torch.Tensor) -> torch.Tensor:
        """Compute spectral reconstruction loss using FFT.
        
        Args:
            reconstructed: Reconstructed signals
            original: Original signals
            
        Returns:
            Spectral loss
        """
        # Compute FFT
        recon_fft = torch.fft.rfft(reconstructed, dim=2)
        orig_fft = torch.fft.rfft(original, dim=2)
        
        # Compare magnitudes
        recon_mag = torch.abs(recon_fft)
        orig_mag = torch.abs(orig_fft)
        
        spectral_loss = self.mse_loss(recon_mag, orig_mag)
        
        return spectral_loss


class ReconstructionLoss(nn.Module):
    """Specialized reconstruction loss for EEG signals."""
    
    def __init__(self,
                 time_weight: float = 1.0,
                 freq_weight: float = 0.5,
                 band_weight: float = 0.3):
        super().__init__()
        
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.band_weight = band_weight
        
        # Frequency bands for EEG
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def forward(self,
                reconstructed: torch.Tensor,
                original: torch.Tensor,
                sampling_rate: float = 1000.0) -> torch.Tensor:
        """Compute reconstruction loss.
        
        Args:
            reconstructed: Reconstructed EEG (batch_size, channels, seq_len)
            original: Original EEG (batch_size, channels, seq_len)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Total reconstruction loss
        """
        total_loss = 0.0
        
        # Time domain loss
        if self.time_weight > 0:
            time_loss = F.mse_loss(reconstructed, original)
            total_loss += self.time_weight * time_loss
        
        # Frequency domain loss
        if self.freq_weight > 0:
            freq_loss = self._frequency_domain_loss(reconstructed, original)
            total_loss += self.freq_weight * freq_loss
        
        # Frequency band loss
        if self.band_weight > 0:
            band_loss = self._frequency_band_loss(reconstructed, original, sampling_rate)
            total_loss += self.band_weight * band_loss
        
        return total_loss
    
    def _frequency_domain_loss(self,
                             reconstructed: torch.Tensor,
                             original: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain reconstruction loss."""
        # FFT
        recon_fft = torch.fft.rfft(reconstructed, dim=2)
        orig_fft = torch.fft.rfft(original, dim=2)
        
        # Magnitude and phase losses
        mag_loss = F.mse_loss(torch.abs(recon_fft), torch.abs(orig_fft))
        phase_loss = F.mse_loss(torch.angle(recon_fft), torch.angle(orig_fft))
        
        return mag_loss + 0.1 * phase_loss
    
    def _frequency_band_loss(self,
                           reconstructed: torch.Tensor,
                           original: torch.Tensor,
                           sampling_rate: float) -> torch.Tensor:
        """Compute frequency band-specific reconstruction loss."""
        seq_len = reconstructed.shape[2]
        freqs = torch.fft.rfftfreq(seq_len, d=1/sampling_rate).to(reconstructed.device)
        
        # FFT
        recon_fft = torch.fft.rfft(reconstructed, dim=2)
        orig_fft = torch.fft.rfft(original, dim=2)
        
        recon_power = torch.abs(recon_fft) ** 2
        orig_power = torch.abs(orig_fft) ** 2
        
        band_loss = 0.0
        
        for band_name, (low, high) in self.bands.items():
            # Find frequency indices for this band
            band_mask = (freqs >= low) & (freqs <= high)
            
            if torch.any(band_mask):
                # Power in this band
                recon_band_power = torch.sum(recon_power[:, :, band_mask], dim=2)
                orig_band_power = torch.sum(orig_power[:, :, band_mask], dim=2)
                
                # MSE loss for this band
                band_mse = F.mse_loss(recon_band_power, orig_band_power)
                band_loss += band_mse
        
        return band_loss / len(self.bands)


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        
        self.loss_type = loss_type
        
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion = self._hinge_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def generator_loss(self, fake_outputs: torch.Tensor) -> torch.Tensor:
        """Compute generator loss (want discriminator to classify fake as real).
        
        Args:
            fake_outputs: Discriminator outputs for generated samples
            
        Returns:
            Generator loss
        """
        if self.loss_type == "hinge":
            return -torch.mean(fake_outputs)
        else:
            real_labels = torch.ones_like(fake_outputs)
            return self.criterion(fake_outputs, real_labels)
    
    def discriminator_loss(self,
                          real_outputs: torch.Tensor,
                          fake_outputs: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss.
        
        Args:
            real_outputs: Discriminator outputs for real samples
            fake_outputs: Discriminator outputs for fake samples
            
        Returns:
            Discriminator loss
        """
        if self.loss_type == "hinge":
            real_loss = torch.relu(1.0 - real_outputs).mean()
            fake_loss = torch.relu(1.0 + fake_outputs).mean()
            return real_loss + fake_loss
        else:
            real_labels = torch.ones_like(real_outputs)
            fake_labels = torch.zeros_like(fake_outputs)
            
            real_loss = self.criterion(real_outputs, real_labels)
            fake_loss = self.criterion(fake_outputs, fake_labels)
            
            return (real_loss + fake_loss) / 2
    
    def _hinge_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Hinge loss implementation."""
        return torch.relu(1.0 - outputs * targets).mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning aligned EEG-text representations."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        
        self.temperature = temperature
        self.margin = margin
    
    def forward(self,
                eeg_features: torch.Tensor,
                text_features: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            eeg_features: EEG features (batch_size, feature_dim)
            text_features: Text features (batch_size, feature_dim)
            labels: Optional binary labels (1 for matching pairs)
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        eeg_norm = F.normalize(eeg_features, dim=1)
        text_norm = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(eeg_norm, text_norm.T) / self.temperature
        
        if labels is None:
            # Assume diagonal corresponds to positive pairs
            batch_size = eeg_features.shape[0]
            labels = torch.eye(batch_size, device=eeg_features.device)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        pos_sim = torch.sum(exp_sim * labels, dim=1)
        all_sim = torch.sum(exp_sim, dim=1)
        
        loss = -torch.log(pos_sim / all_sim).mean()
        
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in token prediction."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model logits (batch_size, num_classes)
            targets: Target labels (batch_size,)
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()