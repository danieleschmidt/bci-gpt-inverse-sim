"""GAN-based inverse mapping for synthetic EEG generation from text."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class Generator(nn.Module):
    """Generator network for creating synthetic EEG from text embeddings."""
    
    def __init__(self,
                 text_embedding_dim: int = 768,
                 noise_dim: int = 100,
                 eeg_channels: int = 9,
                 eeg_sequence_length: int = 1000,
                 hidden_dims: list = [512, 1024, 2048],
                 dropout: float = 0.2):
        super().__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.noise_dim = noise_dim
        self.eeg_channels = eeg_channels
        self.eeg_sequence_length = eeg_sequence_length
        self.hidden_dims = hidden_dims
        
        # Input dimension (text embedding + noise)
        input_dim = text_embedding_dim + noise_dim
        
        # Build encoder for text + noise
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent to EEG decoder
        # First project to flattened EEG dimension
        total_eeg_dim = eeg_channels * eeg_sequence_length
        
        self.latent_to_eeg = nn.Sequential(
            nn.Linear(hidden_dims[-1], total_eeg_dim),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
        
        # Temporal refinement with 1D convolutions
        self.temporal_refiner = nn.Sequential(
            nn.Conv1d(eeg_channels, eeg_channels * 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(eeg_channels * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(eeg_channels * 2, eeg_channels * 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(eeg_channels * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(eeg_channels * 2, eeg_channels, kernel_size=15, padding=7),
            nn.Tanh()
        )
        
        # Frequency shaping layers for realistic EEG bands
        self.frequency_shaper = self._create_frequency_shaper()
        
    def _create_frequency_shaper(self) -> nn.Module:
        """Create frequency shaping network for realistic EEG bands."""
        return nn.Sequential(
            # Delta band (0.5-4 Hz)
            nn.Conv1d(self.eeg_channels, self.eeg_channels, kernel_size=128, padding=64),
            nn.BatchNorm1d(self.eeg_channels),
            nn.LeakyReLU(0.2),
            
            # Alpha/Beta band shaping
            nn.Conv1d(self.eeg_channels, self.eeg_channels, kernel_size=32, padding=16),
            nn.BatchNorm1d(self.eeg_channels),
            nn.LeakyReLU(0.2),
            
            # Final smoothing
            nn.Conv1d(self.eeg_channels, self.eeg_channels, kernel_size=8, padding=4),
            nn.Tanh()
        )
    
    def forward(self, 
                text_embeddings: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate synthetic EEG from text embeddings.
        
        Args:
            text_embeddings: Text embeddings (batch_size, text_embedding_dim)
            noise: Random noise (batch_size, noise_dim). If None, samples random noise.
            
        Returns:
            Generated EEG signals (batch_size, eeg_channels, eeg_sequence_length)
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Concatenate text embeddings and noise
        generator_input = torch.cat([text_embeddings, noise], dim=1)
        
        # Encode to latent space
        latent = self.encoder(generator_input)  # (batch_size, hidden_dims[-1])
        
        # Generate initial EEG
        eeg_flat = self.latent_to_eeg(latent)  # (batch_size, channels * seq_len)
        
        # Reshape to EEG format
        eeg_reshaped = eeg_flat.view(batch_size, self.eeg_channels, self.eeg_sequence_length)
        
        # Temporal refinement
        eeg_refined = self.temporal_refiner(eeg_reshaped)
        
        # Frequency shaping for realistic EEG characteristics
        eeg_final = self.frequency_shaper(eeg_refined)
        
        return eeg_final


class Discriminator(nn.Module):
    """Discriminator network for distinguishing real from synthetic EEG."""
    
    def __init__(self,
                 eeg_channels: int = 9,
                 eeg_sequence_length: int = 1000,
                 text_embedding_dim: int = 768,
                 hidden_dims: list = [2048, 1024, 512],
                 dropout: float = 0.3):
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.eeg_sequence_length = eeg_sequence_length
        self.text_embedding_dim = text_embedding_dim
        
        # EEG feature extractor
        self.eeg_encoder = nn.Sequential(
            # Temporal convolutions
            nn.Conv1d(eeg_channels, 64, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        
        # Calculate conv output size
        conv_output_size = self._calculate_conv_output_size()
        eeg_feature_dim = 256 * conv_output_size
        
        # EEG feature projection
        self.eeg_projection = nn.Sequential(
            nn.Linear(eeg_feature_dim, hidden_dims[0] // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Text embedding projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dims[0] // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Combined discriminator
        discriminator_layers = []
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            discriminator_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        discriminator_layers.append(nn.Linear(prev_dim, 1))
        
        self.discriminator = nn.Sequential(*discriminator_layers)
        
        # Spectral analysis for additional realism check
        self.spectral_discriminator = self._create_spectral_discriminator()
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolution layers."""
        size = self.eeg_sequence_length
        # Three conv layers with stride 4 each
        for _ in range(3):
            size = (size + 2 * 30 - 64) // 4 + 1  # First layer padding
            size = (size + 2 * 14 - 32) // 4 + 1  # Second layer padding  
            size = (size + 2 * 6 - 16) // 4 + 1   # Third layer padding
            break  # Just need the final size
        
        # Rough approximation - the exact calculation depends on padding
        return max(1, self.eeg_sequence_length // (4 ** 3))
    
    def _create_spectral_discriminator(self) -> nn.Module:
        """Create spectral analysis discriminator for frequency domain realism."""
        return nn.Sequential(
            nn.Linear(5, 64),  # 5 frequency bands (delta, theta, alpha, beta, gamma)
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _extract_spectral_features(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from EEG data."""
        batch_size, n_channels, seq_len = eeg_data.shape
        
        # Simple spectral analysis using FFT
        fft = torch.fft.rfft(eeg_data, dim=2)
        power_spectrum = torch.abs(fft) ** 2
        
        # Define frequency bands (assuming 1000 Hz sampling rate)
        freqs = torch.fft.rfftfreq(seq_len, d=1/1000.0).to(eeg_data.device)
        
        # Extract power in different bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        band_powers = []
        for low, high in bands.values():
            band_mask = (freqs >= low) & (freqs <= high)
            if torch.any(band_mask):
                band_power = torch.mean(power_spectrum[:, :, band_mask], dim=(1, 2))
            else:
                band_power = torch.zeros(batch_size, device=eeg_data.device)
            band_powers.append(band_power)
        
        return torch.stack(band_powers, dim=1)  # (batch_size, 5)
    
    def forward(self, 
                eeg_data: torch.Tensor,
                text_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Discriminate between real and synthetic EEG.
        
        Args:
            eeg_data: EEG signals (batch_size, eeg_channels, eeg_sequence_length)
            text_embeddings: Corresponding text embeddings (batch_size, text_embedding_dim)
            
        Returns:
            Dictionary containing discrimination outputs
        """
        batch_size = eeg_data.shape[0]
        
        # Extract EEG features
        eeg_conv_features = self.eeg_encoder(eeg_data)  # (batch, 256, conv_out_size)
        eeg_conv_flat = eeg_conv_features.view(batch_size, -1)
        eeg_features = self.eeg_projection(eeg_conv_flat)
        
        # Process text embeddings if provided
        if text_embeddings is not None:
            text_features = self.text_projection(text_embeddings)
            # Combine EEG and text features
            combined_features = torch.cat([eeg_features, text_features], dim=1)
        else:
            # Use only EEG features (unconditional discrimination)
            combined_features = torch.cat([eeg_features, 
                                         torch.zeros_like(eeg_features)], dim=1)
        
        # Main discrimination
        main_output = self.discriminator(combined_features)
        
        # Spectral discrimination
        spectral_features = self._extract_spectral_features(eeg_data)
        spectral_output = self.spectral_discriminator(spectral_features)
        
        return {
            'main_output': main_output,
            'spectral_output': spectral_output,
            'combined_output': 0.7 * torch.sigmoid(main_output) + 0.3 * spectral_output
        }


class InverseSimulator(nn.Module):
    """Complete GAN-based inverse simulator for text-to-EEG generation."""
    
    def __init__(self,
                 generator_layers: list = [512, 1024, 2048],
                 discriminator_layers: list = [2048, 1024, 512],
                 noise_dim: int = 100,
                 eeg_channels: int = 9,
                 eeg_sequence_length: int = 1000,
                 text_embedding_dim: int = 768,
                 conditional: bool = True):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.conditional = conditional
        
        # Generator
        self.generator = Generator(
            text_embedding_dim=text_embedding_dim,
            noise_dim=noise_dim,
            eeg_channels=eeg_channels,
            eeg_sequence_length=eeg_sequence_length,
            hidden_dims=generator_layers
        )
        
        # Discriminator
        self.discriminator = Discriminator(
            eeg_channels=eeg_channels,
            eeg_sequence_length=eeg_sequence_length,
            text_embedding_dim=text_embedding_dim if conditional else 0,
            hidden_dims=discriminator_layers
        )
        
    def generate(self, 
                 text_embeddings: torch.Tensor,
                 noise: Optional[torch.Tensor] = None,
                 num_samples: int = 1) -> torch.Tensor:
        """Generate synthetic EEG from text embeddings.
        
        Args:
            text_embeddings: Text embeddings (batch_size, text_embedding_dim)
            noise: Random noise (batch_size, noise_dim)
            num_samples: Number of samples to generate per text embedding
            
        Returns:
            Generated EEG signals (batch_size * num_samples, eeg_channels, eeg_sequence_length)
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        # Expand text embeddings for multiple samples
        if num_samples > 1:
            text_embeddings = text_embeddings.repeat_interleave(num_samples, dim=0)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size * num_samples, self.noise_dim, device=device)
        
        # Generate synthetic EEG
        with torch.no_grad():
            synthetic_eeg = self.generator(text_embeddings, noise)
        
        return synthetic_eeg
    
    def discriminate(self,
                    eeg_data: torch.Tensor,
                    text_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Discriminate between real and synthetic EEG.
        
        Args:
            eeg_data: EEG signals (batch_size, eeg_channels, eeg_sequence_length)
            text_embeddings: Text embeddings (batch_size, text_embedding_dim)
            
        Returns:
            Discrimination outputs
        """
        if self.conditional:
            return self.discriminator(eeg_data, text_embeddings)
        else:
            return self.discriminator(eeg_data)
    
    def compute_generator_loss(self,
                              fake_outputs: Dict[str, torch.Tensor],
                              real_eeg: torch.Tensor,
                              fake_eeg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute generator loss with multiple components.
        
        Args:
            fake_outputs: Discriminator outputs for fake EEG
            real_eeg: Real EEG data for reconstruction loss
            fake_eeg: Generated EEG data
            
        Returns:
            Dictionary of loss components
        """
        # Adversarial loss (want discriminator to classify fake as real)
        adv_loss = F.binary_cross_entropy_with_logits(
            fake_outputs['main_output'], 
            torch.ones_like(fake_outputs['main_output'])
        )
        
        # Spectral adversarial loss
        spectral_adv_loss = F.binary_cross_entropy(
            fake_outputs['spectral_output'],
            torch.ones_like(fake_outputs['spectral_output'])
        )
        
        # Feature matching loss (if real EEG provided)
        feature_loss = torch.tensor(0.0, device=fake_eeg.device)
        if real_eeg is not None and real_eeg.shape == fake_eeg.shape:
            # Simple L2 feature matching in spectral domain
            real_fft = torch.abs(torch.fft.rfft(real_eeg, dim=2))
            fake_fft = torch.abs(torch.fft.rfft(fake_eeg, dim=2))
            feature_loss = F.mse_loss(fake_fft, real_fft)
        
        # Total generator loss
        total_loss = adv_loss + 0.5 * spectral_adv_loss + 0.1 * feature_loss
        
        return {
            'total_loss': total_loss,
            'adversarial_loss': adv_loss,
            'spectral_loss': spectral_adv_loss,
            'feature_loss': feature_loss
        }
    
    def compute_discriminator_loss(self,
                                  real_outputs: Dict[str, torch.Tensor],
                                  fake_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss.
        
        Args:
            real_outputs: Discriminator outputs for real EEG
            fake_outputs: Discriminator outputs for fake EEG
            
        Returns:
            Dictionary of loss components
        """
        # Real loss (want to classify real as real)
        real_loss = F.binary_cross_entropy_with_logits(
            real_outputs['main_output'],
            torch.ones_like(real_outputs['main_output'])
        )
        
        # Fake loss (want to classify fake as fake)
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_outputs['main_output'],
            torch.zeros_like(fake_outputs['main_output'])
        )
        
        # Spectral losses
        real_spectral_loss = F.binary_cross_entropy(
            real_outputs['spectral_output'],
            torch.ones_like(real_outputs['spectral_output'])
        )
        
        fake_spectral_loss = F.binary_cross_entropy(
            fake_outputs['spectral_output'],
            torch.zeros_like(fake_outputs['spectral_output'])
        )
        
        # Total discriminator loss
        main_loss = (real_loss + fake_loss) / 2
        spectral_loss = (real_spectral_loss + fake_spectral_loss) / 2
        total_loss = main_loss + 0.5 * spectral_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'spectral_loss': spectral_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        }