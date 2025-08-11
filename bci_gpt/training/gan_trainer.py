"""GAN trainer for inverse EEG synthesis."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
import warnings
from tqdm import tqdm
import logging

try:
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    F = None
    DataLoader = None
    warnings.warn("PyTorch modules not fully available")

from ..core.inverse_gan import Generator, Discriminator, InverseSimulator
from ..utils.metrics import GANMetrics
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class GANTrainer:
    """Advanced GAN trainer for text-to-EEG inverse mapping."""
    
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 device: str = "auto",
                 lr_generator: float = 2e-4,
                 lr_discriminator: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 lambda_gp: float = 10.0,
                 n_critic: int = 5):
        """Initialize GAN trainer.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            device: Device to use ('auto', 'cuda', 'cpu')
            lr_generator: Learning rate for generator
            lr_discriminator: Learning rate for discriminator
            beta1: Adam optimizer beta1 parameter
            beta2: Adam optimizer beta2 parameter
            lambda_gp: Gradient penalty weight
            n_critic: Number of discriminator updates per generator update
        """
        self.device = self._setup_device(device)
        
        # Move models to device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=lr_generator,
            betas=(beta1, beta2)
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_discriminator,
            betas=(beta1, beta2)
        )
        
        # Training parameters
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Metrics tracker
        self.metrics = GANMetrics()
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'gradient_penalty': [],
            'wasserstein_distance': []
        }
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        logger.info(f"Using device: {device_obj}")
        return device_obj
    
    def gradient_penalty(self, 
                        real_samples: torch.Tensor,
                        fake_samples: torch.Tensor,
                        text_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_samples.shape[0]
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Get discriminator output for interpolates
        if text_condition is not None:
            d_interpolates = self.discriminator(interpolates, text_condition)
        else:
            d_interpolates = self.discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(self,
                          real_eeg: torch.Tensor,
                          fake_eeg: torch.Tensor,
                          text_condition: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Train discriminator for one step."""
        self.optimizer_D.zero_grad()
        
        batch_size = real_eeg.shape[0]
        
        # Real samples
        if text_condition is not None:
            real_validity = self.discriminator(real_eeg, text_condition)
        else:
            real_validity = self.discriminator(real_eeg)
        
        # Fake samples  
        if text_condition is not None:
            fake_validity = self.discriminator(fake_eeg.detach(), text_condition)
        else:
            fake_validity = self.discriminator(fake_eeg.detach())
        
        # Wasserstein loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        # Gradient penalty
        gp = self.gradient_penalty(real_eeg, fake_eeg, text_condition)
        
        # Total discriminator loss
        total_d_loss = d_loss + self.lambda_gp * gp
        
        total_d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'gradient_penalty': gp.item(),
            'total_d_loss': total_d_loss.item(),
            'wasserstein_distance': -d_loss.item()
        }
    
    def train_generator(self,
                       fake_eeg: torch.Tensor,
                       text_condition: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Train generator for one step."""
        self.optimizer_G.zero_grad()
        
        # Generator loss
        if text_condition is not None:
            fake_validity = self.discriminator(fake_eeg, text_condition)
        else:
            fake_validity = self.discriminator(fake_eeg)
        
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'generator_loss': g_loss.item()
        }
    
    def train_step(self,
                   real_eeg: torch.Tensor,
                   text_embeddings: Optional[torch.Tensor] = None,
                   noise: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Perform one training step."""
        batch_size = real_eeg.shape[0]
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, 100).to(self.device)  # Standard noise dimension
        
        # Generate fake EEG
        if text_embeddings is not None:
            fake_eeg = self.generator(noise, text_embeddings)
        else:
            fake_eeg = self.generator(noise)
        
        # Train discriminator
        d_metrics = self.train_discriminator(real_eeg, fake_eeg, text_embeddings)
        
        # Train generator (less frequently)
        g_metrics = {}
        if len(self.history['discriminator_loss']) % self.n_critic == 0:
            # Generate fresh fake samples for generator training
            if text_embeddings is not None:
                fake_eeg_g = self.generator(noise, text_embeddings)
            else:
                fake_eeg_g = self.generator(noise)
            
            g_metrics = self.train_generator(fake_eeg_g, text_embeddings)
        
        # Combine metrics
        step_metrics = {**d_metrics, **g_metrics}
        
        # Update history
        if 'generator_loss' in g_metrics:
            self.history['generator_loss'].append(g_metrics['generator_loss'])
        if 'discriminator_loss' in d_metrics:
            self.history['discriminator_loss'].append(d_metrics['discriminator_loss'])
        if 'gradient_penalty' in d_metrics:
            self.history['gradient_penalty'].append(d_metrics['gradient_penalty'])
        if 'wasserstein_distance' in d_metrics:
            self.history['wasserstein_distance'].append(d_metrics['wasserstein_distance'])
        
        return step_metrics
    
    def train_epoch(self,
                    dataloader: DataLoader,
                    epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'generator_loss': 0.0,
            'discriminator_loss': 0.0,
            'gradient_penalty': 0.0,
            'wasserstein_distance': 0.0
        }
        
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Assume batch contains (eeg_data, text_embeddings) or just eeg_data
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    real_eeg, text_embeddings = batch[0], batch[1]
                    real_eeg = real_eeg.to(self.device)
                    text_embeddings = text_embeddings.to(self.device)
                else:
                    real_eeg = batch.to(self.device)
                    text_embeddings = None
                
                # Training step
                step_metrics = self.train_step(real_eeg, text_embeddings)
                
                # Accumulate metrics
                for key in epoch_metrics:
                    if key in step_metrics:
                        epoch_metrics[key] += step_metrics[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f"{step_metrics.get('discriminator_loss', 0):.4f}",
                    'G_loss': f"{step_metrics.get('generator_loss', 0):.4f}",
                    'GP': f"{step_metrics.get('gradient_penalty', 0):.4f}"
                })
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def fit(self,
            train_dataloader: DataLoader,
            epochs: int,
            val_dataloader: Optional[DataLoader] = None,
            save_every: int = 10,
            checkpoint_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """Train the GAN for multiple epochs."""
        logger.info(f"Starting GAN training for {epochs} epochs")
        
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch} - Train D_loss: {train_metrics['discriminator_loss']:.4f}, "
                          f"G_loss: {train_metrics['generator_loss']:.4f}, "
                          f"Val metrics: {val_metrics}")
            else:
                logger.info(f"Epoch {epoch} - D_loss: {train_metrics['discriminator_loss']:.4f}, "
                          f"G_loss: {train_metrics['generator_loss']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"{checkpoint_dir}/gan_epoch_{epoch}.pt", epoch)
        
        return self.history
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validate the GAN."""
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'val_d_loss': 0.0,
            'val_realism_score': 0.0
        }
        
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    real_eeg, text_embeddings = batch[0], batch[1]
                    real_eeg = real_eeg.to(self.device)
                    text_embeddings = text_embeddings.to(self.device)
                else:
                    real_eeg = batch.to(self.device)
                    text_embeddings = None
                
                batch_size = real_eeg.shape[0]
                
                # Generate fake samples
                noise = torch.randn(batch_size, 100).to(self.device)
                if text_embeddings is not None:
                    fake_eeg = self.generator(noise, text_embeddings)
                    real_validity = self.discriminator(real_eeg, text_embeddings)
                    fake_validity = self.discriminator(fake_eeg, text_embeddings)
                else:
                    fake_eeg = self.generator(noise)
                    real_validity = self.discriminator(real_eeg)
                    fake_validity = self.discriminator(fake_eeg)
                
                # Discriminator loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                val_metrics['val_d_loss'] += d_loss.item()
                
                # Realism score (average discriminator output for fake samples)
                realism_score = torch.mean(torch.sigmoid(fake_validity))
                val_metrics['val_realism_score'] += realism_score.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def save_checkpoint(self, filepath: str, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded: {filepath}, epoch {epoch}")
        return epoch
    
    def generate_samples(self,
                        text_embeddings: Optional[torch.Tensor] = None,
                        num_samples: int = 10,
                        noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate synthetic EEG samples."""
        self.generator.eval()
        
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(num_samples, 100).to(self.device)
            
            if text_embeddings is not None:
                # Repeat text embeddings for all samples
                if text_embeddings.shape[0] == 1:
                    text_embeddings = text_embeddings.repeat(num_samples, 1)
                
                synthetic_eeg = self.generator(noise, text_embeddings)
            else:
                synthetic_eeg = self.generator(noise)
            
            return synthetic_eeg.cpu()