"""EEG style transfer for different mental states."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
import warnings
import numpy as np

from ..core.models import EEGEncoder
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class EEGStyleTransfer(nn.Module):
    """Transfer EEG styles between different mental states."""
    
    def __init__(self,
                 eeg_channels: int = 9,
                 sequence_length: int = 1000,
                 latent_dim: int = 256,
                 style_dim: int = 64,
                 num_styles: int = 5):
        """Initialize EEG style transfer model.
        
        Args:
            eeg_channels: Number of EEG channels
            sequence_length: Length of EEG sequences
            latent_dim: Dimension of latent representations
            style_dim: Dimension of style embeddings
            num_styles: Number of different styles (mental states)
        """
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_styles = num_styles
        
        # Content encoder - extracts content independent of style
        self.content_encoder = EEGEncoder(
            n_channels=eeg_channels,
            sequence_length=sequence_length,
            hidden_dim=latent_dim
        )
        
        # Style encoder - extracts style information
        self.style_encoder = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=32, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, style_dim),
            nn.Tanh()
        )
        
        # Style embeddings for different mental states
        self.style_embeddings = nn.Embedding(num_styles, style_dim)
        
        # Decoder - reconstructs EEG from content and style
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + style_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, eeg_channels * sequence_length),
            nn.Tanh()
        )
        
        # Style classifier for adversarial training
        self.style_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_styles)
        )
        
    def encode_content(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Encode EEG content (style-independent features).
        
        Args:
            eeg_data: Input EEG data (batch, channels, sequence_length)
            
        Returns:
            Content features (batch, seq_len, latent_dim)
        """
        content_features = self.content_encoder(eeg_data)
        # Pool over sequence dimension for content representation
        content = torch.mean(content_features, dim=1)  # (batch, latent_dim)
        return content
    
    def encode_style(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Encode EEG style information.
        
        Args:
            eeg_data: Input EEG data (batch, channels, sequence_length)
            
        Returns:
            Style features (batch, style_dim)
        """
        style_features = self.style_encoder(eeg_data)
        return style_features
    
    def decode(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Decode EEG from content and style representations.
        
        Args:
            content: Content features (batch, latent_dim)
            style: Style features (batch, style_dim)
            
        Returns:
            Reconstructed EEG (batch, channels, sequence_length)
        """
        # Combine content and style
        combined = torch.cat([content, style], dim=1)
        
        # Decode to EEG
        decoded_flat = self.decoder(combined)
        decoded_eeg = decoded_flat.view(-1, self.eeg_channels, self.sequence_length)
        
        return decoded_eeg
    
    def forward(self, eeg_data: torch.Tensor, target_style: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for style transfer.
        
        Args:
            eeg_data: Input EEG data (batch, channels, sequence_length)
            target_style: Target style indices (batch,) or None for reconstruction
            
        Returns:
            Dictionary with model outputs
        """
        batch_size = eeg_data.shape[0]
        
        # Encode content and style
        content = self.encode_content(eeg_data)
        original_style = self.encode_style(eeg_data)
        
        outputs = {
            'content': content,
            'original_style': original_style
        }
        
        # Reconstruction with original style
        reconstructed = self.decode(content, original_style)
        outputs['reconstructed'] = reconstructed
        
        # Style classification for adversarial loss
        style_logits = self.style_classifier(content)
        outputs['style_logits'] = style_logits
        
        # Style transfer if target style provided
        if target_style is not None:
            if isinstance(target_style, (int, list)):
                target_style = torch.tensor(target_style, device=eeg_data.device)
            
            # Get target style embeddings
            target_style_embedding = self.style_embeddings(target_style)
            transferred = self.decode(content, target_style_embedding)
            
            outputs['target_style'] = target_style_embedding
            outputs['transferred'] = transferred
        
        return outputs
    
    def transfer_style(self, 
                      eeg_data: torch.Tensor, 
                      source_style: Union[int, str],
                      target_style: Union[int, str]) -> torch.Tensor:
        """Transfer EEG from source to target style.
        
        Args:
            eeg_data: Input EEG data
            source_style: Source mental state (index or name)
            target_style: Target mental state (index or name)
            
        Returns:
            Style-transferred EEG
        """
        self.eval()
        
        # Convert style names to indices if needed
        style_names = {
            'rest': 0,
            'imagined_speech': 1,
            'inner_monologue': 2,
            'motor_imagery': 3,
            'focused_attention': 4
        }
        
        if isinstance(source_style, str):
            source_style = style_names.get(source_style, 0)
        if isinstance(target_style, str):
            target_style = style_names.get(target_style, 1)
        
        with torch.no_grad():
            # Extract content
            content = self.encode_content(eeg_data)
            
            # Get target style embedding
            target_style_tensor = torch.tensor([target_style], device=eeg_data.device)
            target_style_embedding = self.style_embeddings(target_style_tensor)
            
            # Generate transferred EEG
            transferred_eeg = self.decode(content, target_style_embedding)
        
        return transferred_eeg
    
    def compute_style_loss(self, outputs: Dict[str, torch.Tensor], style_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute style transfer losses.
        
        Args:
            outputs: Model outputs dictionary
            style_labels: True style labels
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Reconstruction loss
        if 'reconstructed' in outputs:
            reconstruction_loss = F.mse_loss(outputs['reconstructed'], outputs.get('original_eeg', torch.zeros_like(outputs['reconstructed'])))
            losses['reconstruction'] = reconstruction_loss
        
        # Style classification loss (for content encoder to be style-invariant)
        if 'style_logits' in outputs:
            # Want to maximize entropy (minimize classification accuracy)
            uniform_target = torch.full_like(outputs['style_logits'], 1.0 / self.num_styles)
            style_adversarial_loss = -F.kl_div(
                F.log_softmax(outputs['style_logits'], dim=1),
                uniform_target,
                reduction='batchmean'
            )
            losses['style_adversarial'] = style_adversarial_loss
        
        # Content preservation loss
        if 'transferred' in outputs and 'content' in outputs:
            # Ensure transferred EEG has same content
            transferred_content = self.encode_content(outputs['transferred'])
            content_loss = F.mse_loss(transferred_content, outputs['content'])
            losses['content_preservation'] = content_loss
        
        # Style transfer loss
        if 'transferred' in outputs and 'target_style' in outputs:
            transferred_style = self.encode_style(outputs['transferred'])
            style_transfer_loss = F.mse_loss(transferred_style, outputs['target_style'])
            losses['style_transfer'] = style_transfer_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def interpolate_styles(self, 
                          eeg_data: torch.Tensor,
                          style_a: int,
                          style_b: int,
                          alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between two styles.
        
        Args:
            eeg_data: Input EEG data
            style_a: First style index
            style_b: Second style index
            alpha: Interpolation weight (0 = style_a, 1 = style_b)
            
        Returns:
            Interpolated EEG
        """
        self.eval()
        
        with torch.no_grad():
            # Extract content
            content = self.encode_content(eeg_data)
            
            # Get style embeddings
            style_a_tensor = torch.tensor([style_a], device=eeg_data.device)
            style_b_tensor = torch.tensor([style_b], device=eeg_data.device)
            
            style_a_embed = self.style_embeddings(style_a_tensor)
            style_b_embed = self.style_embeddings(style_b_tensor)
            
            # Interpolate styles
            interpolated_style = (1 - alpha) * style_a_embed + alpha * style_b_embed
            
            # Generate interpolated EEG
            interpolated_eeg = self.decode(content, interpolated_style)
        
        return interpolated_eeg
    
    def analyze_style_space(self, eeg_data_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Analyze the learned style space.
        
        Args:
            eeg_data_list: List of EEG samples from different styles
            
        Returns:
            Style space analysis results
        """
        self.eval()
        
        all_styles = []
        all_contents = []
        
        with torch.no_grad():
            for eeg_data in eeg_data_list:
                style = self.encode_style(eeg_data)
                content = self.encode_content(eeg_data)
                
                all_styles.append(style)
                all_contents.append(content)
        
        # Stack all features
        styles = torch.cat(all_styles, dim=0)
        contents = torch.cat(all_contents, dim=0)
        
        # Compute style space statistics
        style_mean = torch.mean(styles, dim=0)
        style_std = torch.std(styles, dim=0)
        content_mean = torch.mean(contents, dim=0)
        content_std = torch.std(contents, dim=0)
        
        # Compute style separability (simplified)
        style_distances = torch.cdist(styles, styles)
        avg_style_distance = torch.mean(style_distances[style_distances > 0])
        
        return {
            'style_mean': style_mean,
            'style_std': style_std,
            'content_mean': content_mean,
            'content_std': content_std,
            'avg_style_distance': avg_style_distance,
            'num_samples': len(eeg_data_list)
        }


class StyleAugmenter:
    """Augment EEG data with style transfer."""
    
    def __init__(self, style_transfer_model: EEGStyleTransfer):
        self.model = style_transfer_model
    
    def augment_dataset(self, 
                       eeg_data: torch.Tensor,
                       original_styles: torch.Tensor,
                       augmentation_factor: int = 2) -> tuple:
        """Augment dataset with style transfers.
        
        Args:
            eeg_data: Original EEG data (batch, channels, seq_len)
            original_styles: Original style labels (batch,)
            augmentation_factor: How many additional samples per original
            
        Returns:
            Augmented EEG data and labels
        """
        self.model.eval()
        
        augmented_data = [eeg_data]
        augmented_labels = [original_styles]
        
        with torch.no_grad():
            for _ in range(augmentation_factor):
                # Random style transfer for each sample
                target_styles = torch.randint(0, self.model.num_styles, 
                                           (eeg_data.shape[0],), device=eeg_data.device)
                
                # Ensure different from original styles
                same_style_mask = target_styles == original_styles
                while torch.any(same_style_mask):
                    target_styles[same_style_mask] = torch.randint(0, self.model.num_styles, 
                                                                 (torch.sum(same_style_mask),), 
                                                                 device=eeg_data.device)
                    same_style_mask = target_styles == original_styles
                
                # Perform style transfer
                outputs = self.model(eeg_data, target_styles)
                transferred_data = outputs['transferred']
                
                augmented_data.append(transferred_data)
                augmented_labels.append(target_styles)
        
        # Combine all data
        final_data = torch.cat(augmented_data, dim=0)
        final_labels = torch.cat(augmented_labels, dim=0)
        
        return final_data, final_labels