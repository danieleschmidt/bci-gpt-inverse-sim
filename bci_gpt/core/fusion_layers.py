"""Multi-modal fusion layers for EEG and language model integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing EEG and text features."""
    
    def __init__(self, 
                 eeg_dim: int = 256,
                 text_dim: int = 768,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project EEG and text features to common dimension
        self.eeg_projection = nn.Linear(eeg_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention layers
        self.eeg_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.text_to_eeg_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.eeg_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.eeg_ln1 = nn.LayerNorm(hidden_dim)
        self.eeg_ln2 = nn.LayerNorm(hidden_dim)
        self.text_ln1 = nn.LayerNorm(hidden_dim)
        self.text_ln2 = nn.LayerNorm(hidden_dim)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                eeg_features: torch.Tensor,
                text_features: torch.Tensor,
                eeg_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through cross-attention fusion.
        
        Args:
            eeg_features: EEG features (batch_size, eeg_seq_len, eeg_dim)
            text_features: Text features (batch_size, text_seq_len, text_dim)
            eeg_mask: Attention mask for EEG (batch_size, eeg_seq_len)
            text_mask: Attention mask for text (batch_size, text_seq_len)
            
        Returns:
            Fused features (batch_size, text_seq_len, hidden_dim)
        """
        # Project to common dimension
        eeg_proj = self.eeg_projection(eeg_features)  # (batch, eeg_seq, hidden)
        text_proj = self.text_projection(text_features)  # (batch, text_seq, hidden)
        
        # EEG attending to text
        eeg_attended, _ = self.eeg_to_text_attention(
            query=eeg_proj,
            key=text_proj,
            value=text_proj,
            key_padding_mask=text_mask
        )
        eeg_attended = self.eeg_ln1(eeg_proj + eeg_attended)
        eeg_attended = self.eeg_ln2(eeg_attended + self.eeg_ffn(eeg_attended))
        
        # Text attending to EEG
        text_attended, _ = self.text_to_eeg_attention(
            query=text_proj,
            key=eeg_proj,
            value=eeg_proj,
            key_padding_mask=eeg_mask
        )
        text_attended = self.text_ln1(text_proj + text_attended)
        text_attended = self.text_ln2(text_attended + self.text_ffn(text_attended))
        
        # Global pooling for EEG features to match text sequence length
        if eeg_attended.shape[1] != text_attended.shape[1]:
            # Average pool EEG features
            eeg_pooled = torch.mean(eeg_attended, dim=1, keepdim=True)  # (batch, 1, hidden)
            eeg_pooled = eeg_pooled.repeat(1, text_attended.shape[1], 1)  # (batch, text_seq, hidden)
        else:
            eeg_pooled = eeg_attended
        
        # Concatenate and fuse
        fused_input = torch.cat([eeg_pooled, text_attended], dim=-1)  # (batch, text_seq, 2*hidden)
        fused_output = self.fusion_layer(fused_input)  # (batch, text_seq, hidden)
        
        return fused_output


class MultiModalFusion(nn.Module):
    """Multi-modal fusion supporting various modalities beyond EEG and text."""
    
    def __init__(self,
                 modality_dims: dict,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            modality_dims: Dict mapping modality names to their feature dimensions
                          e.g., {'eeg': 256, 'emg': 64, 'eye_tracking': 32}
            hidden_dim: Hidden dimension for fusion
            num_heads: Number of attention heads
            num_layers: Number of fusion layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_names = list(modality_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Projection layers for each modality
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in modality_dims.items()
        })
        
        # Multi-head attention for cross-modal interactions
        self.cross_modal_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization and feed-forward networks
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])
        
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Final aggregation
        self.aggregation_weights = nn.Parameter(
            torch.ones(len(self.modality_names)) / len(self.modality_names)
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, modality_features: dict) -> torch.Tensor:
        """Forward pass through multi-modal fusion.
        
        Args:
            modality_features: Dict mapping modality names to feature tensors
                             Each tensor shape: (batch_size, seq_len, feature_dim)
            
        Returns:
            Fused features (batch_size, seq_len, hidden_dim)
        """
        # Project all modalities to common dimension
        projected_features = {}
        max_seq_len = 0
        
        for name in self.modality_names:
            if name in modality_features:
                features = modality_features[name]
                projected = self.modality_projections[name](features)
                projected_features[name] = projected
                max_seq_len = max(max_seq_len, features.shape[1])
        
        # Align sequence lengths (pad or pool as needed)
        aligned_features = []
        for name in self.modality_names:
            if name in projected_features:
                features = projected_features[name]
                seq_len = features.shape[1]
                
                if seq_len < max_seq_len:
                    # Pad shorter sequences
                    padding = max_seq_len - seq_len
                    features = F.pad(features, (0, 0, 0, padding), mode='replicate')
                elif seq_len > max_seq_len:
                    # Pool longer sequences
                    features = F.adaptive_avg_pool1d(
                        features.transpose(1, 2), max_seq_len
                    ).transpose(1, 2)
                
                aligned_features.append(features)
        
        if not aligned_features:
            raise ValueError("No valid modality features provided")
        
        # Stack features for attention computation
        stacked_features = torch.stack(aligned_features, dim=1)  # (batch, n_modalities, seq_len, hidden)
        batch_size, n_modalities, seq_len, hidden_dim = stacked_features.shape
        
        # Reshape for attention: (batch * seq_len, n_modalities, hidden)
        reshaped_features = stacked_features.transpose(1, 2).reshape(
            batch_size * seq_len, n_modalities, hidden_dim
        )
        
        # Apply cross-modal attention layers
        current_features = reshaped_features
        
        for layer_idx in range(self.num_layers):
            # Multi-head attention
            attended_features, _ = self.cross_modal_attention[layer_idx](
                query=current_features,
                key=current_features,
                value=current_features
            )
            
            # Residual connection and layer norm
            current_features = self.layer_norms[layer_idx * 2](
                current_features + attended_features
            )
            
            # Feed-forward network
            ff_output = self.feed_forwards[layer_idx](current_features)
            
            # Residual connection and layer norm
            current_features = self.layer_norms[layer_idx * 2 + 1](
                current_features + ff_output
            )
        
        # Reshape back: (batch, seq_len, n_modalities, hidden)
        final_features = current_features.reshape(
            batch_size, seq_len, n_modalities, hidden_dim
        )
        
        # Weighted aggregation across modalities
        weights = F.softmax(self.aggregation_weights, dim=0)
        aggregated_features = torch.sum(
            final_features * weights.view(1, 1, -1, 1), dim=2
        )  # (batch, seq_len, hidden)
        
        # Final output projection
        output = self.output_projection(aggregated_features)
        
        return output


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns to weight different modalities dynamically."""
    
    def __init__(self,
                 eeg_dim: int = 256,
                 text_dim: int = 768,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Modality projections
        self.eeg_projection = nn.Linear(eeg_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Attention mechanism for dynamic weighting
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 modalities
            nn.Softmax(dim=-1)
        )
        
        # Cross-attention for modality interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                eeg_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive fusion.
        
        Args:
            eeg_features: EEG features (batch_size, eeg_seq_len, eeg_dim)
            text_features: Text features (batch_size, text_seq_len, text_dim)
            
        Returns:
            Adaptively fused features (batch_size, text_seq_len, hidden_dim)
        """
        # Project to common dimension
        eeg_proj = self.eeg_projection(eeg_features)
        text_proj = self.text_projection(text_features)
        
        # Pool EEG features to match text sequence length
        eeg_pooled = torch.mean(eeg_proj, dim=1, keepdim=True)  # (batch, 1, hidden)
        eeg_expanded = eeg_pooled.repeat(1, text_proj.shape[1], 1)  # (batch, text_seq, hidden)
        
        # Compute adaptive weights
        combined_features = torch.cat([eeg_expanded, text_proj], dim=-1)  # (batch, text_seq, 2*hidden)
        attention_weights = self.attention_net(combined_features)  # (batch, text_seq, 2)
        
        # Apply weights
        eeg_weighted = eeg_expanded * attention_weights[:, :, 0:1]
        text_weighted = text_proj * attention_weights[:, :, 1:2]
        
        # Cross-attention between weighted features
        fused_features, _ = self.cross_attention(
            query=text_weighted,
            key=eeg_weighted,
            value=eeg_weighted
        )
        
        # Residual connection
        fused_features = text_weighted + fused_features
        
        # Final output
        output = self.output_net(fused_features)
        
        return output