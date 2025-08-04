"""BCI-GPT model architecture for imagined speech decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import warnings

try:
    from transformers import (
        GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
        LlamaForCausalLM, LlamaConfig, LlamaTokenizer
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers not available. Install with: pip install transformers")

from .fusion_layers import CrossAttentionFusion


class EEGEncoder(nn.Module):
    """EEG signal encoder with temporal convolution and spatial attention."""
    
    def __init__(self, 
                 n_channels: int = 9,
                 sampling_rate: int = 1000,
                 sequence_length: int = 1000,
                 hidden_dim: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Temporal convolution layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        
        # Calculate conv output size
        conv_output_size = self._get_conv_output_size()
        
        # Project to hidden dimension
        self.projection = nn.Linear(256, hidden_dim)
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(conv_output_size, hidden_dim)
        
    def _get_conv_output_size(self) -> int:
        """Calculate the output size after convolution layers."""
        # Each MaxPool1d(4) reduces size by 4x, applied 3 times
        return self.sequence_length // (4 ** 3)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EEG encoder.
        
        Args:
            x: EEG data (batch_size, n_channels, sequence_length)
            
        Returns:
            Encoded EEG features (batch_size, seq_len, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # Temporal convolution
        x = self.temporal_conv(x)  # (batch, 256, reduced_seq_len)
        
        # Transpose for transformer: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # Project to hidden dimension
        x = self.projection(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        seq_len = x.shape[1]
        if seq_len <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply spatial attention (self-attention across time steps)
        x_attended, _ = self.spatial_attention(x, x, x)
        x = x + x_attended  # Residual connection
        
        # Transformer encoding
        x = self.transformer(x)
        
        return x


class BCIGPTModel(nn.Module):
    """Complete BCI-GPT model for imagined speech decoding."""
    
    def __init__(self,
                 eeg_channels: int = 9,
                 eeg_sampling_rate: int = 1000,
                 sequence_length: int = 1000,
                 language_model: str = "gpt2-medium",
                 fusion_method: str = "cross_attention",
                 latent_dim: int = 256,
                 freeze_lm: bool = False):
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.sequence_length = sequence_length
        self.fusion_method = fusion_method
        self.latent_dim = latent_dim
        
        # EEG encoder
        self.eeg_encoder = EEGEncoder(
            n_channels=eeg_channels,
            sampling_rate=eeg_sampling_rate,
            sequence_length=sequence_length,
            hidden_dim=latent_dim
        )
        
        # Language model
        if HAS_TRANSFORMERS:
            if "gpt2" in language_model.lower():
                self.language_model = GPT2LMHeadModel.from_pretrained(language_model)
                self.tokenizer = GPT2Tokenizer.from_pretrained(language_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            elif "llama" in language_model.lower():
                self.language_model = LlamaForCausalLM.from_pretrained(language_model)
                self.tokenizer = LlamaTokenizer.from_pretrained(language_model)
            else:
                # Default to GPT-2
                self.language_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Create a simple language model if transformers not available
            self.language_model = self._create_simple_lm()
            self.tokenizer = None
            
        # Freeze language model if requested
        if freeze_lm and hasattr(self.language_model, 'parameters'):
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Fusion layer
        if fusion_method == "cross_attention":
            lm_hidden_size = getattr(self.language_model.config, 'hidden_size', 768)
            self.fusion_layer = CrossAttentionFusion(
                eeg_dim=latent_dim,
                text_dim=lm_hidden_size,
                hidden_dim=latent_dim
            )
        else:
            # Simple concatenation fusion
            lm_hidden_size = getattr(self.language_model.config, 'hidden_size', 768)
            self.fusion_layer = nn.Linear(latent_dim + lm_hidden_size, lm_hidden_size)
        
        # Output projection to vocabulary
        vocab_size = getattr(self.language_model.config, 'vocab_size', 50257)
        self.output_projection = nn.Linear(latent_dim, vocab_size)
        
    def _create_simple_lm(self) -> nn.Module:
        """Create a simple language model if transformers not available."""
        vocab_size = 50257  # GPT-2 vocab size
        hidden_size = 768
        
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.config = type('Config', (), {
                    'vocab_size': vocab_size,
                    'hidden_size': hidden_size
                })()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        batch_first=True
                    ),
                    num_layers=6
                )
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                
                # Return in transformers format
                result = type('ModelOutput', (), {})()
                result.logits = logits
                result.last_hidden_state = x
                return result
        
        return SimpleLM(vocab_size, hidden_size)
    
    def forward(self, 
                eeg_data: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through BCI-GPT model.
        
        Args:
            eeg_data: EEG signals (batch_size, channels, sequence_length)
            input_ids: Token IDs for language model (batch_size, seq_len)
            attention_mask: Attention mask for tokens (batch_size, seq_len)
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode EEG signals
        eeg_features = self.eeg_encoder(eeg_data)  # (batch, seq_len, latent_dim)
        
        outputs = {}
        outputs['eeg_features'] = eeg_features
        
        # If no text input, generate from EEG alone
        if input_ids is None:
            # Direct EEG to token mapping
            token_logits = self.output_projection(eeg_features)
            outputs['logits'] = token_logits
            return outputs
        
        # Language model forward pass
        if HAS_TRANSFORMERS:
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_features = lm_outputs.last_hidden_state
            text_logits = lm_outputs.logits
        else:
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = lm_outputs.last_hidden_state
            text_logits = lm_outputs.logits
        
        # Fusion of EEG and text features
        if self.fusion_method == "cross_attention":
            fused_features = self.fusion_layer(eeg_features, text_features)
        else:
            # Simple concatenation and projection
            # Repeat EEG features to match text sequence length
            text_seq_len = text_features.shape[1]
            eeg_mean = torch.mean(eeg_features, dim=1, keepdim=True)  # Average pool
            eeg_expanded = eeg_mean.repeat(1, text_seq_len, 1)
            
            concatenated = torch.cat([eeg_expanded, text_features], dim=-1)
            fused_features = self.fusion_layer(concatenated)
        
        # Generate final logits
        if self.fusion_method == "cross_attention":
            # Use fused features for final prediction
            final_logits = self.output_projection(fused_features)
        else:
            # Add EEG influence to language model logits
            eeg_logits = self.output_projection(
                torch.mean(eeg_features, dim=1, keepdim=True).repeat(1, text_seq_len, 1)
            )
            final_logits = text_logits + 0.1 * eeg_logits  # Weighted combination
        
        outputs.update({
            'text_features': text_features,
            'fused_features': fused_features,
            'logits': final_logits,
            'text_logits': text_logits
        })
        
        return outputs
    
    def generate_text(self, 
                     eeg_data: torch.Tensor,
                     max_length: int = 50,
                     temperature: float = 1.0,
                     top_k: int = 50,
                     top_p: float = 0.9) -> List[str]:
        """Generate text from EEG signals.
        
        Args:
            eeg_data: EEG signals (batch_size, channels, sequence_length)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            List of generated text strings
        """
        self.eval()
        batch_size = eeg_data.shape[0]
        device = eeg_data.device
        
        if not self.tokenizer:
            # If no tokenizer available, return placeholder
            return ["Generated text from EEG"] * batch_size
        
        # Encode EEG
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg_data)
            
            # Initialize with BOS token
            input_ids = torch.full((batch_size, 1), 
                                 self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
                                 device=device, dtype=torch.long)
            
            generated_texts = []
            
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(eeg_data, input_ids)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Append to sequences
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                
                # Check for EOS token
                if torch.all(next_tokens.squeeze() == self.tokenizer.eos_token_id):
                    break
            
            # Decode generated sequences
            for i in range(batch_size):
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                generated_texts.append(text)
        
        return generated_texts
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'BCIGPTModel':
        """Load a pretrained BCI-GPT model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model configuration
        config = checkpoint.get('config', {})
        
        # Create model
        model = cls(**config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def save_pretrained(self, save_path: str) -> None:
        """Save the BCI-GPT model."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'eeg_channels': self.eeg_channels,
                'sequence_length': self.sequence_length,
                'fusion_method': self.fusion_method,
                'latent_dim': self.latent_dim,
            }
        }
        
        torch.save(checkpoint, save_path)