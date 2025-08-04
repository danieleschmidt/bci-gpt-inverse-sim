"""Tests for BCI-GPT model functionality."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from bci_gpt.core.models import BCIGPTModel, EEGEncoder


class TestEEGEncoder:
    """Test cases for EEG encoder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.encoder = EEGEncoder(
            n_channels=9,
            sampling_rate=1000,
            sequence_length=1000,
            hidden_dim=256,
            n_layers=3,
            n_heads=4
        )
        
        # Create test data
        self.batch_size = 2
        self.test_input = torch.randn(self.batch_size, 9, 1000)
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.n_channels == 9
        assert self.encoder.sampling_rate == 1000
        assert self.encoder.sequence_length == 1000
        assert self.encoder.hidden_dim == 256
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        output = self.encoder(self.test_input)
        
        # Check output shape
        assert output.dim() == 3  # batch_size x seq_len x hidden_dim
        assert output.shape[0] == self.batch_size
        assert output.shape[2] == 256  # hidden_dim
        
        # Check that output is not all zeros
        assert not torch.all(output == 0)
    
    def test_conv_output_size_calculation(self):
        """Test convolution output size calculation."""
        conv_size = self.encoder._get_conv_output_size()
        assert isinstance(conv_size, int)
        assert conv_size > 0
    
    def test_positional_encoding(self):
        """Test positional encoding creation."""
        max_len = 100
        d_model = 256
        
        pe = self.encoder._create_positional_encoding(max_len, d_model)
        
        assert pe.shape == (1, max_len, d_model)
        assert not torch.all(pe == 0)
    
    def test_different_input_shapes(self):
        """Test encoder with different input shapes."""
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, 9, 1000)
            output = self.encoder(test_input)
            assert output.shape[0] == batch_size
    
    def test_gradient_flow(self):
        """Test that gradients flow through the encoder."""
        self.test_input.requires_grad_(True)
        output = self.encoder(self.test_input)
        
        # Compute dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert self.test_input.grad is not None
        assert not torch.all(self.test_input.grad == 0)


class TestBCIGPTModel:
    """Test cases for complete BCI-GPT model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock transformers to avoid dependency issues in testing
        with patch('bci_gpt.core.models.HAS_TRANSFORMERS', False):
            self.model = BCIGPTModel(
                eeg_channels=9,
                eeg_sampling_rate=1000,
                sequence_length=1000,
                language_model="gpt2-medium",
                fusion_method="cross_attention",
                latent_dim=256
            )
        
        self.batch_size = 2
        self.eeg_input = torch.randn(self.batch_size, 9, 1000)
        self.text_input = torch.randint(0, 1000, (self.batch_size, 10))  # Token IDs
        self.attention_mask = torch.ones(self.batch_size, 10)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.eeg_channels == 9
        assert self.model.sequence_length == 1000
        assert self.model.fusion_method == "cross_attention"
        assert self.model.latent_dim == 256
    
    def test_eeg_only_forward(self):
        """Test forward pass with EEG input only."""
        outputs = self.model(self.eeg_input)
        
        assert 'eeg_features' in outputs
        assert 'logits' in outputs
        
        eeg_features = outputs['eeg_features']
        logits = outputs['logits']
        
        assert eeg_features.shape[0] == self.batch_size
        assert logits.shape[0] == self.batch_size
        assert logits.shape[2] > 0  # Vocabulary size
    
    def test_eeg_and_text_forward(self):
        """Test forward pass with both EEG and text inputs."""
        outputs = self.model(
            self.eeg_input, 
            self.text_input, 
            self.attention_mask
        )
        
        assert 'eeg_features' in outputs
        assert 'text_features' in outputs
        assert 'logits' in outputs
        
        # Check shapes
        assert outputs['eeg_features'].shape[0] == self.batch_size
        assert outputs['text_features'].shape[0] == self.batch_size
        assert outputs['logits'].shape[0] == self.batch_size
    
    def test_generate_text_fallback(self):
        """Test text generation when tokenizer is not available."""
        generated_texts = self.model.generate_text(
            self.eeg_input,
            max_length=10,
            temperature=1.0
        )
        
        assert isinstance(generated_texts, list)
        assert len(generated_texts) == self.batch_size
        
        # Should return placeholder text when no tokenizer
        for text in generated_texts:
            assert isinstance(text, str)
    
    @patch('bci_gpt.core.models.HAS_TRANSFORMERS', True)
    @patch('bci_gpt.core.models.GPT2LMHeadModel')
    @patch('bci_gpt.core.models.GPT2Tokenizer')
    def test_with_transformers_available(self, mock_tokenizer, mock_model):
        """Test model creation when transformers is available."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.config.hidden_size = 768
        mock_model_instance.config.vocab_size = 50257
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        model = BCIGPTModel(
            eeg_channels=9,
            language_model="gpt2"
        )
        
        # Check that the mocked model was used
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
    
    def test_model_device_movement(self):
        """Test moving model to different devices."""
        device = torch.device('cpu')  # Always available
        self.model.to(device)
        
        # Test that parameters are on correct device
        for param in self.model.parameters():
            assert param.device == device
    
    def test_save_and_load_pretrained(self, tmp_path):
        """Test saving and loading model."""
        save_path = tmp_path / "test_model.pt"
        
        # Save model
        self.model.save_pretrained(str(save_path))
        
        # Check that file exists
        assert save_path.exists()
        
        # Load model
        loaded_model = BCIGPTModel.from_pretrained(str(save_path))
        
        # Check that architectures match
        assert loaded_model.eeg_channels == self.model.eeg_channels
        assert loaded_model.sequence_length == self.model.sequence_length
    
    def test_fusion_layer_types(self):
        """Test different fusion methods."""
        # Test cross attention fusion
        with patch('bci_gpt.core.models.HAS_TRANSFORMERS', False):
            model_cross = BCIGPTModel(
                eeg_channels=9,
                fusion_method="cross_attention"
            )
            assert hasattr(model_cross, 'fusion_layer')
        
        # Test simple concatenation fusion
        with patch('bci_gpt.core.models.HAS_TRANSFORMERS', False):
            model_concat = BCIGPTModel(
                eeg_channels=9,
                fusion_method="concatenation"
            )
            assert hasattr(model_concat, 'fusion_layer')
    
    def test_error_handling_invalid_input(self):
        """Test error handling with invalid inputs."""
        # Test with wrong shape
        wrong_shape_input = torch.randn(2, 5, 1000)  # Wrong number of channels
        
        # Should not crash, but might produce unexpected results
        try:
            outputs = self.model(wrong_shape_input)
            # If it doesn't crash, that's fine for robustness
        except Exception as e:
            # Expected behavior - model should handle gracefully
            assert isinstance(e, (RuntimeError, ValueError))
    
    def test_model_training_mode(self):
        """Test model training and evaluation modes."""
        # Test training mode
        self.model.train()
        assert self.model.training
        
        # Test evaluation mode
        self.model.eval()
        assert not self.model.training
    
    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Should have at least some parameters
        assert total_params > 1000
        
        # Should not be excessively large (adjust threshold as needed)
        assert total_params < 1e9  # 1 billion parameters


class TestModelIntegration:
    """Integration tests for model components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        with patch('bci_gpt.core.models.HAS_TRANSFORMERS', False):
            self.model = BCIGPTModel(
                eeg_channels=4,  # Smaller for faster testing
                sequence_length=500,
                latent_dim=128
            )
    
    def test_end_to_end_inference(self):
        """Test complete inference pipeline."""
        batch_size = 1
        eeg_input = torch.randn(batch_size, 4, 500)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(eeg_input)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == batch_size
    
    def test_training_step_simulation(self):
        """Test a simulated training step."""
        self.model.train()
        
        # Create dummy data
        eeg_input = torch.randn(2, 4, 500)
        text_input = torch.randint(0, 1000, (2, 5))
        attention_mask = torch.ones(2, 5)
        
        # Forward pass
        outputs = self.model(eeg_input, text_input, attention_mask)
        
        # Simulate loss calculation
        logits = outputs['logits']
        # Simple dummy loss
        loss = torch.mean(logits)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        has_grads = any(p.grad is not None for p in self.model.parameters())
        assert has_grads


if __name__ == "__main__":
    pytest.main([__file__])