"""Tests for inverse GAN functionality."""

import pytest
import torch
import numpy as np

from bci_gpt.core.inverse_gan import Generator, Discriminator, InverseSimulator


class TestGenerator:
    """Test cases for GAN generator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = Generator(
            text_embedding_dim=768,
            noise_dim=100,
            eeg_channels=9,
            eeg_sequence_length=1000,
            hidden_dims=[256, 512]
        )
        
        self.batch_size = 2
        self.text_embeddings = torch.randn(self.batch_size, 768)
        self.noise = torch.randn(self.batch_size, 100)
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        assert self.generator.text_embedding_dim == 768
        assert self.generator.noise_dim == 100
        assert self.generator.eeg_channels == 9
        assert self.generator.eeg_sequence_length == 1000
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        fake_eeg = self.generator(self.text_embeddings, self.noise)
        
        # Check output shape
        expected_shape = (self.batch_size, 9, 1000)
        assert fake_eeg.shape == expected_shape
        
        # Check output range (should be bounded by tanh)
        assert torch.all(fake_eeg >= -1)
        assert torch.all(fake_eeg <= 1)
        
        # Check that output is not all zeros
        assert not torch.all(fake_eeg == 0)
    
    def test_generator_without_noise(self):
        """Test generator with automatic noise sampling."""
        fake_eeg = self.generator(self.text_embeddings)  # No noise provided
        
        assert fake_eeg.shape == (self.batch_size, 9, 1000)
        assert not torch.all(fake_eeg == 0)
    
    def test_generator_gradient_flow(self):
        """Test gradient flow through generator."""
        self.text_embeddings.requires_grad_(True)
        fake_eeg = self.generator(self.text_embeddings, self.noise)
        
        loss = fake_eeg.sum()
        loss.backward()
        
        # Check gradients exist
        assert self.text_embeddings.grad is not None
        assert not torch.all(self.text_embeddings.grad == 0)
    
    def test_generator_different_batch_sizes(self):
        """Test generator with different batch sizes."""
        for batch_size in [1, 4, 8]:
            text_emb = torch.randn(batch_size, 768)
            fake_eeg = self.generator(text_emb)
            assert fake_eeg.shape[0] == batch_size


class TestDiscriminator:
    """Test cases for GAN discriminator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.discriminator = Discriminator(
            eeg_channels=9,
            eeg_sequence_length=1000,
            text_embedding_dim=768,
            hidden_dims=[512, 256]
        )
        
        self.batch_size = 2
        self.eeg_data = torch.randn(self.batch_size, 9, 1000)
        self.text_embeddings = torch.randn(self.batch_size, 768)
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        assert self.discriminator.eeg_channels == 9
        assert self.discriminator.eeg_sequence_length == 1000
        assert self.discriminator.text_embedding_dim == 768
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        outputs = self.discriminator(self.eeg_data, self.text_embeddings)
        
        # Check output structure
        assert isinstance(outputs, dict)
        assert 'main_output' in outputs
        assert 'spectral_output' in outputs
        assert 'combined_output' in outputs
        
        # Check output shapes
        main_output = outputs['main_output']
        spectral_output = outputs['spectral_output']
        combined_output = outputs['combined_output']
        
        assert main_output.shape[0] == self.batch_size
        assert spectral_output.shape[0] == self.batch_size
        assert combined_output.shape[0] == self.batch_size
        
        # Check that spectral output is in [0, 1] (sigmoid activated)
        assert torch.all(spectral_output >= 0)
        assert torch.all(spectral_output <= 1)
    
    def test_discriminator_without_text(self):
        """Test discriminator without text embeddings."""
        outputs = self.discriminator(self.eeg_data)  # No text embeddings
        
        assert isinstance(outputs, dict)
        assert 'main_output' in outputs
    
    def test_spectral_feature_extraction(self):
        """Test spectral feature extraction."""
        spectral_features = self.discriminator._extract_spectral_features(self.eeg_data)
        
        # Should return features for each batch item
        assert spectral_features.shape[0] == self.batch_size
        assert spectral_features.shape[1] == 5  # 5 frequency bands
        
        # Features should be non-negative (power values)
        assert torch.all(spectral_features >= 0)
    
    def test_conv_output_size_calculation(self):
        """Test convolution output size calculation."""
        conv_size = self.discriminator._calculate_conv_output_size()
        assert isinstance(conv_size, int)
        assert conv_size > 0
    
    def test_discriminator_gradient_flow(self):
        """Test gradient flow through discriminator."""
        self.eeg_data.requires_grad_(True)
        outputs = self.discriminator(self.eeg_data, self.text_embeddings)
        
        loss = outputs['combined_output'].sum()
        loss.backward()
        
        assert self.eeg_data.grad is not None
        assert not torch.all(self.eeg_data.grad == 0)


class TestInverseSimulator:
    """Test cases for complete inverse simulator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.inverse_sim = InverseSimulator(
            generator_layers=[256, 512],
            discriminator_layers=[512, 256],
            noise_dim=50,
            eeg_channels=4,  # Smaller for faster testing
            eeg_sequence_length=500,
            text_embedding_dim=384,
            conditional=True
        )
        
        self.batch_size = 2
        self.text_embeddings = torch.randn(self.batch_size, 384)
        self.real_eeg = torch.randn(self.batch_size, 4, 500)
    
    def test_inverse_simulator_initialization(self):
        """Test inverse simulator initialization."""
        assert hasattr(self.inverse_sim, 'generator')
        assert hasattr(self.inverse_sim, 'discriminator')
        assert self.inverse_sim.noise_dim == 50
        assert self.inverse_sim.conditional == True
    
    def test_generate_synthetic_eeg(self):
        """Test synthetic EEG generation."""
        synthetic_eeg = self.inverse_sim.generate(self.text_embeddings)
        
        # Check shape
        expected_shape = (self.batch_size, 4, 500)
        assert synthetic_eeg.shape == expected_shape
        
        # Check that it's not all zeros
        assert not torch.all(synthetic_eeg == 0)
    
    def test_generate_multiple_samples(self):
        """Test generating multiple samples per text."""
        num_samples = 3
        synthetic_eeg = self.inverse_sim.generate(
            self.text_embeddings, 
            num_samples=num_samples
        )
        
        expected_shape = (self.batch_size * num_samples, 4, 500)
        assert synthetic_eeg.shape == expected_shape
    
    def test_discriminate_real_vs_fake(self):
        """Test discrimination between real and fake EEG."""
        # Generate fake EEG
        fake_eeg = self.inverse_sim.generate(self.text_embeddings)
        
        # Discriminate real EEG
        real_outputs = self.inverse_sim.discriminate(self.real_eeg, self.text_embeddings)
        
        # Discriminate fake EEG
        fake_outputs = self.inverse_sim.discriminate(fake_eeg, self.text_embeddings)
        
        # Check that outputs have correct structure
        for outputs in [real_outputs, fake_outputs]:
            assert isinstance(outputs, dict)
            assert 'main_output' in outputs
            assert 'spectral_output' in outputs
    
    def test_generator_loss_computation(self):
        """Test generator loss computation."""
        # Generate fake EEG
        fake_eeg = self.inverse_sim.generate(self.text_embeddings)
        
        # Get discriminator outputs for fake EEG
        fake_outputs = self.inverse_sim.discriminate(fake_eeg, self.text_embeddings)
        
        # Compute generator loss
        gen_losses = self.inverse_sim.compute_generator_loss(
            fake_outputs, self.real_eeg, fake_eeg
        )
        
        # Check loss structure
        assert isinstance(gen_losses, dict)
        assert 'total_loss' in gen_losses
        assert 'adversarial_loss' in gen_losses
        assert 'spectral_loss' in gen_losses
        
        # Check that losses are scalars
        for loss_name, loss_value in gen_losses.items():
            assert loss_value.dim() == 0  # Scalar
            assert not torch.isnan(loss_value)
    
    def test_discriminator_loss_computation(self):
        """Test discriminator loss computation."""
        # Generate fake EEG
        fake_eeg = self.inverse_sim.generate(self.text_embeddings)
        
        # Get discriminator outputs
        real_outputs = self.inverse_sim.discriminate(self.real_eeg, self.text_embeddings)
        fake_outputs = self.inverse_sim.discriminate(fake_eeg.detach(), self.text_embeddings)
        
        # Compute discriminator loss
        disc_losses = self.inverse_sim.compute_discriminator_loss(real_outputs, fake_outputs)
        
        # Check loss structure
        assert isinstance(disc_losses, dict)
        assert 'total_loss' in disc_losses
        assert 'main_loss' in disc_losses
        assert 'real_loss' in disc_losses
        assert 'fake_loss' in disc_losses
        
        # Check that losses are scalars
        for loss_name, loss_value in disc_losses.items():
            assert loss_value.dim() == 0
            assert not torch.isnan(loss_value)
    
    def test_unconditional_mode(self):
        """Test unconditional inverse simulator."""
        unconditional_sim = InverseSimulator(
            conditional=False,
            text_embedding_dim=0  # No text conditioning
        )
        
        # Generate without text embeddings
        noise = torch.randn(2, 100)
        synthetic_eeg = unconditional_sim.generator(
            torch.zeros(2, 1),  # Dummy text embeddings
            noise
        )
        
        assert synthetic_eeg.shape[0] == 2
    
    def test_training_simulation(self):
        """Test simulated training step."""
        # Set to training mode
        self.inverse_sim.train()
        
        # Generate fake EEG
        fake_eeg = self.inverse_sim.generate(self.text_embeddings)
        
        # Discriminator forward pass
        real_outputs = self.inverse_sim.discriminate(self.real_eeg, self.text_embeddings)
        fake_outputs = self.inverse_sim.discriminate(fake_eeg, self.text_embeddings)
        
        # Compute losses
        gen_losses = self.inverse_sim.compute_generator_loss(
            fake_outputs, self.real_eeg, fake_eeg
        )
        disc_losses = self.inverse_sim.compute_discriminator_loss(real_outputs, fake_outputs)
        
        # Test backward pass
        gen_loss = gen_losses['total_loss']
        gen_loss.backward(retain_graph=True)
        
        disc_loss = disc_losses['total_loss'] 
        disc_loss.backward()
        
        # Check that gradients were computed
        gen_has_grads = any(p.grad is not None for p in self.inverse_sim.generator.parameters())
        disc_has_grads = any(p.grad is not None for p in self.inverse_sim.discriminator.parameters())
        
        assert gen_has_grads
        assert disc_has_grads
    
    def test_device_movement(self):
        """Test moving inverse simulator to different devices."""
        device = torch.device('cpu')
        self.inverse_sim.to(device)
        
        # Test generation on correct device
        text_emb = torch.randn(1, 384, device=device)
        synthetic_eeg = self.inverse_sim.generate(text_emb)
        
        assert synthetic_eeg.device == device
    
    def test_eval_mode(self):
        """Test evaluation mode."""
        self.inverse_sim.eval()
        
        with torch.no_grad():
            synthetic_eeg = self.inverse_sim.generate(self.text_embeddings)
            outputs = self.inverse_sim.discriminate(synthetic_eeg, self.text_embeddings)
        
        assert synthetic_eeg.shape == (self.batch_size, 4, 500)
        assert isinstance(outputs, dict)


class TestInverseSimulatorIntegration:
    """Integration tests for inverse simulator."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.inverse_sim = InverseSimulator(
            generator_layers=[128, 256],
            discriminator_layers=[256, 128],
            noise_dim=32,
            eeg_channels=2,  # Very small for fast testing
            eeg_sequence_length=100,
            text_embedding_dim=64
        )
    
    def test_end_to_end_generation(self):
        """Test end-to-end generation pipeline."""
        # Create text embedding
        text_emb = torch.randn(1, 64)
        
        # Generate EEG
        self.inverse_sim.eval()
        with torch.no_grad():
            synthetic_eeg = self.inverse_sim.generate(text_emb)
        
        # Validate EEG
        assert synthetic_eeg.shape == (1, 2, 100)
        assert torch.all(torch.isfinite(synthetic_eeg))
        
        # Check EEG has reasonable properties
        eeg_std = torch.std(synthetic_eeg)
        assert eeg_std > 0.01  # Should have some variation
        assert eeg_std < 10.0   # Should not be too extreme
    
    def test_consistency_across_runs(self):
        """Test that model produces consistent results with same seed."""
        text_emb = torch.randn(1, 64)
        
        # Generate with fixed noise
        noise = torch.randn(1, 32)
        
        self.inverse_sim.eval()
        with torch.no_grad():
            eeg1 = self.inverse_sim.generate(text_emb, noise)
            eeg2 = self.inverse_sim.generate(text_emb, noise)
        
        # Should be identical with same noise
        assert torch.allclose(eeg1, eeg2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])