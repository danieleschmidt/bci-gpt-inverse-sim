"""Text-to-EEG generation using trained inverse models."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
import warnings

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers not available for text encoding")

from ..core.inverse_gan import InverseSimulator
from ..preprocessing.eeg_processor import EEGProcessor
from .validation import SyntheticEEGValidator


@dataclass
class GenerationConfig:
    """Configuration for EEG generation from text."""
    duration: float = 2.0  # seconds
    sampling_rate: int = 1000  # Hz
    noise_level: float = 0.1
    style: str = "imagined_speech"  # or "inner_monologue", "subvocalization"
    num_samples: int = 1
    seed: Optional[int] = None
    temperature: float = 1.0
    top_k: Optional[int] = None
    use_guidance: bool = True
    guidance_scale: float = 1.5


class TextToEEG:
    """Generate synthetic EEG signals from text input."""
    
    def __init__(self,
                 inverse_model_path: str,
                 text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 eeg_template_path: Optional[str] = None,
                 device: str = "cuda"):
        """Initialize text-to-EEG generator.
        
        Args:
            inverse_model_path: Path to trained inverse GAN model
            text_encoder_model: HuggingFace model for text encoding
            eeg_template_path: Optional path to EEG template statistics
            device: Device for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load inverse simulator
        try:
            self.inverse_simulator = self._load_inverse_model(inverse_model_path)
        except Exception as e:
            warnings.warn(f"Could not load inverse model: {e}. Creating dummy model.")
            self.inverse_simulator = InverseSimulator()
            
        self.inverse_simulator.to(self.device)
        self.inverse_simulator.eval()
        
        # Load text encoder
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model)
                self.text_encoder = AutoModel.from_pretrained(text_encoder_model)
                self.text_encoder.to(self.device)
                self.text_encoder.eval()
            except Exception as e:
                warnings.warn(f"Could not load text encoder: {e}")
                self.tokenizer = None
                self.text_encoder = None
        else:
            self.tokenizer = None
            self.text_encoder = None
        
        # Load EEG template statistics if available
        self.eeg_template = self._load_eeg_template(eeg_template_path) if eeg_template_path else None
        
        # EEG processor for post-processing
        self.eeg_processor = EEGProcessor()
        
        # Validator for quality assessment
        self.validator = SyntheticEEGValidator()
        
        # Style-specific parameters
        self.style_configs = {
            "imagined_speech": {
                "frequency_emphasis": {"alpha": 1.2, "beta": 1.1, "gamma": 0.9},
                "temporal_smoothing": 0.8,
                "spatial_coherence": 1.0
            },
            "inner_monologue": {
                "frequency_emphasis": {"alpha": 1.3, "beta": 0.9, "gamma": 0.8},
                "temporal_smoothing": 1.2,
                "spatial_coherence": 0.9
            },
            "subvocalization": {
                "frequency_emphasis": {"alpha": 0.9, "beta": 1.3, "gamma": 1.1},
                "temporal_smoothing": 0.7,
                "spatial_coherence": 1.1
            }
        }
    
    def _load_inverse_model(self, model_path: str) -> InverseSimulator:
        """Load trained inverse simulator model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint.get('config', {})
        
        # Create model
        model = InverseSimulator(**config)
        
        # Load state dict
        if 'inverse_simulator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['inverse_simulator_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _load_eeg_template(self, template_path: str) -> Dict:
        """Load EEG template statistics."""
        try:
            if template_path.endswith('.npy'):
                template_data = np.load(template_path, allow_pickle=True).item()
            elif template_path.endswith('.pkl'):
                import pickle
                with open(template_path, 'rb') as f:
                    template_data = pickle.load(f)
            else:
                # Try loading as numpy array
                template_data = {'mean': np.load(template_path)}
            
            return template_data
        except Exception as e:
            warnings.warn(f"Could not load EEG template: {e}")
            return None
    
    def generate(self,
                 text: Union[str, List[str]],
                 config: Optional[GenerationConfig] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate synthetic EEG from text.
        
        Args:
            text: Input text string or list of strings
            config: Generation configuration
            
        Returns:
            Generated EEG signals (channels x samples) or list of arrays
        """
        if config is None:
            config = GenerationConfig()
        
        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Handle single text vs batch
        if isinstance(text, str):
            texts = [text]
            return_single = True
        else:
            texts = text
            return_single = False
        
        generated_eeg_list = []
        
        for text_input in texts:
            # Encode text
            text_embedding = self._encode_text(text_input)
            if text_embedding is None:
                # Fallback to random embedding
                text_embedding = torch.randn(1, 768, device=self.device)
            
            # Generate multiple samples if requested
            all_samples = []
            for _ in range(config.num_samples):
                # Generate EEG
                eeg_signal = self._generate_eeg_from_embedding(text_embedding, config)
                
                # Post-process
                eeg_signal = self._post_process_eeg(eeg_signal, config)
                
                all_samples.append(eeg_signal)
            
            if config.num_samples == 1:
                generated_eeg_list.append(all_samples[0])
            else:
                generated_eeg_list.append(all_samples)
        
        if return_single:
            return generated_eeg_list[0]
        else:
            return generated_eeg_list
    
    def _encode_text(self, text: str) -> Optional[torch.Tensor]:
        """Encode text to embedding vector."""
        if not self.tokenizer or not self.text_encoder:
            return None
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get text embeddings
                outputs = self.text_encoder(**inputs)
                
                # Pool embeddings (mean pooling)
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Masked mean pooling
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                pooled_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                return pooled_embedding  # (1, embedding_dim)
                
        except Exception as e:
            warnings.warn(f"Text encoding failed: {e}")
            return None
    
    def _generate_eeg_from_embedding(self,
                                   text_embedding: torch.Tensor,
                                   config: GenerationConfig) -> np.ndarray:
        """Generate EEG from text embedding."""
        with torch.no_grad():
            # Sample noise
            noise_dim = getattr(self.inverse_simulator, 'noise_dim', 100)
            noise = torch.randn(1, noise_dim, device=self.device) * config.noise_level
            
            # Apply temperature to noise
            if config.temperature != 1.0:
                noise = noise * config.temperature
            
            # Generate EEG
            generated_eeg = self.inverse_simulator.generate(text_embedding, noise)
            
            # Convert to numpy
            eeg_numpy = generated_eeg[0].cpu().numpy()  # (channels, samples)
            
            return eeg_numpy
    
    def _post_process_eeg(self,
                         eeg_signal: np.ndarray,
                         config: GenerationConfig) -> np.ndarray:
        """Post-process generated EEG signal."""
        # Apply style-specific processing
        if config.style in self.style_configs:
            eeg_signal = self._apply_style(eeg_signal, config.style, config.sampling_rate)
        
        # Ensure correct duration
        target_samples = int(config.duration * config.sampling_rate)
        if eeg_signal.shape[1] != target_samples:
            eeg_signal = self._resize_eeg(eeg_signal, target_samples)
        
        # Apply EEG template statistics if available
        if self.eeg_template:
            eeg_signal = self._apply_template_statistics(eeg_signal)
        
        # Final noise addition
        if config.noise_level > 0:
            noise = np.random.normal(0, config.noise_level * 0.1, eeg_signal.shape)
            eeg_signal = eeg_signal + noise
        
        return eeg_signal
    
    def _apply_style(self, 
                    eeg_signal: np.ndarray,
                    style: str,
                    sampling_rate: int) -> np.ndarray:
        """Apply style-specific modifications to EEG signal."""
        style_config = self.style_configs[style]
        modified_signal = eeg_signal.copy()
        
        # Frequency emphasis
        freq_emphasis = style_config.get("frequency_emphasis", {})
        if freq_emphasis:
            modified_signal = self._apply_frequency_emphasis(
                modified_signal, freq_emphasis, sampling_rate
            )
        
        # Temporal smoothing
        smoothing_factor = style_config.get("temporal_smoothing", 1.0)
        if smoothing_factor != 1.0:
            modified_signal = self._apply_temporal_smoothing(modified_signal, smoothing_factor)
        
        # Spatial coherence adjustment
        coherence_factor = style_config.get("spatial_coherence", 1.0)
        if coherence_factor != 1.0:
            modified_signal = self._adjust_spatial_coherence(modified_signal, coherence_factor)
        
        return modified_signal
    
    def _apply_frequency_emphasis(self,
                                eeg_signal: np.ndarray,
                                emphasis_config: Dict[str, float],
                                sampling_rate: int) -> np.ndarray:
        """Apply frequency-specific emphasis to EEG signal."""
        try:
            import scipy.signal
            
            # Define frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, min(100, sampling_rate // 2 - 1))
            }
            
            modified_signal = np.zeros_like(eeg_signal)
            
            for ch_idx in range(eeg_signal.shape[0]):
                signal_ch = eeg_signal[ch_idx]
                
                for band_name, (low, high) in bands.items():
                    if band_name in emphasis_config and high < sampling_rate // 2:
                        # Extract band
                        sos = scipy.signal.butter(4, [low, high], 
                                                btype='band', 
                                                fs=sampling_rate, 
                                                output='sos')
                        band_signal = scipy.signal.sosfilt(sos, signal_ch)
                        
                        # Apply emphasis
                        emphasis = emphasis_config[band_name]
                        modified_signal[ch_idx] += band_signal * emphasis
                    else:
                        # Default emphasis of 1.0 for unspecified bands
                        if band_name not in emphasis_config and high < sampling_rate // 2:
                            sos = scipy.signal.butter(4, [low, high], 
                                                    btype='band', 
                                                    fs=sampling_rate, 
                                                    output='sos')
                            band_signal = scipy.signal.sosfilt(sos, signal_ch)
                            modified_signal[ch_idx] += band_signal
            
            return modified_signal
            
        except ImportError:
            warnings.warn("SciPy not available for frequency emphasis")
            return eeg_signal
    
    def _apply_temporal_smoothing(self,
                                eeg_signal: np.ndarray,
                                smoothing_factor: float) -> np.ndarray:
        """Apply temporal smoothing to EEG signal."""
        try:
            import scipy.signal
            
            # Design smoothing filter
            if smoothing_factor > 1.0:
                # More smoothing - lower cutoff frequency
                cutoff = min(40.0 / smoothing_factor, 100.0)
                b, a = scipy.signal.butter(2, cutoff, fs=1000, btype='low')
                
                smoothed_signal = np.zeros_like(eeg_signal)
                for ch_idx in range(eeg_signal.shape[0]):
                    smoothed_signal[ch_idx] = scipy.signal.filtfilt(b, a, eeg_signal[ch_idx])
                
                return smoothed_signal
            elif smoothing_factor < 1.0:
                # Less smoothing - add high frequency content
                high_freq_noise = np.random.normal(0, 0.05, eeg_signal.shape)
                return eeg_signal + high_freq_noise * (1.0 - smoothing_factor)
            else:
                return eeg_signal
                
        except ImportError:
            return eeg_signal
    
    def _adjust_spatial_coherence(self,
                                eeg_signal: np.ndarray,
                                coherence_factor: float) -> np.ndarray:
        """Adjust spatial coherence between EEG channels."""
        if coherence_factor == 1.0:
            return eeg_signal
            
        n_channels, n_samples = eeg_signal.shape
        modified_signal = eeg_signal.copy()
        
        if coherence_factor > 1.0:
            # Increase coherence - mix signals from nearby channels
            mixing_strength = (coherence_factor - 1.0) * 0.2
            
            for ch_idx in range(n_channels):
                # Mix with adjacent channels
                neighbors = []
                if ch_idx > 0:
                    neighbors.append(ch_idx - 1)
                if ch_idx < n_channels - 1:
                    neighbors.append(ch_idx + 1)
                
                if neighbors:
                    neighbor_signal = np.mean([eeg_signal[n] for n in neighbors], axis=0)
                    modified_signal[ch_idx] = (
                        (1 - mixing_strength) * eeg_signal[ch_idx] + 
                        mixing_strength * neighbor_signal
                    )
        
        elif coherence_factor < 1.0:
            # Decrease coherence - add independent noise
            independence_noise = np.random.normal(0, 0.1, eeg_signal.shape)
            noise_strength = (1.0 - coherence_factor) * 0.3
            modified_signal = eeg_signal + independence_noise * noise_strength
        
        return modified_signal
    
    def _resize_eeg(self, eeg_signal: np.ndarray, target_samples: int) -> np.ndarray:
        """Resize EEG signal to target number of samples."""
        current_samples = eeg_signal.shape[1]
        
        if current_samples == target_samples:
            return eeg_signal
        
        # Use interpolation for resizing
        from scipy.interpolate import interp1d
        
        try:
            old_indices = np.linspace(0, 1, current_samples)
            new_indices = np.linspace(0, 1, target_samples)
            
            resized_signal = np.zeros((eeg_signal.shape[0], target_samples))
            
            for ch_idx in range(eeg_signal.shape[0]):
                interpolator = interp1d(old_indices, eeg_signal[ch_idx], 
                                      kind='cubic', bounds_error=False, fill_value='extrapolate')
                resized_signal[ch_idx] = interpolator(new_indices)
            
            return resized_signal
            
        except ImportError:
            # Fallback to simple resampling
            if target_samples > current_samples:
                # Pad with reflection
                pad_samples = target_samples - current_samples
                return np.pad(eeg_signal, ((0, 0), (0, pad_samples)), mode='reflect')
            else:
                # Truncate
                return eeg_signal[:, :target_samples]
    
    def _apply_template_statistics(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Apply EEG template statistics to generated signal."""
        if not self.eeg_template:
            return eeg_signal
        
        modified_signal = eeg_signal.copy()
        
        # Apply mean and std from template
        if 'mean' in self.eeg_template:
            target_mean = self.eeg_template['mean']
            if target_mean.shape[0] == eeg_signal.shape[0]:
                current_mean = np.mean(modified_signal, axis=1, keepdims=True)
                modified_signal = modified_signal - current_mean + target_mean.reshape(-1, 1)
        
        if 'std' in self.eeg_template:
            target_std = self.eeg_template['std']
            if target_std.shape[0] == eeg_signal.shape[0]:
                current_std = np.std(modified_signal, axis=1, keepdims=True)
                scale_factor = target_std.reshape(-1, 1) / (current_std + 1e-8)
                modified_signal = modified_signal * scale_factor
        
        return modified_signal
    
    def validate_synthetic(self,
                          synthetic_eeg: np.ndarray,
                          real_eeg_stats: Optional[str] = None) -> Dict[str, float]:
        """Validate synthetic EEG quality.
        
        Args:
            synthetic_eeg: Generated EEG signal (channels x samples)
            real_eeg_stats: Path to real EEG statistics file
            
        Returns:
            Dictionary of validation metrics
        """
        return self.validator.validate(synthetic_eeg, real_eeg_stats)
    
    def interpolate_texts(self,
                         text1: str,
                         text2: str,
                         alpha: float = 0.5,
                         config: Optional[GenerationConfig] = None) -> np.ndarray:
        """Generate EEG by interpolating between two texts.
        
        Args:
            text1: First text
            text2: Second text
            alpha: Interpolation factor (0 = text1, 1 = text2)
            config: Generation configuration
            
        Returns:
            Interpolated EEG signal
        """
        # Encode both texts
        embedding1 = self._encode_text(text1)
        embedding2 = self._encode_text(text2)
        
        if embedding1 is None or embedding2 is None:
            raise ValueError("Could not encode texts")
        
        # Interpolate embeddings
        interpolated_embedding = (1 - alpha) * embedding1 + alpha * embedding2
        
        # Generate EEG
        if config is None:
            config = GenerationConfig()
        
        eeg_signal = self._generate_eeg_from_embedding(interpolated_embedding, config)
        eeg_signal = self._post_process_eeg(eeg_signal, config)
        
        return eeg_signal
    
    def get_generation_statistics(self) -> Dict[str, any]:
        """Get statistics about the generation model.
        
        Returns:
            Dictionary of model statistics
        """
        stats = {
            'model_type': 'InverseSimulator',
            'has_text_encoder': self.text_encoder is not None,
            'has_eeg_template': self.eeg_template is not None,
            'supported_styles': list(self.style_configs.keys()),
            'device': str(self.device)
        }
        
        if hasattr(self.inverse_simulator, 'generator'):
            gen_params = sum(p.numel() for p in self.inverse_simulator.generator.parameters())
            stats['generator_parameters'] = gen_params
        
        if hasattr(self.inverse_simulator, 'discriminator'):
            disc_params = sum(p.numel() for p in self.inverse_simulator.discriminator.parameters())
            stats['discriminator_parameters'] = disc_params
        
        return stats