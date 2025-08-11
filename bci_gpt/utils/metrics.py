"""Metrics and evaluation utilities for BCI-GPT models."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    from scipy import stats
    from scipy.signal import welch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available for advanced metrics")

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class GANMetrics:
    """Metrics for evaluating GAN-based EEG generation quality."""
    
    def __init__(self):
        self.history = {
            'fid_scores': [],
            'is_scores': [],
            'spectral_distances': [],
            'realism_scores': []
        }
    
    def basic_similarity(self, 
                        real_features: torch.Tensor,
                        fake_features: torch.Tensor) -> float:
        """Calculate basic feature similarity when scipy not available."""
        real_mean = torch.mean(real_features, dim=0)
        fake_mean = torch.mean(fake_features, dim=0)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(real_mean.unsqueeze(0), fake_mean.unsqueeze(0))
        return float(cos_sim.item())
    
    def spectral_distance_simple(self,
                               real_eeg: torch.Tensor,
                               fake_eeg: torch.Tensor) -> Dict[str, float]:
        """Simple spectral comparison using PyTorch FFT."""
        # Use PyTorch FFT for basic spectral analysis
        real_fft = torch.fft.rfft(real_eeg, dim=2)
        fake_fft = torch.fft.rfft(fake_eeg, dim=2)
        
        real_power = torch.abs(real_fft) ** 2
        fake_power = torch.abs(fake_fft) ** 2
        
        # Average power across batch and channels
        real_power_mean = torch.mean(real_power, dim=(0, 1))
        fake_power_mean = torch.mean(fake_power, dim=(0, 1))
        
        # Simple L2 distance
        spectral_distance = F.mse_loss(real_power_mean, fake_power_mean)
        
        return {
            'spectral_mse': float(spectral_distance.item()),
            'average_spectral_distance': float(spectral_distance.item())
        }
    
    def evaluate_generation_quality(self,
                                  real_eeg: torch.Tensor,
                                  fake_eeg: torch.Tensor,
                                  discriminator: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """Basic evaluation of generation quality."""
        metrics = {}
        
        # Simple spectral analysis
        spectral_metrics = self.spectral_distance_simple(real_eeg, fake_eeg)
        metrics.update(spectral_metrics)
        
        # Basic statistical comparison
        real_mean = torch.mean(real_eeg)
        fake_mean = torch.mean(fake_eeg)
        real_std = torch.std(real_eeg)
        fake_std = torch.std(fake_eeg)
        
        metrics['mean_difference'] = float(torch.abs(real_mean - fake_mean).item())
        metrics['std_difference'] = float(torch.abs(real_std - fake_std).item())
        
        # Discriminator-based realism score
        if discriminator is not None:
            with torch.no_grad():
                fake_outputs = discriminator(fake_eeg)
                if isinstance(fake_outputs, dict):
                    realism_score = torch.mean(fake_outputs.get('combined_output', 
                                             torch.sigmoid(fake_outputs.get('main_output', torch.tensor(0.5)))))
                else:
                    realism_score = torch.mean(torch.sigmoid(fake_outputs))
                
                metrics['realism_score'] = float(realism_score.item())
        
        return metrics


class BCIMetrics:
    """Metrics for BCI decoding performance evaluation."""
    
    def __init__(self):
        self.history = {
            'accuracy': [],
            'word_error_rate': [],
            'information_transfer_rate': []
        }
    
    def calculate_accuracy(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor) -> float:
        """Calculate classification accuracy."""
        if predictions.shape != targets.shape:
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=-1)
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=-1)
        
        correct = (predictions == targets).float()
        accuracy = torch.mean(correct).item()
        
        self.history['accuracy'].append(accuracy)
        return accuracy
    
    def word_error_rate(self,
                       predicted_text: str,
                       reference_text: str) -> float:
        """Calculate Word Error Rate (WER)."""
        pred_words = predicted_text.lower().split()
        ref_words = reference_text.lower().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        # Simple edit distance
        if len(pred_words) == 0:
            return 1.0
        
        # Count matching words (simplified)
        matches = sum(1 for p, r in zip(pred_words, ref_words) if p == r)
        wer = 1.0 - (matches / len(ref_words))
        
        self.history['word_error_rate'].append(wer)
        return wer
    
    def calculate_itr(self,
                     accuracy: float,
                     num_classes: int,
                     trial_duration: float) -> float:
        """Calculate Information Transfer Rate (ITR) in bits per minute."""
        if accuracy <= 1.0 / num_classes or num_classes <= 1:
            return 0.0
        
        # Simplified ITR calculation
        bits_per_trial = np.log2(num_classes) * accuracy
        itr_per_minute = (bits_per_trial / trial_duration) * 60
        
        self.history['information_transfer_rate'].append(itr_per_minute)
        return max(0.0, itr_per_minute)


class PerformanceTracker:
    """Track system performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'throughput': [],
            'inference_time': []
        }
    
    def record_throughput(self, samples_per_second: float) -> None:
        self.metrics['throughput'].append(samples_per_second)
    
    def record_inference_time(self, time_ms: float) -> None:
        self.metrics['inference_time'].append(time_ms)
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get basic summary statistics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                summary[metric_name] = {'mean': 0.0, 'min': 0.0, 'max': 0.0}
        
        return summary


# Alias for backwards compatibility
EEGMetrics = BCIMetrics