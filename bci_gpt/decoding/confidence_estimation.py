"""Confidence estimation for EEG decoding predictions."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from scipy.stats import entropy
import warnings

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available for advanced confidence estimation")


class ConfidenceEstimator:
    """Estimates confidence in EEG decoding predictions using multiple methods."""
    
    def __init__(self, 
                 method: str = "ensemble",
                 calibration_data: Optional[Dict] = None):
        """Initialize confidence estimator.
        
        Args:
            method: Confidence estimation method ("entropy", "max_prob", "ensemble")
            calibration_data: Optional calibration data for better estimates
        """
        self.method = method
        self.calibration_data = calibration_data
        
        # Initialize anomaly detector for outlier detection
        if HAS_SKLEARN:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
            self._is_calibrated = False
        else:
            self.anomaly_detector = None
            self._is_calibrated = False
    
    def estimate_confidence(self,
                          token_probabilities: np.ndarray,
                          eeg_features: Optional[np.ndarray] = None,
                          sequence_history: Optional[List[float]] = None) -> float:
        """Estimate confidence for a prediction.
        
        Args:
            token_probabilities: Probability distribution over tokens
            eeg_features: EEG feature representation
            sequence_history: History of previous confidence scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.method == "entropy":
            return self._entropy_based_confidence(token_probabilities)
        elif self.method == "max_prob":
            return self._max_probability_confidence(token_probabilities)
        elif self.method == "ensemble":
            return self._ensemble_confidence(token_probabilities, eeg_features, sequence_history)
        else:
            raise ValueError(f"Unknown confidence method: {self.method}")
    
    def _entropy_based_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate confidence based on prediction entropy.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Confidence score (lower entropy = higher confidence)
        """
        # Normalize probabilities
        probs = probabilities / np.sum(probabilities)
        
        # Calculate entropy
        ent = entropy(probs)
        max_entropy = np.log(len(probs))
        
        # Convert to confidence (0 = max entropy, 1 = min entropy)
        confidence = 1.0 - (ent / max_entropy)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _max_probability_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate confidence based on maximum probability.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Confidence score (highest probability)
        """
        return float(np.max(probabilities))
    
    def _ensemble_confidence(self,
                           probabilities: np.ndarray,
                           eeg_features: Optional[np.ndarray] = None,
                           sequence_history: Optional[List[float]] = None) -> float:
        """Calculate ensemble confidence using multiple methods.
        
        Args:
            probabilities: Probability distribution
            eeg_features: EEG feature representation
            sequence_history: History of previous confidence scores
            
        Returns:
            Combined confidence score
        """
        confidences = []
        weights = []
        
        # Entropy-based confidence
        entropy_conf = self._entropy_based_confidence(probabilities)
        confidences.append(entropy_conf)
        weights.append(0.3)
        
        # Max probability confidence
        max_prob_conf = self._max_probability_confidence(probabilities)
        confidences.append(max_prob_conf)
        weights.append(0.2)
        
        # Prediction sharpness (how peaked is the distribution)
        sharpness_conf = self._sharpness_confidence(probabilities)
        confidences.append(sharpness_conf)
        weights.append(0.2)
        
        # EEG signal quality confidence
        if eeg_features is not None:
            signal_quality_conf = self._signal_quality_confidence(eeg_features)
            confidences.append(signal_quality_conf)
            weights.append(0.2)
        
        # Temporal consistency confidence
        if sequence_history is not None and len(sequence_history) > 1:
            temporal_conf = self._temporal_consistency_confidence(sequence_history)
            confidences.append(temporal_conf)
            weights.append(0.1)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted combination
        final_confidence = np.sum(np.array(confidences) * weights)
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _sharpness_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate confidence based on distribution sharpness.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Confidence score (sharper distribution = higher confidence)
        """
        # Calculate Gini coefficient as measure of inequality/sharpness
        sorted_probs = np.sort(probabilities)
        n = len(sorted_probs)
        cumsum = np.cumsum(sorted_probs)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # Gini ranges from 0 (uniform) to ~1 (very peaked)
        return np.clip(gini, 0.0, 1.0)
    
    def _signal_quality_confidence(self, eeg_features: np.ndarray) -> float:
        """Calculate confidence based on EEG signal quality.
        
        Args:
            eeg_features: EEG feature representation
            
        Returns:
            Confidence score based on signal quality
        """
        try:
            # Check for anomalies in EEG features
            if self.anomaly_detector is not None and self._is_calibrated:
                # Reshape for sklearn
                features_2d = eeg_features.reshape(1, -1)
                features_scaled = self.scaler.transform(features_2d)
                
                # Get anomaly score (-1 = anomaly, 1 = normal)
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                
                # Convert to confidence (higher score = more normal = higher confidence)
                confidence = (anomaly_score + 1) / 2  # Normalize to [0, 1]
                
            else:
                # Simple heuristic based on feature statistics
                feature_std = np.std(eeg_features)
                feature_mean = np.mean(np.abs(eeg_features))
                
                # Reasonable signal should have moderate variance and mean
                std_score = 1.0 - min(1.0, abs(feature_std - 0.5) / 0.5)
                mean_score = 1.0 - min(1.0, abs(feature_mean - 0.3) / 0.3)
                
                confidence = (std_score + mean_score) / 2
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception:
            # Fallback to neutral confidence
            return 0.5
    
    def _temporal_consistency_confidence(self, sequence_history: List[float]) -> float:
        """Calculate confidence based on temporal consistency.
        
        Args:
            sequence_history: History of previous confidence scores
            
        Returns:
            Confidence score based on temporal stability
        """
        if len(sequence_history) < 2:
            return 0.5
        
        # Calculate variance in recent confidence scores
        recent_scores = sequence_history[-5:]  # Last 5 predictions
        variance = np.var(recent_scores)
        
        # Lower variance = higher consistency = higher confidence
        consistency_score = 1.0 / (1.0 + variance * 10)  # Scale variance
        
        # Also consider trend - declining confidence is concerning
        if len(recent_scores) >= 3:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            trend_penalty = max(0, -trend)  # Penalty for negative trend
            consistency_score *= (1.0 - trend_penalty)
        
        return np.clip(consistency_score, 0.0, 1.0)
    
    def calibrate(self, 
                  eeg_features_list: List[np.ndarray],
                  true_labels: List[int],
                  predicted_probs: List[np.ndarray]) -> None:
        """Calibrate confidence estimator with ground truth data.
        
        Args:
            eeg_features_list: List of EEG feature arrays
            true_labels: List of true token labels
            predicted_probs: List of predicted probability distributions
        """
        if not HAS_SKLEARN or not eeg_features_list:
            warnings.warn("Cannot calibrate without sklearn or data")
            return
        
        try:
            # Prepare EEG features for anomaly detection
            all_features = []
            for features in eeg_features_list:
                all_features.append(features.flatten())
            
            features_array = np.array(all_features)
            
            # Fit scaler and anomaly detector
            features_scaled = self.scaler.fit_transform(features_array)
            self.anomaly_detector.fit(features_scaled)
            
            self._is_calibrated = True
            
            # Calculate calibration metrics
            confidences = []
            accuracies = []
            
            for i, (features, true_label, probs) in enumerate(
                zip(eeg_features_list, true_labels, predicted_probs)
            ):
                conf = self.estimate_confidence(probs, features)
                confidences.append(conf)
                
                pred_label = np.argmax(probs)
                accuracy = 1.0 if pred_label == true_label else 0.0
                accuracies.append(accuracy)
            
            # Store calibration data
            self.calibration_data = {
                'confidences': confidences,
                'accuracies': accuracies,
                'mean_confidence': np.mean(confidences),
                'mean_accuracy': np.mean(accuracies),
                'correlation': np.corrcoef(confidences, accuracies)[0, 1]
            }
            
            print(f"Calibration complete:")
            print(f"  Mean confidence: {self.calibration_data['mean_confidence']:.3f}")
            print(f"  Mean accuracy: {self.calibration_data['mean_accuracy']:.3f}")
            print(f"  Confidence-accuracy correlation: {self.calibration_data['correlation']:.3f}")
            
        except Exception as e:
            warnings.warn(f"Calibration failed: {e}")
    
    def get_calibration_curve(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibration curve for reliability analysis.
        
        Args:
            n_bins: Number of confidence bins
            
        Returns:
            Tuple of (bin_confidences, bin_accuracies)
        """
        if not self.calibration_data:
            warnings.warn("No calibration data available")
            return np.array([]), np.array([])
        
        confidences = np.array(self.calibration_data['confidences'])
        accuracies = np.array(self.calibration_data['accuracies'])
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.any(in_bin):
                bin_confidence = np.mean(confidences[in_bin])
                bin_accuracy = np.mean(accuracies[in_bin])
            else:
                bin_confidence = (bin_lower + bin_upper) / 2
                bin_accuracy = 0.0
            
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
        
        return np.array(bin_confidences), np.array(bin_accuracies)
    
    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE).
        
        Args:
            n_bins: Number of confidence bins
            
        Returns:
            ECE score (lower is better)
        """
        if not self.calibration_data:
            return float('inf')
        
        bin_confidences, bin_accuracies = self.get_calibration_curve(n_bins)
        
        if len(bin_confidences) == 0:
            return float('inf')
        
        # Calculate ECE
        confidences = np.array(self.calibration_data['confidences'])
        ece = 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_weight = bin_size / len(confidences)
                bin_conf = bin_confidences[i]
                bin_acc = bin_accuracies[i]
                
                ece += bin_weight * abs(bin_conf - bin_acc)
        
        return ece
    
    def get_confidence_stats(self) -> Dict[str, float]:
        """Get confidence estimation statistics.
        
        Returns:
            Dictionary of confidence statistics
        """
        if not self.calibration_data:
            return {}
        
        stats = {
            'mean_confidence': self.calibration_data['mean_confidence'],
            'mean_accuracy': self.calibration_data['mean_accuracy'],
            'confidence_accuracy_correlation': self.calibration_data['correlation'],
            'expected_calibration_error': self.expected_calibration_error(),
            'is_calibrated': self._is_calibrated
        }
        
        return stats