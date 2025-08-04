"""Performance metrics and evaluation tools for BCI systems."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available for advanced metrics")


@dataclass
class ClassificationMetrics:
    """Container for classification performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]


@dataclass
class ITRMetrics:
    """Information Transfer Rate metrics."""
    itr_bits_per_min: float
    itr_bits_per_trial: float
    selection_time: float
    accuracy: float
    num_classes: int


@dataclass
class DecodingMetrics:
    """Metrics for continuous decoding performance."""
    word_error_rate: float
    character_error_rate: float
    bleu_score: float
    perplexity: float
    decoding_latency: float


class BCIMetrics:
    """Comprehensive metrics calculator for BCI systems."""
    
    def __init__(self):
        """Initialize BCI metrics calculator."""
        pass
    
    def calculate_itr(self,
                     accuracy: float,
                     num_classes: int,
                     trial_duration: float) -> float:
        """Calculate Information Transfer Rate (ITR).
        
        Args:
            accuracy: Classification accuracy (0-1)
            num_classes: Number of possible classes/symbols
            trial_duration: Time per trial in seconds
            
        Returns:
            ITR in bits per minute
        """
        if accuracy <= 1.0 / num_classes:
            # If accuracy is at chance level or below, ITR is 0
            return 0.0
        
        # ITR formula from Wolpaw et al. (2000)
        if accuracy >= 1.0:
            accuracy = 0.999  # Avoid log(0)
        
        # Bits per selection
        bits_per_selection = (
            np.log2(num_classes) + 
            accuracy * np.log2(accuracy) + 
            (1 - accuracy) * np.log2((1 - accuracy) / (num_classes - 1))
        )
        
        # Convert to bits per minute
        selections_per_minute = 60.0 / trial_duration
        itr_bits_per_min = bits_per_selection * selections_per_minute
        
        return max(0.0, itr_bits_per_min)
    
    def calculate_comprehensive_itr(self,
                                   accuracy: float,
                                   num_classes: int,
                                   trial_duration: float) -> ITRMetrics:
        """Calculate comprehensive ITR metrics.
        
        Args:
            accuracy: Classification accuracy (0-1)
            num_classes: Number of possible classes
            trial_duration: Time per trial in seconds
            
        Returns:
            ITRMetrics object with detailed ITR information
        """
        itr_bits_per_min = self.calculate_itr(accuracy, num_classes, trial_duration)
        itr_bits_per_trial = itr_bits_per_min * (trial_duration / 60.0)
        
        return ITRMetrics(
            itr_bits_per_min=itr_bits_per_min,
            itr_bits_per_trial=itr_bits_per_trial,
            selection_time=trial_duration,
            accuracy=accuracy,
            num_classes=num_classes
        )
    
    def word_error_rate(self,
                       predicted: Union[str, List[str]],
                       reference: Union[str, List[str]]) -> float:
        """Calculate Word Error Rate (WER).
        
        Args:
            predicted: Predicted text(s)
            reference: Reference/ground truth text(s)
            
        Returns:
            WER as a float (0.0 = perfect, higher = worse)
        """
        if isinstance(predicted, str):
            predicted = [predicted]
        if isinstance(reference, str):
            reference = [reference]
        
        if len(predicted) != len(reference):
            raise ValueError("Predicted and reference lists must have same length")
        
        total_errors = 0
        total_words = 0
        
        for pred, ref in zip(predicted, reference):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Calculate edit distance (Levenshtein distance)
            errors = self._edit_distance(pred_words, ref_words)
            
            total_errors += errors
            total_words += len(ref_words)
        
        if total_words == 0:
            return 0.0 if total_errors == 0 else float('inf')
        
        return total_errors / total_words
    
    def character_error_rate(self,
                           predicted: Union[str, List[str]],
                           reference: Union[str, List[str]]) -> float:
        """Calculate Character Error Rate (CER).
        
        Args:
            predicted: Predicted text(s)
            reference: Reference/ground truth text(s)
            
        Returns:
            CER as a float (0.0 = perfect, higher = worse)
        """
        if isinstance(predicted, str):
            predicted = [predicted]
        if isinstance(reference, str):
            reference = [reference]
        
        if len(predicted) != len(reference):
            raise ValueError("Predicted and reference lists must have same length")
        
        total_errors = 0
        total_chars = 0
        
        for pred, ref in zip(predicted, reference):
            pred_chars = list(pred.lower())
            ref_chars = list(ref.lower())
            
            errors = self._edit_distance(pred_chars, ref_chars)
            
            total_errors += errors
            total_chars += len(ref_chars)
        
        if total_chars == 0:
            return 0.0 if total_errors == 0 else float('inf')
        
        return total_errors / total_chars
    
    def _edit_distance(self, seq1: List, seq2: List) -> int:
        """Calculate edit distance (Levenshtein distance) between two sequences."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )
        
        return dp[m][n]
    
    def bleu_score(self,
                  predicted: Union[str, List[str]],
                  reference: Union[str, List[str]],
                  n_gram: int = 4) -> float:
        """Calculate BLEU score for text generation.
        
        Args:
            predicted: Predicted text(s)
            reference: Reference text(s)
            n_gram: Maximum n-gram order
            
        Returns:
            BLEU score (0.0-1.0, higher is better)
        """
        if isinstance(predicted, str):
            predicted = [predicted]
        if isinstance(reference, str):
            reference = [reference]
        
        if len(predicted) != len(reference):
            raise ValueError("Predicted and reference lists must have same length")
        
        bleu_scores = []
        
        for pred, ref in zip(predicted, reference):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if not pred_tokens or not ref_tokens:
                bleu_scores.append(0.0)
                continue
            
            # Calculate n-gram precisions
            precisions = []
            
            for n in range(1, n_gram + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                
                # Count matches
                matches = 0
                for ngram in pred_ngrams:
                    if ngram in ref_ngrams:
                        matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
                
                precision = matches / sum(pred_ngrams.values())
                precisions.append(precision)
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / len(pred_tokens)))
            
            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                geometric_mean = np.exp(np.mean([np.log(p) for p in precisions]))
                bleu = bp * geometric_mean
            else:
                bleu = 0.0
            
            bleu_scores.append(bleu)
        
        return np.mean(bleu_scores)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """Extract n-grams from token sequence."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def calculate_classification_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       class_names: Optional[List[str]] = None) -> ClassificationMetrics:
        """Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names for detailed metrics
            
        Returns:
            ClassificationMetrics object
        """
        if not HAS_SKLEARN:
            # Basic accuracy calculation
            accuracy = np.mean(y_true == y_pred)
            return ClassificationMetrics(
                accuracy=accuracy,
                precision=accuracy,  # Approximation
                recall=accuracy,     # Approximation
                f1_score=accuracy,   # Approximation
                confusion_matrix=np.eye(len(np.unique(y_true))),
                per_class_metrics={}
            )
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        # Get per-class precision, recall, f1-score
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                per_class_metrics[class_name] = {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1_score': f1_per_class[i]
                }
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            per_class_metrics=per_class_metrics
        )
    
    def calculate_decoding_metrics(self,
                                 predicted_texts: List[str],
                                 reference_texts: List[str],
                                 decoding_times: List[float]) -> DecodingMetrics:
        """Calculate comprehensive decoding metrics.
        
        Args:
            predicted_texts: List of predicted texts
            reference_texts: List of reference texts
            decoding_times: List of decoding times in seconds
            
        Returns:
            DecodingMetrics object
        """
        wer = self.word_error_rate(predicted_texts, reference_texts)
        cer = self.character_error_rate(predicted_texts, reference_texts)
        bleu = self.bleu_score(predicted_texts, reference_texts)
        
        # Calculate perplexity (simplified)
        # In practice, would need actual model probabilities
        avg_word_length = np.mean([len(text.split()) for text in predicted_texts])
        perplexity = np.exp(wer) * avg_word_length  # Rough approximation
        
        avg_latency = np.mean(decoding_times)
        
        return DecodingMetrics(
            word_error_rate=wer,
            character_error_rate=cer,
            bleu_score=bleu,
            perplexity=perplexity,
            decoding_latency=avg_latency
        )
    
    def practical_itr(self,
                     accuracy: float,
                     num_classes: int,
                     selection_time: float,
                     inter_trial_interval: float = 0.0) -> float:
        """Calculate practical ITR accounting for inter-trial intervals.
        
        Args:
            accuracy: Classification accuracy
            num_classes: Number of classes
            selection_time: Time for selection in seconds
            inter_trial_interval: Additional time between trials
            
        Returns:
            Practical ITR in bits per minute
        """
        total_time = selection_time + inter_trial_interval
        return self.calculate_itr(accuracy, num_classes, total_time)
    
    def online_performance_metrics(self,
                                  session_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate online performance metrics from session data.
        
        Args:
            session_data: Dictionary with keys like 'accuracies', 'times', 'confidences'
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        if 'accuracies' in session_data:
            accuracies = session_data['accuracies']
            metrics['mean_accuracy'] = np.mean(accuracies)
            metrics['accuracy_std'] = np.std(accuracies)
            metrics['accuracy_trend'] = self._calculate_trend(accuracies)
        
        if 'times' in session_data:
            times = session_data['times']
            metrics['mean_response_time'] = np.mean(times)
            metrics['response_time_std'] = np.std(times)
            metrics['time_trend'] = self._calculate_trend(times)
        
        if 'confidences' in session_data:
            confidences = session_data['confidences']
            metrics['mean_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
            metrics['confidence_trend'] = self._calculate_trend(confidences)
        
        # Calculate workload index (combination of time and accuracy)
        if 'accuracies' in session_data and 'times' in session_data:
            # Higher accuracy and lower time = lower workload
            workload = np.mean(times) / (np.mean(accuracies) + 0.1)
            metrics['workload_index'] = workload
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) in a series of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        return slope
    
    def generate_report(self,
                       classification_metrics: Optional[ClassificationMetrics] = None,
                       itr_metrics: Optional[ITRMetrics] = None,
                       decoding_metrics: Optional[DecodingMetrics] = None) -> str:
        """Generate a comprehensive performance report.
        
        Args:
            classification_metrics: Classification performance metrics
            itr_metrics: Information transfer rate metrics
            decoding_metrics: Decoding performance metrics
            
        Returns:
            Formatted performance report
        """
        report = "=== BCI Performance Report ===\n\n"
        
        if classification_metrics:
            report += "Classification Performance:\n"
            report += f"  Accuracy: {classification_metrics.accuracy:.3f}\n"
            report += f"  Precision: {classification_metrics.precision:.3f}\n"
            report += f"  Recall: {classification_metrics.recall:.3f}\n"
            report += f"  F1-Score: {classification_metrics.f1_score:.3f}\n\n"
            
            if classification_metrics.per_class_metrics:
                report += "Per-Class Metrics:\n"
                for class_name, metrics in classification_metrics.per_class_metrics.items():
                    report += f"  {class_name}:\n"
                    report += f"    Precision: {metrics['precision']:.3f}\n"
                    report += f"    Recall: {metrics['recall']:.3f}\n"
                    report += f"    F1-Score: {metrics['f1_score']:.3f}\n"
                report += "\n"
        
        if itr_metrics:
            report += "Information Transfer Rate:\n"
            report += f"  ITR: {itr_metrics.itr_bits_per_min:.2f} bits/min\n"
            report += f"  Bits per trial: {itr_metrics.itr_bits_per_trial:.3f}\n"
            report += f"  Selection time: {itr_metrics.selection_time:.2f} seconds\n"
            report += f"  Accuracy: {itr_metrics.accuracy:.3f}\n"
            report += f"  Number of classes: {itr_metrics.num_classes}\n\n"
        
        if decoding_metrics:
            report += "Decoding Performance:\n"
            report += f"  Word Error Rate: {decoding_metrics.word_error_rate:.3f}\n"
            report += f"  Character Error Rate: {decoding_metrics.character_error_rate:.3f}\n"
            report += f"  BLEU Score: {decoding_metrics.bleu_score:.3f}\n"
            report += f"  Perplexity: {decoding_metrics.perplexity:.2f}\n"
            report += f"  Avg. Latency: {decoding_metrics.decoding_latency:.3f} seconds\n\n"
        
        # Performance interpretation
        report += "Performance Assessment:\n"
        
        if classification_metrics:
            if classification_metrics.accuracy >= 0.9:
                report += "  Classification: Excellent (≥90%)\n"
            elif classification_metrics.accuracy >= 0.8:
                report += "  Classification: Good (80-89%)\n"
            elif classification_metrics.accuracy >= 0.7:
                report += "  Classification: Fair (70-79%)\n"
            else:
                report += "  Classification: Needs improvement (<70%)\n"
        
        if itr_metrics:
            if itr_metrics.itr_bits_per_min >= 40:
                report += "  ITR: Excellent (≥40 bits/min)\n"
            elif itr_metrics.itr_bits_per_min >= 25:
                report += "  ITR: Good (25-39 bits/min)\n"
            elif itr_metrics.itr_bits_per_min >= 15:
                report += "  ITR: Fair (15-24 bits/min)\n"
            else:
                report += "  ITR: Needs improvement (<15 bits/min)\n"
        
        if decoding_metrics:
            if decoding_metrics.word_error_rate <= 0.1:
                report += "  Decoding: Excellent (≤10% WER)\n"
            elif decoding_metrics.word_error_rate <= 0.2:
                report += "  Decoding: Good (10-20% WER)\n"
            elif decoding_metrics.word_error_rate <= 0.3:
                report += "  Decoding: Fair (20-30% WER)\n"
            else:
                report += "  Decoding: Needs improvement (>30% WER)\n"
        
        return report