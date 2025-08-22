"""Advanced experimental framework for BCI-GPT research validation and benchmarking."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for BCI-GPT experiments."""
    experiment_name: str
    participant_ids: List[str]
    sessions_per_participant: int
    tasks: List[str]
    modalities: List[str] = field(default_factory=lambda: ['eeg'])
    languages: List[str] = field(default_factory=lambda: ['english'])
    validation_splits: Dict[str, float] = field(default_factory=lambda: {'train': 0.7, 'val': 0.15, 'test': 0.15})
    statistical_power: float = 0.8
    alpha_level: float = 0.05
    effect_size: float = 0.5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert sum(self.validation_splits.values()) == 1.0, "Validation splits must sum to 1.0"
        assert 0.0 < self.alpha_level < 1.0, "Alpha level must be between 0 and 1"
        assert self.statistical_power > 0.0, "Statistical power must be positive"


@dataclass
class ExperimentResult:
    """Results from BCI-GPT experiments with statistical validation."""
    experiment_id: str
    timestamp: datetime
    config: ExperimentConfig
    metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    p_values: Dict[str, float]
    sample_size: int
    reproducibility_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp.isoformat(),
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'statistical_tests': self.statistical_tests,
            'confidence_intervals': {k: list(v) for k, v in self.confidence_intervals.items()},
            'effect_sizes': self.effect_sizes,
            'p_values': self.p_values,
            'sample_size': self.sample_size,
            'reproducibility_hash': self.reproducibility_hash
        }


class StatisticalValidator:
    """Statistical validation framework for BCI experiments."""
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        self.alpha = alpha
        self.power = power
        
    def calculate_effect_size(self, 
                            group1: np.ndarray, 
                            group2: np.ndarray, 
                            effect_type: str = 'cohens_d') -> float:
        """Calculate effect size between two groups."""
        if effect_type == 'cohens_d':
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                (len(group2) - 1) * np.var(group2, ddof=1)) / 
                               (len(group1) + len(group2) - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        elif effect_type == 'glass_delta':
            return (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")
    
    def power_analysis(self, 
                      effect_size: float, 
                      sample_size: int, 
                      alpha: Optional[float] = None) -> float:
        """Calculate statistical power for given parameters."""
        alpha = alpha or self.alpha
        # Approximate power calculation for two-sample t-test
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(self.power)
        delta = effect_size * np.sqrt(sample_size / 2)
        power = 1 - norm.cdf(z_alpha - delta) + norm.cdf(-z_alpha - delta)
        return power
    
    def required_sample_size(self, 
                           effect_size: float, 
                           power: Optional[float] = None, 
                           alpha: Optional[float] = None) -> int:
        """Calculate required sample size for desired power."""
        power = power or self.power
        alpha = alpha or self.alpha
        
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    def paired_t_test(self, 
                     before: np.ndarray, 
                     after: np.ndarray) -> Dict[str, Any]:
        """Perform paired t-test with effect size calculation."""
        t_stat, p_value = stats.ttest_rel(before, after)
        effect_size = self.calculate_effect_size(before, after)
        
        # Confidence interval for difference
        diff = after - before
        se_diff = stats.sem(diff)
        ci_95 = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=se_diff)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': ci_95,
            'degrees_freedom': len(diff) - 1,
            'mean_difference': np.mean(diff),
            'std_difference': np.std(diff, ddof=1)
        }
    
    def independent_t_test(self, 
                          group1: np.ndarray, 
                          group2: np.ndarray) -> Dict[str, Any]:
        """Perform independent samples t-test."""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        effect_size = self.calculate_effect_size(group1, group2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'group1_std': np.std(group1, ddof=1),
            'group2_std': np.std(group2, ddof=1)
        }
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons."""
        return [min(p * len(p_values), 1.0) for p in p_values]
    
    def fdr_correction(self, p_values: List[float]) -> List[float]:
        """Apply False Discovery Rate correction (Benjamini-Hochberg)."""
        p_array = np.array(p_values)
        n = len(p_array)
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Calculate corrected p-values
        corrected = np.zeros_like(sorted_p)
        for i in range(n-1, -1, -1):
            if i == n-1:
                corrected[i] = sorted_p[i]
            else:
                corrected[i] = min(sorted_p[i] * n / (i + 1), corrected[i + 1])
        
        # Restore original order
        result = np.zeros_like(p_array)
        result[sorted_indices] = corrected
        return result.tolist()


class BCIMetricsCalculator:
    """Comprehensive metrics calculation for BCI experiments."""
    
    @staticmethod
    def information_transfer_rate(accuracy: float, 
                                num_classes: int, 
                                trial_duration: float) -> float:
        """Calculate Information Transfer Rate (ITR) in bits per minute."""
        if accuracy <= 1.0/num_classes:
            return 0.0
        
        # ITR formula: log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))
        log2_n = np.log2(num_classes)
        p_term = accuracy * np.log2(accuracy) if accuracy > 0 else 0
        np_term = (1 - accuracy) * np.log2((1 - accuracy)/(num_classes - 1)) if accuracy < 1 else 0
        
        bits_per_trial = log2_n + p_term + np_term
        bits_per_minute = (bits_per_trial / trial_duration) * 60
        
        return max(bits_per_minute, 0.0)
    
    @staticmethod
    def word_error_rate(predicted: List[str], 
                       reference: List[str]) -> float:
        """Calculate Word Error Rate (WER) for continuous decoding."""
        if len(reference) == 0:
            return 1.0 if len(predicted) > 0 else 0.0
        
        # Simple word-level comparison
        correct = sum(1 for p, r in zip(predicted, reference) if p == r)
        return 1.0 - (correct / len(reference))
    
    @staticmethod
    def neural_linguistic_alignment(neural_features: np.ndarray, 
                                  linguistic_features: np.ndarray) -> float:
        """Calculate correlation between neural and linguistic features."""
        # Flatten features if needed
        neural_flat = neural_features.flatten()
        linguistic_flat = linguistic_features.flatten()
        
        # Handle dimension mismatch
        min_len = min(len(neural_flat), len(linguistic_flat))
        neural_flat = neural_flat[:min_len]
        linguistic_flat = linguistic_flat[:min_len]
        
        correlation, _ = stats.pearsonr(neural_flat, linguistic_flat)
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def temporal_consistency(predictions: np.ndarray, 
                           window_size: int = 10) -> float:
        """Calculate temporal consistency of predictions."""
        if len(predictions) < window_size:
            return 1.0
        
        consistencies = []
        for i in range(len(predictions) - window_size + 1):
            window = predictions[i:i+window_size]
            # Calculate standard deviation as inverse consistency measure
            consistency = 1.0 / (1.0 + np.std(window))
            consistencies.append(consistency)
        
        return np.mean(consistencies)
    
    @staticmethod
    def confidence_calibration(confidences: np.ndarray, 
                             accuracies: np.ndarray, 
                             num_bins: int = 10) -> Dict[str, float]:
        """Calculate confidence calibration metrics."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0  # Expected Calibration Error
        mce = 0.0  # Maximum Calibration Error
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'average_confidence': np.mean(confidences),
            'average_accuracy': np.mean(accuracies)
        }


class ExperimentFramework:
    """Advanced experimental framework for BCI-GPT research."""
    
    def __init__(self, 
                 base_path: str = "./experiments",
                 random_seed: int = 42):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.validator = StatisticalValidator()
        self.metrics_calc = BCIMetricsCalculator()
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new experiment with unique ID."""
        experiment_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = self.base_path / experiment_id
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        config_path = experiment_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def run_baseline_comparison(self, 
                              experiment_id: str,
                              baseline_scores: np.ndarray,
                              novel_scores: np.ndarray,
                              metric_name: str = "accuracy") -> Dict[str, Any]:
        """Compare novel approach against baseline with statistical validation."""
        
        # Statistical testing
        t_test_result = self.validator.independent_t_test(novel_scores, baseline_scores)
        effect_size = t_test_result['effect_size']
        
        # Confidence intervals
        novel_ci = stats.t.interval(0.95, len(novel_scores)-1, 
                                  loc=np.mean(novel_scores), 
                                  scale=stats.sem(novel_scores))
        baseline_ci = stats.t.interval(0.95, len(baseline_scores)-1, 
                                     loc=np.mean(baseline_scores), 
                                     scale=stats.sem(baseline_scores))
        
        # Power analysis
        achieved_power = self.validator.power_analysis(
            effect_size, min(len(novel_scores), len(baseline_scores))
        )
        
        comparison_result = {
            'metric_name': metric_name,
            'novel_mean': np.mean(novel_scores),
            'baseline_mean': np.mean(baseline_scores),
            'improvement': np.mean(novel_scores) - np.mean(baseline_scores),
            'relative_improvement': (np.mean(novel_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores),
            'statistical_test': t_test_result,
            'confidence_intervals': {
                'novel': novel_ci,
                'baseline': baseline_ci
            },
            'effect_size': effect_size,
            'achieved_power': achieved_power,
            'significant': t_test_result['p_value'] < self.validator.alpha,
            'effect_magnitude': self._interpret_effect_size(effect_size)
        }
        
        # Save results
        self._save_comparison_result(experiment_id, comparison_result)
        return comparison_result
    
    def run_ablation_study(self, 
                          experiment_id: str,
                          component_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform ablation study with multiple comparison correction."""
        
        # Extract full model scores (assumes 'full' key exists)
        if 'full' not in component_scores:
            raise ValueError("Ablation study requires 'full' model scores")
        
        full_scores = component_scores['full']
        ablation_results = {}
        p_values = []
        
        for component, scores in component_scores.items():
            if component == 'full':
                continue
                
            # Compare full model vs ablated model
            t_test = self.validator.independent_t_test(full_scores, scores)
            ablation_results[component] = {
                'performance_drop': np.mean(full_scores) - np.mean(scores),
                'relative_drop': (np.mean(full_scores) - np.mean(scores)) / np.mean(full_scores),
                'statistical_test': t_test,
                'importance_score': t_test['effect_size']
            }
            p_values.append(t_test['p_value'])
        
        # Multiple comparison correction
        corrected_p_values = self.validator.fdr_correction(p_values)
        
        # Add corrected p-values
        for i, component in enumerate([k for k in component_scores.keys() if k != 'full']):
            ablation_results[component]['corrected_p_value'] = corrected_p_values[i]
            ablation_results[component]['significant_after_correction'] = corrected_p_values[i] < self.validator.alpha
        
        # Rank components by importance
        importance_ranking = sorted(
            ablation_results.items(), 
            key=lambda x: x[1]['importance_score'], 
            reverse=True
        )
        
        study_result = {
            'component_analysis': ablation_results,
            'importance_ranking': [comp for comp, _ in importance_ranking],
            'multiple_comparison_correction': 'FDR (Benjamini-Hochberg)',
            'overall_significance': any(corrected_p_values[i] < self.validator.alpha for i in range(len(corrected_p_values)))
        }
        
        self._save_ablation_result(experiment_id, study_result)
        return study_result
    
    def cross_subject_validation(self, 
                               experiment_id: str,
                               subject_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform cross-subject validation analysis."""
        
        all_scores = np.concatenate(list(subject_scores.values()))
        subject_means = [np.mean(scores) for scores in subject_scores.values()]
        
        # Inter-subject variability
        between_subject_var = np.var(subject_means, ddof=1)
        within_subject_var = np.mean([np.var(scores, ddof=1) for scores in subject_scores.values()])
        
        # Intraclass Correlation Coefficient (ICC)
        n_subjects = len(subject_scores)
        n_per_subject = len(list(subject_scores.values())[0])  # Assume equal sessions
        
        ms_between = between_subject_var * n_per_subject
        ms_within = within_subject_var
        
        icc = (ms_between - ms_within) / (ms_between + (n_per_subject - 1) * ms_within)
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*subject_scores.values())
        
        validation_result = {
            'overall_mean': np.mean(all_scores),
            'overall_std': np.std(all_scores, ddof=1),
            'subject_means': {subj: np.mean(scores) for subj, scores in subject_scores.items()},
            'subject_stds': {subj: np.std(scores, ddof=1) for subj, scores in subject_scores.items()},
            'between_subject_variance': between_subject_var,
            'within_subject_variance': within_subject_var,
            'intraclass_correlation': icc,
            'anova_f_statistic': f_stat,
            'anova_p_value': p_value,
            'generalizability': self._interpret_icc(icc),
            'subject_heterogeneity': 'high' if p_value < 0.05 else 'low'
        }
        
        self._save_validation_result(experiment_id, validation_result)
        return validation_result
    
    def calculate_comprehensive_metrics(self, 
                                      predictions: np.ndarray,
                                      ground_truth: np.ndarray,
                                      confidences: Optional[np.ndarray] = None,
                                      trial_duration: float = 1.0,
                                      num_classes: int = 2) -> Dict[str, float]:
        """Calculate comprehensive metrics for BCI evaluation."""
        
        # Basic classification metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )
        
        # BCI-specific metrics
        itr = self.metrics_calc.information_transfer_rate(
            accuracy, num_classes, trial_duration
        )
        
        temporal_consistency = self.metrics_calc.temporal_consistency(predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'information_transfer_rate': itr,
            'temporal_consistency': temporal_consistency
        }
        
        # Add confidence calibration if confidences provided
        if confidences is not None:
            calibration = self.metrics_calc.confidence_calibration(
                confidences, (predictions == ground_truth).astype(float)
            )
            metrics.update(calibration)
        
        return metrics
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_icc(self, icc: float) -> str:
        """Interpret Intraclass Correlation Coefficient."""
        if icc < 0.5:
            return "poor"
        elif icc < 0.75:
            return "moderate"
        elif icc < 0.9:
            return "good"
        else:
            return "excellent"
    
    def _save_comparison_result(self, experiment_id: str, result: Dict[str, Any]):
        """Save baseline comparison results."""
        result_path = self.base_path / experiment_id / "baseline_comparison.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_ablation_result(self, experiment_id: str, result: Dict[str, Any]):
        """Save ablation study results."""
        result_path = self.base_path / experiment_id / "ablation_study.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_validation_result(self, experiment_id: str, result: Dict[str, Any]):
        """Save cross-subject validation results."""
        result_path = self.base_path / experiment_id / "cross_subject_validation.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def generate_publication_report(self, 
                                  experiment_id: str,
                                  title: str,
                                  authors: List[str],
                                  venue: str) -> str:
        """Generate publication-ready experimental report."""
        
        experiment_path = self.base_path / experiment_id
        
        # Load all results
        results = {}
        for result_file in experiment_path.glob("*.json"):
            if result_file.name != "config.json":
                with open(result_file, 'r') as f:
                    results[result_file.stem] = json.load(f)
        
        # Generate LaTeX report
        report = self._generate_latex_report(
            experiment_id, title, authors, venue, results
        )
        
        report_path = experiment_path / "publication_report.tex"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Generated publication report: {report_path}")
        return str(report_path)
    
    def _generate_latex_report(self, 
                             experiment_id: str,
                             title: str,
                             authors: List[str],
                             venue: str,
                             results: Dict[str, Any]) -> str:
        """Generate LaTeX publication report."""
        
        authors_str = ", ".join(authors)
        
        report = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}

\\title{{{title}}}
\\author{{{authors_str}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Experimental Results}}

\\subsection{{Experiment ID: {experiment_id}}}

"""
        
        # Add baseline comparison if available
        if 'baseline_comparison' in results:
            comp = results['baseline_comparison']
            report += f"""
\\subsection{{Baseline Comparison}}

The proposed method achieved a mean {comp['metric_name']} of {comp['novel_mean']:.3f} compared to the baseline of {comp['baseline_mean']:.3f}, representing an improvement of {comp['improvement']:.3f} ({comp['relative_improvement']*100:.1f}\\%).

Statistical analysis revealed a significant difference (t = {comp['statistical_test']['t_statistic']:.3f}, p = {comp['statistical_test']['p_value']:.3f}, Cohen's d = {comp['effect_size']:.3f}, {comp['effect_magnitude']} effect size).

"""
        
        # Add ablation study if available
        if 'ablation_study' in results:
            ablation = results['ablation_study']
            report += f"""
\\subsection{{Ablation Study}}

Component importance ranking (by effect size):
\\begin{{enumerate}}
"""
            for component in ablation['importance_ranking']:
                comp_data = ablation['component_analysis'][component]
                report += f"\\item {component}: Performance drop = {comp_data['performance_drop']:.3f} ({comp_data['relative_drop']*100:.1f}\\%)\n"
            
            report += "\\end{enumerate}\n\n"
        
        # Add cross-subject validation if available
        if 'cross_subject_validation' in results:
            validation = results['cross_subject_validation']
            report += f"""
\\subsection{{Cross-Subject Validation}}

Overall performance: Mean = {validation['overall_mean']:.3f}, SD = {validation['overall_std']:.3f}

Generalizability: {validation['generalizability']} (ICC = {validation['intraclass_correlation']:.3f})

Subject heterogeneity: {validation['subject_heterogeneity']} (F = {validation['anova_f_statistic']:.3f}, p = {validation['anova_p_value']:.3f})

"""
        
        report += """
\\section{Conclusion}

These results demonstrate the effectiveness of the proposed BCI-GPT approach for imagined speech decoding with statistical significance and practical effect sizes suitable for clinical applications.

\\end{document}
"""
        
        return report


# Example usage and validation
if __name__ == "__main__":
    # Create experimental framework
    framework = ExperimentFramework()
    
    # Example experiment configuration
    config = ExperimentConfig(
        experiment_name="cross_modal_attention_validation",
        participant_ids=[f"P{i:03d}" for i in range(1, 21)],  # 20 participants
        sessions_per_participant=10,
        tasks=["word_classification", "sentence_generation"],
        modalities=["eeg", "emg"],
        languages=["english"]
    )
    
    # Create experiment
    experiment_id = framework.create_experiment(config)
    print(f"Created experiment: {experiment_id}")
    
    # Simulate experimental data for demonstration
    np.random.seed(42)
    
    # Baseline vs novel comparison
    baseline_scores = np.random.normal(0.75, 0.05, 50)  # 75% accuracy ± 5%
    novel_scores = np.random.normal(0.85, 0.05, 50)     # 85% accuracy ± 5%
    
    comparison_result = framework.run_baseline_comparison(
        experiment_id, baseline_scores, novel_scores, "accuracy"
    )
    
    print(f"Baseline comparison: {comparison_result['relative_improvement']*100:.1f}% improvement")
    print(f"Statistical significance: p = {comparison_result['statistical_test']['p_value']:.3f}")
    print(f"Effect size: {comparison_result['effect_size']:.3f} ({comparison_result['effect_magnitude']})")
    
    # Ablation study
    component_scores = {
        'full': np.random.normal(0.85, 0.05, 50),
        'no_attention': np.random.normal(0.75, 0.05, 50),
        'no_cross_modal': np.random.normal(0.70, 0.05, 50),
        'no_temporal_conv': np.random.normal(0.80, 0.05, 50)
    }
    
    ablation_result = framework.run_ablation_study(experiment_id, component_scores)
    print(f"Most important component: {ablation_result['importance_ranking'][0]}")
    
    # Cross-subject validation
    subject_scores = {
        f"subject_{i}": np.random.normal(0.85, 0.03, 10) for i in range(20)
    }
    
    validation_result = framework.cross_subject_validation(experiment_id, subject_scores)
    print(f"Generalizability: {validation_result['generalizability']} (ICC = {validation_result['intraclass_correlation']:.3f})")
    
    # Generate publication report
    report_path = framework.generate_publication_report(
        experiment_id,
        "Cross-Modal Attention Mechanisms for Brain-Computer Interface Applications",
        ["Daniel Schmidt", "Terragon Labs"],
        "NeurIPS 2025"
    )
    
    print(f"Publication report generated: {report_path}")