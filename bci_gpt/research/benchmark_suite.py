"""Comprehensive benchmark suite for BCI-GPT research validation and comparison."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    benchmark_name: str
    dataset_paths: List[str]
    models_to_test: List[str]
    evaluation_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'precision', 'recall'])
    cross_validation_folds: int = 5
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = 'auto'
    save_predictions: bool = True
    save_intermediate_results: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: ['paired_t_test', 'wilcoxon'])
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class BenchmarkResult:
    """Result from benchmark execution."""
    benchmark_id: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    per_fold_metrics: Dict[str, List[float]]
    statistical_significance: Dict[str, Any]
    runtime_seconds: float
    memory_usage_mb: Optional[float]
    predictions: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_id': self.benchmark_id,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'metrics': self.metrics,
            'per_fold_metrics': self.per_fold_metrics,
            'statistical_significance': self.statistical_significance,
            'runtime_seconds': self.runtime_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'hyperparameters': self.hyperparameters
        }


class BaseDataset(ABC):
    """Abstract base class for benchmark datasets."""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset and return features, labels."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        pass
    
    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data for model input."""
        pass


class SyntheticEEGDataset(BaseDataset):
    """Synthetic EEG dataset for reproducible benchmarking."""
    
    def __init__(self, 
                 n_samples: int = 1000,
                 n_channels: int = 9,
                 sequence_length: int = 1000,
                 n_classes: int = 5,
                 sampling_rate: int = 1000,
                 noise_level: float = 0.1,
                 random_seed: int = 42):
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic EEG data with realistic patterns."""
        
        # Generate base signals with different frequency components
        time = np.linspace(0, self.sequence_length / self.sampling_rate, self.sequence_length)
        
        features = []
        labels = []
        
        for i in range(self.n_samples):
            # Random class assignment
            label = np.random.randint(0, self.n_classes)
            
            # Class-specific frequency patterns
            class_freqs = {
                0: [8, 12],   # Alpha band
                1: [13, 30],  # Beta band
                2: [30, 50],  # Gamma band
                3: [4, 7],    # Theta band
                4: [0.5, 4]   # Delta band
            }
            
            freq_range = class_freqs.get(label, [8, 12])
            
            # Generate multi-channel EEG
            eeg_channels = []
            for ch in range(self.n_channels):
                # Base signal with class-specific frequency
                freq = np.random.uniform(freq_range[0], freq_range[1])
                phase = np.random.uniform(0, 2*np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                
                signal = amplitude * np.sin(2 * np.pi * freq * time + phase)
                
                # Add noise and artifacts
                noise = np.random.normal(0, self.noise_level, len(time))
                artifacts = np.random.exponential(0.1, len(time)) * np.random.choice([-1, 1], len(time))
                
                channel_signal = signal + noise + 0.1 * artifacts
                eeg_channels.append(channel_signal)
            
            eeg_data = np.array(eeg_channels)  # Shape: (n_channels, sequence_length)
            
            features.append(eeg_data)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'name': 'SyntheticEEG',
            'n_samples': self.n_samples,
            'n_channels': self.n_channels,
            'sequence_length': self.sequence_length,
            'n_classes': self.n_classes,
            'sampling_rate': self.sampling_rate,
            'noise_level': self.noise_level,
            'random_seed': self.random_seed
        }
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess EEG data."""
        # Z-score normalization per channel
        normalized_data = []
        for sample in data:
            normalized_sample = []
            for channel in sample:
                normalized_channel = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
                normalized_sample.append(normalized_channel)
            normalized_data.append(normalized_sample)
        
        return np.array(normalized_data)


class BaseModel(ABC):
    """Abstract base class for benchmark models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name."""
        pass


class SimpleTransformerModel(BaseModel):
    """Simple transformer-based model for benchmarking."""
    
    def __init__(self, 
                 n_channels: int = 9,
                 sequence_length: int = 1000,
                 n_classes: int = 5,
                 hidden_dim: int = 128,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 device: str = 'cpu'):
        
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device
        
        self._build_model()
    
    def _build_model(self):
        """Build transformer model."""
        
        class TransformerEEGClassifier(nn.Module):
            def __init__(self, n_channels, sequence_length, n_classes, hidden_dim, n_layers, n_heads):
                super().__init__()
                
                # Channel embedding
                self.channel_embedding = nn.Linear(sequence_length, hidden_dim)
                
                # Positional encoding for channels
                self.pos_encoding = nn.Parameter(torch.randn(n_channels, hidden_dim))
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * n_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, n_classes)
                )
            
            def forward(self, x):
                # x shape: (batch_size, n_channels, sequence_length)
                batch_size = x.size(0)
                
                # Embed each channel
                x = self.channel_embedding(x)  # (batch_size, n_channels, hidden_dim)
                
                # Add positional encoding
                x = x + self.pos_encoding.unsqueeze(0)
                
                # Apply transformer
                x = self.transformer(x)  # (batch_size, n_channels, hidden_dim)
                
                # Flatten and classify
                x = x.view(batch_size, -1)
                x = self.classifier(x)
                
                return x
        
        self.model = TransformerEEGClassifier(
            self.n_channels, self.sequence_length, self.n_classes,
            self.hidden_dim, self.n_layers, self.n_heads
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for epoch in range(50):  # Fixed epochs for benchmark
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_name(self) -> str:
        """Get model name."""
        return f"SimpleTransformer_h{self.hidden_dim}_l{self.n_layers}"


class RandomForestBaseline(BaseModel):
    """Random Forest baseline for comparison."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        self.n_estimators = n_estimators
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        # Flatten EEG data for traditional ML
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X_flat)
    
    def get_name(self) -> str:
        """Get model name."""
        return f"RandomForest_{self.n_estimators}"


class BenchmarkSuite:
    """Comprehensive benchmark suite for BCI-GPT evaluation."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.benchmark_id = self._generate_benchmark_id()
        
    def _generate_benchmark_id(self) -> str:
        """Generate unique benchmark ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"benchmark_{timestamp}_{config_hash}"
    
    def register_dataset(self, dataset: BaseDataset) -> str:
        """Register a dataset for benchmarking."""
        dataset_info = dataset.get_info()
        dataset_name = dataset_info['name']
        logger.info(f"Registered dataset: {dataset_name}")
        return dataset_name
    
    def register_model(self, model: BaseModel) -> str:
        """Register a model for benchmarking."""
        model_name = model.get_name()
        logger.info(f"Registered model: {model_name}")
        return model_name
    
    def run_single_benchmark(self, 
                           model: BaseModel, 
                           dataset: BaseDataset,
                           random_seed: int = 42) -> BenchmarkResult:
        """Run benchmark for single model-dataset pair."""
        
        logger.info(f"Running benchmark: {model.get_name()} on {dataset.get_info()['name']}")
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Load and preprocess data
        start_time = time.time()
        X, y = dataset.load_data()
        X = dataset.preprocess(X)
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=random_seed)
        
        fold_metrics = {metric: [] for metric in self.config.evaluation_metrics}
        all_predictions = []
        all_ground_truth = []
        all_confidences = []
        
        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            logger.debug(f"Processing fold {fold + 1}/{self.config.cross_validation_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Calculate metrics
            fold_result = self._calculate_metrics(y_test, predictions, probabilities)
            
            for metric in self.config.evaluation_metrics:
                if metric in fold_result:
                    fold_metrics[metric].append(fold_result[metric])
            
            # Store predictions for overall analysis
            all_predictions.extend(predictions)
            all_ground_truth.extend(y_test)
            all_confidences.extend(np.max(probabilities, axis=1))
        
        # Calculate overall metrics
        overall_metrics = {}
        for metric in self.config.evaluation_metrics:
            if fold_metrics[metric]:
                overall_metrics[metric] = np.mean(fold_metrics[metric])
                overall_metrics[f"{metric}_std"] = np.std(fold_metrics[metric])
        
        # Statistical significance tests
        statistical_tests = self._run_statistical_tests(fold_metrics)
        
        # Runtime calculation
        runtime = time.time() - start_time
        
        # Memory usage (approximate)
        memory_usage = self._estimate_memory_usage(model, X)
        
        # Create result object
        result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            model_name=model.get_name(),
            dataset_name=dataset.get_info()['name'],
            metrics=overall_metrics,
            per_fold_metrics=fold_metrics,
            statistical_significance=statistical_tests,
            runtime_seconds=runtime,
            memory_usage_mb=memory_usage,
            predictions=np.array(all_predictions) if self.config.save_predictions else None,
            ground_truth=np.array(all_ground_truth) if self.config.save_predictions else None,
            confidence_scores=np.array(all_confidences) if self.config.save_predictions else None
        )
        
        self.results.append(result)
        logger.info(f"Completed benchmark: {model.get_name()} - Accuracy: {overall_metrics.get('accuracy', 0):.3f}")
        
        return result
    
    def run_comprehensive_benchmark(self, 
                                  models: List[BaseModel], 
                                  datasets: List[BaseDataset]) -> List[BenchmarkResult]:
        """Run comprehensive benchmark across all models and datasets."""
        
        logger.info(f"Starting comprehensive benchmark with {len(models)} models and {len(datasets)} datasets")
        
        all_results = []
        
        for dataset in datasets:
            for model in models:
                for seed in self.config.random_seeds:
                    try:
                        result = self.run_single_benchmark(model, dataset, seed)
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in benchmark {model.get_name()}-{dataset.get_info()['name']}: {e}")
        
        logger.info(f"Completed comprehensive benchmark with {len(all_results)} results")
        return all_results
    
    def _calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Information Transfer Rate (for BCI)
        n_classes = len(np.unique(y_true))
        trial_duration = 1.0  # Assume 1 second trials
        metrics['itr'] = self._calculate_itr(metrics['accuracy'], n_classes, trial_duration)
        
        # Confidence calibration
        confidences = np.max(y_proba, axis=1)
        correct_predictions = (y_pred == y_true).astype(float)
        calibration_error = self._calculate_calibration_error(confidences, correct_predictions)
        metrics['calibration_error'] = calibration_error
        
        return metrics
    
    def _calculate_itr(self, accuracy: float, n_classes: int, trial_duration: float) -> float:
        """Calculate Information Transfer Rate (ITR) in bits per minute."""
        if accuracy <= 1.0/n_classes:
            return 0.0
        
        log2_n = np.log2(n_classes)
        p_term = accuracy * np.log2(accuracy) if accuracy > 0 else 0
        np_term = (1 - accuracy) * np.log2((1 - accuracy)/(n_classes - 1)) if accuracy < 1 else 0
        
        bits_per_trial = log2_n + p_term + np_term
        bits_per_minute = (bits_per_trial / trial_duration) * 60
        
        return max(bits_per_minute, 0.0)
    
    def _calculate_calibration_error(self, 
                                   confidences: np.ndarray, 
                                   accuracies: np.ndarray, 
                                   n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
        
        return ece
    
    def _run_statistical_tests(self, fold_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Run statistical significance tests."""
        
        tests = {}
        
        for metric_name, values in fold_metrics.items():
            if len(values) >= 2:
                # One-sample t-test against chance level
                chance_level = 0.2 if 'accuracy' in metric_name else 0.0
                t_stat, p_value = stats.ttest_1samp(values, chance_level)
                
                tests[f"{metric_name}_vs_chance"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': (np.mean(values) - chance_level) / np.std(values)
                }
        
        return tests
    
    def _estimate_memory_usage(self, model: BaseModel, X: np.ndarray) -> Optional[float]:
        """Estimate memory usage in MB."""
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                # PyTorch model
                total_params = sum(p.numel() for p in model.model.parameters())
                memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
                return memory_mb
            else:
                # Approximate for other models
                return len(str(model).encode()) / (1024 * 1024)
        except:
            return None
    
    def generate_comparison_report(self, output_path: Path) -> Path:
        """Generate comprehensive comparison report."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create results DataFrame
        results_data = []
        for result in self.results:
            row = {
                'model': result.model_name,
                'dataset': result.dataset_name,
                'runtime': result.runtime_seconds,
                'memory_mb': result.memory_usage_mb
            }
            row.update(result.metrics)
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # Generate report
        report = f"""# BCI-GPT Benchmark Report
        
**Benchmark ID:** {self.benchmark_id}
**Generated:** {datetime.now().isoformat()}

## Summary Statistics

"""
        
        # Overall statistics
        for metric in ['accuracy', 'f1', 'itr']:
            if metric in df.columns:
                report += f"### {metric.upper()}\n"
                report += f"- Mean: {df[metric].mean():.3f} ± {df[metric].std():.3f}\n"
                report += f"- Best: {df[metric].max():.3f} ({df.loc[df[metric].idxmax(), 'model']})\n"
                report += f"- Range: [{df[metric].min():.3f}, {df[metric].max():.3f}]\n\n"
        
        # Model comparison
        report += "## Model Comparison\n\n"
        model_summary = df.groupby('model').agg({
            'accuracy': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'runtime': 'mean'
        }).round(3)
        
        report += model_summary.to_markdown() + "\n\n"
        
        # Statistical significance
        report += "## Statistical Analysis\n\n"
        for result in self.results:
            if result.statistical_significance:
                report += f"### {result.model_name}\n"
                for test_name, test_result in result.statistical_significance.items():
                    significance = "✅ Significant" if test_result['significant'] else "❌ Not significant"
                    report += f"- {test_name}: p = {test_result['p_value']:.3f} ({significance})\n"
                report += "\n"
        
        # Save report
        report_path = output_path / f"{self.benchmark_id}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed results
        results_path = output_path / f"{self.benchmark_id}_results.csv"
        df.to_csv(results_path, index=False)
        
        # Save JSON results
        json_path = output_path / f"{self.benchmark_id}_results.json"
        with open(json_path, 'w') as f:
            json.dump([result.to_dict() for result in self.results], f, indent=2, default=str)
        
        logger.info(f"Generated benchmark report: {report_path}")
        return report_path
    
    def create_visualization(self, output_path: Path) -> Dict[str, Path]:
        """Create benchmark visualization plots."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        plots = {}
        
        # Performance comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract data for plotting
        models = []
        accuracies = []
        for result in self.results:
            models.append(result.model_name)
            accuracies.append(result.metrics.get('accuracy', 0))
        
        # Box plot of accuracies by model
        df_plot = pd.DataFrame({'Model': models, 'Accuracy': accuracies})
        sns.boxplot(data=df_plot, x='Model', y='Accuracy', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        performance_plot = output_path / f"{self.benchmark_id}_performance.png"
        plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['performance'] = performance_plot
        
        # Runtime comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        runtimes = [result.runtime_seconds for result in self.results]
        df_runtime = pd.DataFrame({'Model': models, 'Runtime': runtimes})
        sns.barplot(data=df_runtime, x='Model', y='Runtime', ax=ax)
        ax.set_title('Runtime Comparison')
        ax.set_ylabel('Runtime (seconds)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        runtime_plot = output_path / f"{self.benchmark_id}_runtime.png"
        plt.savefig(runtime_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['runtime'] = runtime_plot
        
        logger.info(f"Generated visualization plots: {list(plots.keys())}")
        return plots


# Example usage and validation
if __name__ == "__main__":
    # Create benchmark configuration
    config = BenchmarkConfig(
        benchmark_name="BCI_GPT_Validation",
        dataset_paths=[],  # Will use synthetic data
        models_to_test=["SimpleTransformer", "RandomForest"],
        cross_validation_folds=3,  # Reduced for faster testing
        random_seeds=[42, 123, 456]
    )
    
    # Create benchmark suite
    suite = BenchmarkSuite(config)
    
    # Create synthetic dataset
    dataset = SyntheticEEGDataset(
        n_samples=500,  # Reduced for faster testing
        n_channels=9,
        sequence_length=1000,
        n_classes=5
    )
    
    # Create models
    transformer_model = SimpleTransformerModel(
        n_channels=9,
        sequence_length=1000,
        n_classes=5,
        device=config.device
    )
    
    rf_model = RandomForestBaseline(n_estimators=50)  # Reduced for faster testing
    
    # Run comprehensive benchmark
    results = suite.run_comprehensive_benchmark(
        models=[transformer_model, rf_model],
        datasets=[dataset]
    )
    
    # Generate report
    output_path = Path("./benchmark_results")
    report_path = suite.generate_comparison_report(output_path)
    plots = suite.create_visualization(output_path)
    
    print(f"Benchmark completed!")
    print(f"Results: {len(results)} benchmark runs")
    print(f"Report: {report_path}")
    print(f"Plots: {list(plots.keys())}")
    
    # Print summary
    for result in results:
        print(f"{result.model_name}: Accuracy = {result.metrics.get('accuracy', 0):.3f}, "
              f"F1 = {result.metrics.get('f1', 0):.3f}, "
              f"Runtime = {result.runtime_seconds:.1f}s")