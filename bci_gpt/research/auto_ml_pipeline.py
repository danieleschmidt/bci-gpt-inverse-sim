"""Automated machine learning pipeline for BCI-GPT research with neural architecture search."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
import itertools
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

# AutoML libraries
try:
    import autosklearn.classification
    HAS_AUTOSKLEARN = True
except ImportError:
    HAS_AUTOSKLEARN = False
    warnings.warn("Auto-sklearn not available. Install with: pip install auto-sklearn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for automated machine learning pipeline."""
    
    # Neural Architecture Search
    nas_search_space: Dict[str, List[Any]] = field(default_factory=lambda: {
        'hidden_dims': [[64], [128], [256], [128, 64], [256, 128], [512, 256, 128]],
        'activation_functions': ['relu', 'gelu', 'swish', 'leaky_relu'],
        'dropout_rates': [0.1, 0.2, 0.3, 0.4, 0.5],
        'batch_norm': [True, False],
        'skip_connections': [True, False]
    })
    
    # Hyperparameter optimization
    optimization_trials: int = 100
    optimization_timeout: int = 3600  # seconds
    pruning_enabled: bool = True
    
    # Data preprocessing
    preprocessing_options: List[str] = field(default_factory=lambda: [
        'standard_scaling', 'min_max_scaling', 'robust_scaling', 'quantile_transform'
    ])
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'select_k_best', 'mutual_info', 'variance_threshold'
    ])
    feature_selection_k: List[int] = field(default_factory=lambda: [50, 100, 200, 'all'])
    
    # Model ensembling
    enable_ensembling: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'stacking', 'bagging'])
    max_ensemble_models: int = 5
    
    # Evaluation
    cv_folds: int = 5
    test_size: float = 0.2
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'precision', 'recall'])
    
    # Resource constraints
    max_training_time_per_model: int = 300  # seconds
    memory_limit_gb: float = 8.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@dataclass
class NeuralArchitectureSpec:
    """Specification for neural architecture."""
    hidden_dims: List[int]
    activation: str
    dropout_rate: float
    batch_norm: bool
    skip_connections: bool
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'skip_connections': self.skip_connections,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }


class DynamicNeuralNetwork(nn.Module):
    """Dynamically configurable neural network for NAS."""
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 arch_spec: NeuralArchitectureSpec):
        super().__init__()
        
        self.arch_spec = arch_spec
        self.layers = nn.ModuleList()
        
        # Build network layers
        current_size = input_size
        
        for i, hidden_dim in enumerate(arch_spec.hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(current_size, hidden_dim))
            
            # Batch normalization
            if arch_spec.batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            activation_layer = self._get_activation_layer(arch_spec.activation)
            self.layers.append(activation_layer)
            
            # Dropout
            if arch_spec.dropout_rate > 0:
                self.layers.append(nn.Dropout(arch_spec.dropout_rate))
            
            current_size = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(current_size, output_size)
        
        # Skip connection tracking
        self.skip_connections = arch_spec.skip_connections
        self.skip_indices = self._identify_skip_indices() if self.skip_connections else []
    
    def _get_activation_layer(self, activation: str) -> nn.Module:
        """Get activation layer by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _identify_skip_indices(self) -> List[Tuple[int, int]]:
        """Identify potential skip connection points."""
        skip_indices = []
        linear_indices = []
        
        # Find linear layer indices
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                linear_indices.append(i)
        
        # Create skip connections for every other layer
        for i in range(0, len(linear_indices) - 1, 2):
            if i + 1 < len(linear_indices):
                skip_indices.append((linear_indices[i], linear_indices[i + 1]))
        
        return skip_indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional skip connections."""
        skip_outputs = {}
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and self.skip_connections:
                # Store output for potential skip connection
                skip_outputs[i] = x
            
            x = layer(x)
            
            # Apply skip connection if applicable
            if self.skip_connections and i in [end for _, end in self.skip_indices]:
                start_idx = next(start for start, end in self.skip_indices if end == i)
                if start_idx in skip_outputs and skip_outputs[start_idx].shape == x.shape:
                    x = x + skip_outputs[start_idx]
        
        # Output layer
        x = self.output_layer(x)
        return x


class NeuralArchitectureSearch:
    """Neural Architecture Search for BCI-GPT models."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.search_history = []
        self.best_architecture = None
        self.best_score = -np.inf
        
    def search_architectures(self, 
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray) -> Dict[str, Any]:
        """Search for optimal neural architecture."""
        
        logger.info("Starting Neural Architecture Search...")
        
        if HAS_OPTUNA:
            return self._optuna_search(X_train, y_train, X_val, y_val)
        else:
            return self._grid_search(X_train, y_train, X_val, y_val)
    
    def _optuna_search(self, 
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray) -> Dict[str, Any]:
        """Optuna-based architecture search."""
        
        def objective(trial):
            # Sample architecture hyperparameters
            hidden_dims = trial.suggest_categorical(
                'hidden_dims', 
                self.config.nas_search_space['hidden_dims']
            )
            activation = trial.suggest_categorical(
                'activation',
                self.config.nas_search_space['activation_functions']
            )
            dropout_rate = trial.suggest_categorical(
                'dropout_rate',
                self.config.nas_search_space['dropout_rates']
            )
            batch_norm = trial.suggest_categorical(
                'batch_norm',
                self.config.nas_search_space['batch_norm']
            )
            skip_connections = trial.suggest_categorical(
                'skip_connections',
                self.config.nas_search_space['skip_connections']
            )
            
            # Additional hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            
            # Create architecture specification
            arch_spec = NeuralArchitectureSpec(
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                skip_connections=skip_connections,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            
            # Train and evaluate model
            score = self._evaluate_architecture(arch_spec, X_train, y_train, X_val, y_val, trial)
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner() if self.config.pruning_enabled else None
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.optimization_trials,
            timeout=self.config.optimization_timeout
        )
        
        # Get best architecture
        best_params = study.best_params
        self.best_architecture = NeuralArchitectureSpec(
            hidden_dims=best_params['hidden_dims'],
            activation=best_params['activation'],
            dropout_rate=best_params['dropout_rate'],
            batch_norm=best_params['batch_norm'],
            skip_connections=best_params['skip_connections'],
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
        self.best_score = study.best_value
        
        return {
            'best_architecture': self.best_architecture.to_dict(),
            'best_score': self.best_score,
            'study': study,
            'optimization_history': [(t.number, t.value) for t in study.trials if t.value is not None]
        }
    
    def _grid_search(self, 
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray) -> Dict[str, Any]:
        """Grid search fallback for architecture search."""
        
        logger.info("Using grid search for architecture optimization...")
        
        # Create parameter grid
        param_combinations = itertools.product(
            self.config.nas_search_space['hidden_dims'],
            self.config.nas_search_space['activation_functions'],
            self.config.nas_search_space['dropout_rates'],
            self.config.nas_search_space['batch_norm'],
            self.config.nas_search_space['skip_connections']
        )
        
        best_score = -np.inf
        best_arch = None
        all_results = []
        
        for i, (hidden_dims, activation, dropout_rate, batch_norm, skip_connections) in enumerate(param_combinations):
            if i >= self.config.optimization_trials:
                break
            
            arch_spec = NeuralArchitectureSpec(
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                skip_connections=skip_connections
            )
            
            score = self._evaluate_architecture(arch_spec, X_train, y_train, X_val, y_val)
            all_results.append((i, score, arch_spec.to_dict()))
            
            if score > best_score:
                best_score = score
                best_arch = arch_spec
            
            logger.debug(f"Trial {i}: Score = {score:.4f}")
        
        self.best_architecture = best_arch
        self.best_score = best_score
        
        return {
            'best_architecture': best_arch.to_dict() if best_arch else None,
            'best_score': best_score,
            'optimization_history': [(r[0], r[1]) for r in all_results]
        }
    
    def _evaluate_architecture(self, 
                             arch_spec: NeuralArchitectureSpec,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             trial: Optional[Any] = None) -> float:
        """Evaluate a single architecture."""
        
        try:
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Create model
            input_size = X_train.shape[1]
            output_size = len(np.unique(y_train))
            model = DynamicNeuralNetwork(input_size, output_size, arch_spec)
            
            # Setup training
            optimizer = optim.Adam(
                model.parameters(),
                lr=arch_spec.learning_rate,
                weight_decay=arch_spec.weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            model.train()
            best_val_score = -np.inf
            patience_counter = 0
            
            for epoch in range(100):  # Max epochs
                # Training
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_predictions = torch.argmax(val_outputs, dim=1)
                        val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
                    
                    # Early stopping
                    if val_accuracy > best_val_score + self.config.early_stopping_min_delta:
                        best_val_score = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.early_stopping_patience:
                        break
                    
                    # Optuna pruning
                    if trial and HAS_OPTUNA:
                        trial.report(val_accuracy, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    
                    model.train()
            
            return best_val_score
            
        except Exception as e:
            logger.warning(f"Error evaluating architecture: {e}")
            return -1.0


class DataPreprocessingPipeline:
    """Automated data preprocessing pipeline."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.preprocessing_steps = []
        self.feature_selector = None
        
    def optimize_preprocessing(self, 
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize data preprocessing pipeline."""
        
        logger.info("Optimizing data preprocessing pipeline...")
        
        best_score = -np.inf
        best_pipeline = None
        all_results = []
        
        # Test different preprocessing combinations
        for scaler_name in self.config.preprocessing_options:
            for selector_name in self.config.feature_selection_methods:
                for k_features in self.config.feature_selection_k:
                    
                    # Create preprocessing pipeline
                    pipeline_steps = []
                    
                    # Scaling
                    scaler = self._get_scaler(scaler_name)
                    pipeline_steps.append(('scaler', scaler))
                    
                    # Feature selection
                    if k_features != 'all':
                        k_actual = min(k_features, X_train.shape[1])
                        selector = self._get_feature_selector(selector_name, k_actual)
                        pipeline_steps.append(('selector', selector))
                    
                    # Apply preprocessing
                    X_train_processed, X_val_processed = self._apply_preprocessing(
                        X_train, X_val, pipeline_steps
                    )
                    
                    # Quick evaluation with simple model
                    score = self._evaluate_preprocessing(
                        X_train_processed, y_train, X_val_processed, y_val
                    )
                    
                    all_results.append({
                        'scaler': scaler_name,
                        'selector': selector_name,
                        'k_features': k_features,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_pipeline = pipeline_steps.copy()
        
        return {
            'best_pipeline': best_pipeline,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def _get_scaler(self, scaler_name: str):
        """Get scaler by name."""
        scalers = {
            'standard_scaling': StandardScaler(),
            'min_max_scaling': MinMaxScaler(),
            'robust_scaling': RobustScaler(),
            'quantile_transform': StandardScaler()  # Simplified
        }
        return scalers.get(scaler_name, StandardScaler())
    
    def _get_feature_selector(self, selector_name: str, k: int):
        """Get feature selector by name."""
        selectors = {
            'select_k_best': SelectKBest(f_classif, k=k),
            'mutual_info': SelectKBest(mutual_info_classif, k=k),
            'variance_threshold': SelectKBest(f_classif, k=k)  # Simplified
        }
        return selectors.get(selector_name, SelectKBest(f_classif, k=k))
    
    def _apply_preprocessing(self, 
                           X_train: np.ndarray,
                           X_val: np.ndarray,
                           pipeline_steps: List[Tuple[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing pipeline."""
        
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        for step_name, transformer in pipeline_steps:
            X_train_processed = transformer.fit_transform(X_train_processed, y_train if step_name == 'selector' else None)
            X_val_processed = transformer.transform(X_val_processed)
        
        return X_train_processed, X_val_processed
    
    def _evaluate_preprocessing(self, 
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray) -> float:
        """Quick evaluation of preprocessing with simple model."""
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Simple random forest for quick evaluation
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        return accuracy


class AutoMLPipeline:
    """Complete AutoML pipeline for BCI-GPT research."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.nas = NeuralArchitectureSearch(config)
        self.preprocessor = DataPreprocessingPipeline(config)
        self.best_model = None
        self.best_pipeline = None
        self.experiment_history = []
        
    def fit(self, 
            X: np.ndarray,
            y: np.ndarray,
            X_test: Optional[np.ndarray] = None,
            y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit complete AutoML pipeline."""
        
        logger.info("Starting AutoML pipeline...")
        start_time = time.time()
        
        # Split data if test set not provided
        if X_test is None or y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, stratify=y, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        # Further split training data for validation
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Step 1: Optimize preprocessing
        logger.info("Step 1: Optimizing preprocessing pipeline...")
        preprocessing_result = self.preprocessor.optimize_preprocessing(
            X_train_inner, y_train_inner, X_val, y_val
        )
        
        # Apply best preprocessing
        best_preprocessing = preprocessing_result['best_pipeline']
        X_train_processed = X_train_inner.copy()
        X_val_processed = X_val.copy()
        X_test_processed = X_test.copy()
        
        # Store preprocessing transformers for later use
        fitted_transformers = []
        for step_name, transformer in best_preprocessing:
            if step_name == 'selector':
                X_train_processed = transformer.fit_transform(X_train_processed, y_train_inner)
            else:
                X_train_processed = transformer.fit_transform(X_train_processed)
            
            X_val_processed = transformer.transform(X_val_processed)
            X_test_processed = transformer.transform(X_test_processed)
            fitted_transformers.append((step_name, transformer))
        
        self.best_pipeline = fitted_transformers
        
        # Step 2: Neural Architecture Search
        logger.info("Step 2: Neural Architecture Search...")
        nas_result = self.nas.search_architectures(
            X_train_processed, y_train_inner, X_val_processed, y_val
        )
        
        # Step 3: Train best model on full training set
        logger.info("Step 3: Training final model...")
        best_arch = self.nas.best_architecture
        
        # Combine train and validation sets
        X_full_train = np.vstack([X_train_processed, X_val_processed])
        y_full_train = np.hstack([y_train_inner, y_val])
        
        # Train final model
        final_model = self._train_final_model(
            best_arch, X_full_train, y_full_train, X_test_processed, y_test
        )
        
        # Step 4: Final evaluation
        logger.info("Step 4: Final evaluation...")
        final_metrics = self._evaluate_final_model(
            final_model, X_test_processed, y_test
        )
        
        total_time = time.time() - start_time
        
        # Compile results
        automl_results = {
            'best_preprocessing': [step[0] for step in best_preprocessing],
            'preprocessing_score': preprocessing_result['best_score'],
            'best_architecture': nas_result['best_architecture'],
            'nas_score': nas_result['best_score'],
            'final_metrics': final_metrics,
            'total_time_seconds': total_time,
            'model': final_model,
            'preprocessing_pipeline': self.best_pipeline
        }
        
        self.experiment_history.append(automl_results)
        
        logger.info(f"AutoML pipeline completed in {total_time:.1f} seconds")
        logger.info(f"Final test accuracy: {final_metrics.get('accuracy', 0):.4f}")
        
        return automl_results
    
    def _train_final_model(self, 
                          arch_spec: NeuralArchitectureSpec,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> DynamicNeuralNetwork:
        """Train final model with best architecture."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create model
        input_size = X_train.shape[1]
        output_size = len(np.unique(y_train))
        model = DynamicNeuralNetwork(input_size, output_size, arch_spec)
        
        # Setup training
        optimizer = optim.Adam(
            model.parameters(),
            lr=arch_spec.learning_rate,
            weight_decay=arch_spec.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        model.train()
        best_val_score = -np.inf
        patience_counter = 0
        
        for epoch in range(200):  # More epochs for final training
            # Training
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_predictions = torch.argmax(val_outputs, dim=1)
                    val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_accuracy > best_val_score + self.config.early_stopping_min_delta:
                    best_val_score = val_accuracy
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience * 2:  # More patience for final training
                    break
                
                model.train()
        
        # Load best model state
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        return model
    
    def _evaluate_final_model(self, 
                            model: DynamicNeuralNetwork,
                            X_test: np.ndarray,
                            y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate final model on test set."""
        
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        pred_np = predictions.numpy()
        true_np = y_test_tensor.numpy()
        
        metrics = {
            'accuracy': accuracy_score(true_np, pred_np),
            'f1_score': f1_score(true_np, pred_np, average='weighted'),
            'precision': precision_score(true_np, pred_np, average='weighted'),
            'recall': recall_score(true_np, pred_np, average='weighted')
        }
        
        # Additional BCI-specific metrics
        n_classes = len(np.unique(true_np))
        trial_duration = 1.0  # Assume 1 second trials
        
        # Information Transfer Rate
        if metrics['accuracy'] > 1.0/n_classes:
            log2_n = np.log2(n_classes)
            p = metrics['accuracy']
            p_term = p * np.log2(p) if p > 0 else 0
            np_term = (1 - p) * np.log2((1 - p)/(n_classes - 1)) if p < 1 else 0
            
            bits_per_trial = log2_n + p_term + np_term
            itr = (bits_per_trial / trial_duration) * 60  # bits per minute
            metrics['itr'] = max(itr, 0.0)
        else:
            metrics['itr'] = 0.0
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained AutoML pipeline."""
        
        if self.best_model is None or self.best_pipeline is None:
            raise ValueError("AutoML pipeline has not been fitted yet")
        
        # Apply preprocessing pipeline
        X_processed = X.copy()
        for step_name, transformer in self.best_pipeline:
            X_processed = transformer.transform(X_processed)
        
        # Make predictions
        X_tensor = torch.FloatTensor(X_processed)
        self.best_model.eval()
        
        with torch.no_grad():
            outputs = self.best_model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.numpy()
    
    def generate_automl_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive AutoML report."""
        
        if not self.experiment_history:
            return {'error': 'No experiments have been run'}
        
        latest_experiment = self.experiment_history[-1]
        
        report = {
            'pipeline_summary': {
                'preprocessing_steps': latest_experiment['best_preprocessing'],
                'architecture_summary': {
                    'hidden_layers': len(latest_experiment['best_architecture']['hidden_dims']),
                    'total_parameters': sum(latest_experiment['best_architecture']['hidden_dims']),
                    'activation': latest_experiment['best_architecture']['activation'],
                    'dropout_rate': latest_experiment['best_architecture']['dropout_rate']
                },
                'training_time_seconds': latest_experiment['total_time_seconds']
            },
            'performance_metrics': latest_experiment['final_metrics'],
            'optimization_scores': {
                'preprocessing_optimization': latest_experiment['preprocessing_score'],
                'architecture_optimization': latest_experiment['nas_score']
            },
            'model_complexity': {
                'architecture_spec': latest_experiment['best_architecture'],
                'preprocessing_pipeline': latest_experiment['best_preprocessing']
            }
        }
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            report_file = save_path / 'automl_report.json'
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"AutoML report saved to: {report_file}")
        
        return report


# Example usage and validation
if __name__ == "__main__":
    # Create AutoML configuration
    automl_config = AutoMLConfig(
        optimization_trials=20,  # Reduced for faster testing
        cv_folds=3,
        early_stopping_patience=5
    )
    
    # Generate synthetic BCI data for testing
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 200
    n_classes = 5
    
    # Create synthetic EEG-like data
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it learnable
    for i in range(n_classes):
        class_indices = np.arange(i * n_samples // n_classes, (i + 1) * n_samples // n_classes)
        feature_subset = np.arange(i * n_features // n_classes, (i + 1) * n_features // n_classes)
        X[class_indices[:, None], feature_subset] += np.random.normal(2, 0.5, 
                                                                     (len(class_indices), len(feature_subset)))
    
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Add some noise
    noise_samples = n_samples % n_classes
    if noise_samples > 0:
        y = np.concatenate([y, np.random.randint(0, n_classes, noise_samples)])
        X = np.vstack([X, np.random.randn(noise_samples, n_features)])
    
    # Shuffle data
    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    print(f"Generated synthetic BCI data: {X.shape}, {len(np.unique(y))} classes")
    
    # Create and run AutoML pipeline
    automl = AutoMLPipeline(automl_config)
    
    # Fit pipeline
    results = automl.fit(X, y)
    
    print("AutoML Results:")
    print(f"Best preprocessing: {results['best_preprocessing']}")
    print(f"Best architecture: {results['best_architecture']}")
    print(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
    print(f"Final F1-score: {results['final_metrics']['f1_score']:.4f}")
    print(f"Information Transfer Rate: {results['final_metrics']['itr']:.2f} bits/min")
    print(f"Total time: {results['total_time_seconds']:.1f} seconds")
    
    # Generate report
    report = automl.generate_automl_report()
    print(f"\\nAutoML Report generated with {len(report)} sections")
    
    print("AutoML pipeline validation completed!")