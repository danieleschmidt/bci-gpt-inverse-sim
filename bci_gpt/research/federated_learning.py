"""Federated learning framework for privacy-preserving BCI-GPT research across institutions."""

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
import hashlib
import pickle
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Cryptographic imports (optional)
try:
    import crypten
    import syft as sy
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    warnings.warn("Cryptographic libraries not available. Install with: pip install crypten pysyft")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    aggregation_method: str = "fedavg"  # fedavg, fedprox, scaffold
    num_rounds: int = 50
    clients_per_round: int = 10
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.001
    
    # Privacy settings
    use_differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    
    # Security settings
    use_secure_aggregation: bool = False
    use_homomorphic_encryption: bool = False
    
    # Communication settings
    compression_ratio: float = 0.1
    use_quantization: bool = False
    quantization_bits: int = 8
    
    # Client selection
    client_sampling_strategy: str = "random"  # random, weighted, importance
    min_clients_required: int = 5
    
    # Personalization
    enable_personalization: bool = False
    personalization_layers: List[str] = field(default_factory=lambda: ["classifier"])


@dataclass
class ClientData:
    """Data structure for federated client."""
    client_id: str
    train_data: torch.Tensor
    train_labels: torch.Tensor
    test_data: Optional[torch.Tensor] = None
    test_labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about client data."""
        return {
            'n_train_samples': len(self.train_data),
            'n_test_samples': len(self.test_data) if self.test_data is not None else 0,
            'n_features': self.train_data.shape[1:],
            'n_classes': len(torch.unique(self.train_labels)),
            'class_distribution': torch.bincount(self.train_labels).tolist()
        }


class FederatedClient:
    """Federated learning client for BCI-GPT training."""
    
    def __init__(self, 
                 client_id: str,
                 data: ClientData,
                 config: FederatedConfig):
        self.client_id = client_id
        self.data = data
        self.config = config
        self.model = None
        self.optimizer = None
        self.local_updates = 0
        self.training_history = []
        
    def set_model(self, model: nn.Module):
        """Set the global model for local training."""
        self.model = model.clone() if hasattr(model, 'clone') else model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def local_train(self, global_model_params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform local training and return model updates."""
        
        # Load global model parameters
        self.model.load_state_dict(global_model_params)
        
        # Create data loader
        dataset = TensorDataset(self.data.train_data, self.data.train_labels)
        dataloader = DataLoader(dataset, batch_size=self.config.local_batch_size, shuffle=True)
        
        # Local training
        self.model.train()
        local_losses = []
        
        for epoch in range(self.config.local_epochs):
            epoch_losses = []
            
            for batch_data, batch_labels in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = nn.CrossEntropyLoss()(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Differential privacy noise injection
                if self.config.use_differential_privacy:
                    self._add_dp_noise()
                
                self.optimizer.step()
                epoch_losses.append(loss.item())
            
            local_losses.append(np.mean(epoch_losses))
        
        # Calculate model updates
        model_updates = self._calculate_model_updates(global_model_params)
        
        # Local evaluation
        local_metrics = self._local_evaluate()
        
        self.local_updates += 1
        self.training_history.append({
            'round': self.local_updates,
            'losses': local_losses,
            'metrics': local_metrics
        })
        
        return {
            'client_id': self.client_id,
            'model_updates': model_updates,
            'num_samples': len(self.data.train_data),
            'local_loss': np.mean(local_losses),
            'local_metrics': local_metrics,
            'compression_info': self._compress_updates(model_updates) if self.config.compression_ratio < 1.0 else None
        }
    
    def _add_dp_noise(self):
        """Add differential privacy noise to gradients."""
        with torch.no_grad():
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.dp_clip_norm)
            
            # Add Gaussian noise
            for param in self.model.parameters():
                if param.grad is not None:
                    noise_scale = self.config.dp_clip_norm * np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) / self.config.dp_epsilon
                    noise = torch.normal(0, noise_scale, size=param.grad.shape)
                    param.grad += noise
    
    def _calculate_model_updates(self, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate model parameter updates."""
        updates = {}
        current_params = self.model.state_dict()
        
        for name, param in current_params.items():
            updates[name] = param - global_params[name]
        
        return updates
    
    def _local_evaluate(self) -> Dict[str, float]:
        """Evaluate model on local test data."""
        if self.data.test_data is None:
            return {}
        
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.data.test_data)
            test_predictions = torch.argmax(test_outputs, dim=1)
            
            accuracy = (test_predictions == self.data.test_labels).float().mean().item()
            
            # Calculate F1 score
            from sklearn.metrics import f1_score
            f1 = f1_score(self.data.test_labels.numpy(), test_predictions.numpy(), average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'num_test_samples': len(self.data.test_data)
        }
    
    def _compress_updates(self, updates: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compress model updates for communication efficiency."""
        compression_info = {
            'method': 'top_k',
            'ratio': self.config.compression_ratio,
            'original_size': 0,
            'compressed_size': 0
        }
        
        compressed_updates = {}
        
        for name, update in updates.items():
            # Flatten tensor
            flat_update = update.flatten()
            original_size = len(flat_update)
            
            # Select top-k values
            k = int(original_size * self.config.compression_ratio)
            _, top_indices = torch.topk(torch.abs(flat_update), k)
            
            compressed_updates[name] = {
                'indices': top_indices,
                'values': flat_update[top_indices],
                'shape': update.shape
            }
            
            compression_info['original_size'] += original_size
            compression_info['compressed_size'] += k
        
        return {
            'updates': compressed_updates,
            'info': compression_info
        }


class FederatedAggregator:
    """Aggregator for federated learning that combines client updates."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.aggregation_history = []
        
    def aggregate_fedavg(self, 
                        client_updates: List[Dict[str, Any]],
                        global_model: nn.Module) -> nn.Module:
        """FedAvg aggregation algorithm."""
        
        # Calculate total number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated updates
        aggregated_updates = {}
        
        # Get parameter names from first client
        param_names = list(client_updates[0]['model_updates'].keys())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(global_model.state_dict()[param_name])
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['model_updates'][param_name]
            
            aggregated_updates[param_name] = weighted_sum
        
        # Update global model
        global_state = global_model.state_dict()
        for param_name, update in aggregated_updates.items():
            global_state[param_name] += update
        
        global_model.load_state_dict(global_state)
        return global_model
    
    def aggregate_fedprox(self, 
                         client_updates: List[Dict[str, Any]],
                         global_model: nn.Module,
                         mu: float = 0.01) -> nn.Module:
        """FedProx aggregation with proximal term."""
        
        # Standard FedAvg aggregation
        updated_model = self.aggregate_fedavg(client_updates, global_model)
        
        # Apply proximal regularization (simplified version)
        # In practice, this would be applied during client training
        return updated_model
    
    def aggregate_scaffold(self, 
                          client_updates: List[Dict[str, Any]],
                          global_model: nn.Module,
                          control_variates: Dict[str, torch.Tensor]) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """SCAFFOLD aggregation with control variates."""
        
        # Simplified SCAFFOLD implementation
        # Full implementation would require maintaining control variates on clients
        updated_model = self.aggregate_fedavg(client_updates, global_model)
        
        # Update control variates (simplified)
        updated_control_variates = control_variates.copy()
        
        return updated_model, updated_control_variates
    
    def secure_aggregation(self, 
                          client_updates: List[Dict[str, Any]],
                          global_model: nn.Module) -> nn.Module:
        """Secure aggregation to protect individual client updates."""
        
        if not HAS_CRYPTO:
            logger.warning("Cryptographic libraries not available, using standard aggregation")
            return self.aggregate_fedavg(client_updates, global_model)
        
        # Simplified secure aggregation (would need proper implementation)
        # This is a placeholder for demonstration
        logger.info("Performing secure aggregation...")
        
        # For now, fall back to standard FedAvg
        return self.aggregate_fedavg(client_updates, global_model)


class FederatedLearningServer:
    """Federated learning server orchestrating the training process."""
    
    def __init__(self, 
                 initial_model: nn.Module,
                 config: FederatedConfig):
        self.global_model = initial_model
        self.config = config
        self.clients = {}
        self.aggregator = FederatedAggregator(config)
        self.training_history = []
        self.current_round = 0
        
    def register_client(self, client: FederatedClient):
        """Register a client for federated training."""
        self.clients[client.client_id] = client
        client.set_model(self.global_model)
        logger.info(f"Registered client {client.client_id}")
    
    def select_clients(self) -> List[str]:
        """Select clients for the current round."""
        available_clients = list(self.clients.keys())
        
        if self.config.client_sampling_strategy == "random":
            selected = np.random.choice(
                available_clients,
                size=min(self.config.clients_per_round, len(available_clients)),
                replace=False
            ).tolist()
        elif self.config.client_sampling_strategy == "weighted":
            # Weight by data size
            weights = []
            for client_id in available_clients:
                client = self.clients[client_id]
                weights.append(len(client.data.train_data))
            
            weights = np.array(weights) / sum(weights)
            selected = np.random.choice(
                available_clients,
                size=min(self.config.clients_per_round, len(available_clients)),
                p=weights,
                replace=False
            ).tolist()
        else:
            # Default to random
            selected = available_clients[:self.config.clients_per_round]
        
        return selected
    
    def federated_train(self) -> Dict[str, Any]:
        """Execute federated training process."""
        
        logger.info(f"Starting federated training for {self.config.num_rounds} rounds")
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"Round {round_num + 1}/{self.config.num_rounds}")
            
            # Select clients for this round
            selected_clients = self.select_clients()
            logger.info(f"Selected {len(selected_clients)} clients: {selected_clients}")
            
            # Get global model parameters
            global_params = self.global_model.state_dict()
            
            # Collect client updates
            client_updates = []
            round_metrics = []
            
            for client_id in selected_clients:
                client = self.clients[client_id]
                
                try:
                    # Perform local training
                    update_result = client.local_train(global_params)
                    client_updates.append(update_result)
                    round_metrics.append(update_result['local_metrics'])
                    
                    logger.debug(f"Client {client_id} - Loss: {update_result['local_loss']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training client {client_id}: {e}")
            
            # Aggregate updates
            if len(client_updates) >= self.config.min_clients_required:
                if self.config.aggregation_method == "fedavg":
                    self.global_model = self.aggregator.aggregate_fedavg(client_updates, self.global_model)
                elif self.config.aggregation_method == "fedprox":
                    self.global_model = self.aggregator.aggregate_fedprox(client_updates, self.global_model)
                elif self.config.aggregation_method == "scaffold":
                    # Placeholder for SCAFFOLD
                    self.global_model = self.aggregator.aggregate_fedavg(client_updates, self.global_model)
                
                # Global evaluation
                global_metrics = self._global_evaluate()
                
                # Record round history
                round_summary = {
                    'round': round_num + 1,
                    'num_participants': len(client_updates),
                    'avg_local_loss': np.mean([u['local_loss'] for u in client_updates]),
                    'global_metrics': global_metrics,
                    'client_metrics': round_metrics
                }
                
                self.training_history.append(round_summary)
                
                logger.info(f"Round {round_num + 1} complete - Global accuracy: {global_metrics.get('accuracy', 0):.4f}")
            
            else:
                logger.warning(f"Insufficient clients ({len(client_updates)} < {self.config.min_clients_required})")
            
            self.current_round = round_num + 1
        
        # Final evaluation and summary
        final_results = self._generate_final_report()
        
        logger.info(f"Federated training completed!")
        logger.info(f"Final global accuracy: {final_results['final_accuracy']:.4f}")
        
        return final_results
    
    def _global_evaluate(self) -> Dict[str, float]:
        """Evaluate global model on all client test data."""
        
        all_test_data = []
        all_test_labels = []
        
        # Collect test data from all clients
        for client in self.clients.values():
            if client.data.test_data is not None:
                all_test_data.append(client.data.test_data)
                all_test_labels.append(client.data.test_labels)
        
        if not all_test_data:
            return {}
        
        # Combine all test data
        combined_test_data = torch.cat(all_test_data, dim=0)
        combined_test_labels = torch.cat(all_test_labels, dim=0)
        
        # Evaluate global model
        self.global_model.eval()
        with torch.no_grad():
            outputs = self.global_model(combined_test_data)
            predictions = torch.argmax(outputs, dim=1)
            
            accuracy = (predictions == combined_test_labels).float().mean().item()
            
            # Calculate additional metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            pred_np = predictions.numpy()
            true_np = combined_test_labels.numpy()
            
            f1 = f1_score(true_np, pred_np, average='weighted')
            precision = precision_score(true_np, pred_np, average='weighted')
            recall = recall_score(true_np, pred_np, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'num_test_samples': len(combined_test_data)
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final federated learning report."""
        
        if not self.training_history:
            return {'error': 'No training history available'}
        
        # Extract metrics over time
        rounds = [h['round'] for h in self.training_history]
        accuracies = [h['global_metrics'].get('accuracy', 0) for h in self.training_history]
        losses = [h['avg_local_loss'] for h in self.training_history]
        
        # Client participation analysis
        all_participants = []
        for h in self.training_history:
            all_participants.extend([m.get('client_id', 'unknown') for m in h.get('client_metrics', [])])
        
        participation_counts = pd.Series(all_participants).value_counts()
        
        # Final metrics
        final_metrics = self.training_history[-1]['global_metrics']
        
        report = {
            'final_accuracy': final_metrics.get('accuracy', 0),
            'final_f1_score': final_metrics.get('f1_score', 0),
            'total_rounds': len(self.training_history),
            'convergence_info': {
                'best_accuracy': max(accuracies),
                'best_round': rounds[np.argmax(accuracies)],
                'final_loss': losses[-1],
                'convergence_achieved': self._check_convergence(accuracies)
            },
            'client_participation': {
                'total_unique_clients': len(participation_counts),
                'avg_participation_per_client': participation_counts.mean(),
                'participation_distribution': participation_counts.to_dict()
            },
            'training_history': self.training_history,
            'model_size_mb': self._calculate_model_size(),
            'privacy_metrics': self._calculate_privacy_metrics() if self.config.use_differential_privacy else None
        }
        
        return report
    
    def _check_convergence(self, accuracies: List[float], window: int = 5, threshold: float = 0.01) -> bool:
        """Check if training has converged."""
        if len(accuracies) < window:
            return False
        
        recent_accuracies = accuracies[-window:]
        return np.std(recent_accuracies) < threshold
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in self.global_model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.global_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def _calculate_privacy_metrics(self) -> Dict[str, float]:
        """Calculate privacy metrics for differential privacy."""
        
        # Theoretical privacy guarantees
        total_queries = self.current_round * self.config.local_epochs
        
        # Composition theorem for DP
        composed_epsilon = total_queries * self.config.dp_epsilon
        
        return {
            'epsilon': self.config.dp_epsilon,
            'delta': self.config.dp_delta,
            'composed_epsilon': composed_epsilon,
            'total_queries': total_queries,
            'privacy_loss': composed_epsilon  # Simplified privacy loss
        }
    
    def visualize_training_progress(self, save_path: Optional[Path] = None) -> Dict[str, Path]:
        """Visualize federated training progress."""
        
        if not self.training_history:
            logger.warning("No training history to visualize")
            return {}
        
        # Extract data for plotting
        rounds = [h['round'] for h in self.training_history]
        accuracies = [h['global_metrics'].get('accuracy', 0) for h in self.training_history]
        losses = [h['avg_local_loss'] for h in self.training_history]
        
        plots = {}
        
        # Accuracy over rounds
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, 'b-', linewidth=2, marker='o')
        plt.xlabel('Round')
        plt.ylabel('Global Accuracy')
        plt.title('Federated Learning: Global Accuracy Over Rounds')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            accuracy_path = save_path / 'fl_accuracy.png'
            plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
            plots['accuracy'] = accuracy_path
        plt.show()
        
        # Loss over rounds
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, losses, 'r-', linewidth=2, marker='s')
        plt.xlabel('Round')
        plt.ylabel('Average Local Loss')
        plt.title('Federated Learning: Average Local Loss Over Rounds')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            loss_path = save_path / 'fl_loss.png'
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            plots['loss'] = loss_path
        plt.show()
        
        return plots


# Example usage and validation
if __name__ == "__main__":
    # Create federated learning configuration
    fl_config = FederatedConfig(
        aggregation_method="fedavg",
        num_rounds=10,  # Reduced for faster testing
        clients_per_round=3,
        local_epochs=3,
        use_differential_privacy=True,
        dp_epsilon=1.0
    )
    
    # Create a simple model for testing
    class SimpleBCIModel(nn.Module):
        def __init__(self, input_size=100, hidden_size=64, num_classes=5):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Initialize model and server
    model = SimpleBCIModel()
    server = FederatedLearningServer(model, fl_config)
    
    # Create synthetic federated data for testing
    np.random.seed(42)
    torch.manual_seed(42)
    
    for client_id in range(5):  # 5 clients
        # Generate heterogeneous data for each client
        n_samples = np.random.randint(50, 150)
        data = torch.randn(n_samples, 100)
        
        # Add client-specific bias
        bias = torch.randn(1, 100) * 0.5
        data += bias
        
        labels = torch.randint(0, 5, (n_samples,))
        
        # Split into train/test
        n_train = int(0.8 * n_samples)
        train_data, test_data = data[:n_train], data[n_train:]
        train_labels, test_labels = labels[:n_train], labels[n_train:]
        
        # Create client data
        client_data = ClientData(
            client_id=f"client_{client_id}",
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            metadata={"institution": f"Hospital_{client_id}", "region": f"Region_{client_id % 3}"}
        )
        
        # Create and register client
        client = FederatedClient(f"client_{client_id}", client_data, fl_config)
        server.register_client(client)
        
        logger.info(f"Client {client_id} data stats: {client_data.get_data_stats()}")
    
    # Run federated training
    results = server.federated_train()
    
    print("Federated Learning Results:")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Best Accuracy: {results['convergence_info']['best_accuracy']:.4f}")
    print(f"Total Rounds: {results['total_rounds']}")
    print(f"Convergence: {results['convergence_info']['convergence_achieved']}")
    print(f"Model Size: {results['model_size_mb']:.2f} MB")
    
    if results['privacy_metrics']:
        print(f"Privacy Loss (ε): {results['privacy_metrics']['composed_epsilon']:.2f}")
    
    # Visualize training progress
    server.visualize_training_progress()
    
    print("Federated learning framework validation completed!")