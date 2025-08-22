"""Distributed computation framework for large-scale BCI-GPT research experiments."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
import os
import psutil
import ray
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask
from dask.distributed import Client, as_completed as dask_as_completed
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed computation."""
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    use_cuda: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_workers: int = 4
    batch_size_per_gpu: int = 32
    
    # Ray configuration
    use_ray: bool = False
    ray_address: Optional[str] = None
    ray_num_cpus: Optional[int] = None
    ray_num_gpus: Optional[int] = None
    
    # Dask configuration  
    use_dask: bool = False
    dask_scheduler_address: Optional[str] = None
    dask_n_workers: int = 4
    
    def __post_init__(self):
        if self.ray_num_cpus is None:
            self.ray_num_cpus = psutil.cpu_count()
        if self.ray_num_gpus is None:
            self.ray_num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0


class DistributedTrainer:
    """Distributed training for BCI-GPT models."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.device = None
        self.local_rank = None
        
    def setup_distributed(self, rank: int, world_size: int):
        """Initialize distributed training."""
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        if self.config.use_cuda and torch.cuda.is_available():
            self.local_rank = rank % torch.cuda.device_count()
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cpu')
            self.local_rank = rank
        
        logger.info(f"Rank {rank}: Initialized distributed training on {self.device}")
    
    def cleanup_distributed(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        model = model.to(self.device)
        
        if dist.is_initialized() and torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model
    
    def create_distributed_dataloader(self, 
                                    dataset: torch.utils.data.Dataset,
                                    batch_size: Optional[int] = None,
                                    shuffle: bool = True) -> DataLoader:
        """Create distributed data loader."""
        
        batch_size = batch_size or self.config.batch_size_per_gpu
        
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True,
                num_workers=2
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=2
            )
        
        return dataloader
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor across all processes."""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        return tensor
    
    def gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes."""
        if not dist.is_initialized():
            return [tensor]
        
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        
        return gathered_tensors


class RayDistributedFramework:
    """Ray-based distributed framework for parallel experiments."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.ray_initialized = False
        
    def initialize_ray(self):
        """Initialize Ray cluster."""
        if not self.ray_initialized:
            try:
                if self.config.ray_address:
                    ray.init(address=self.config.ray_address)
                else:
                    ray.init(
                        num_cpus=self.config.ray_num_cpus,
                        num_gpus=self.config.ray_num_gpus
                    )
                self.ray_initialized = True
                logger.info(f"Ray initialized with {ray.cluster_resources()}")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}")
                self.ray_initialized = False
    
    def shutdown_ray(self):
        """Shutdown Ray cluster."""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
    
    @ray.remote
    class RemoteExperiment:
        """Remote experiment execution on Ray."""
        
        def __init__(self, experiment_config: Dict[str, Any]):
            self.config = experiment_config
            self.results = []
        
        def run_experiment(self, 
                         model_config: Dict[str, Any],
                         data_config: Dict[str, Any],
                         seed: int) -> Dict[str, Any]:
            """Run single experiment remotely."""
            
            # Set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Simulate experiment (replace with actual implementation)
            import time
            start_time = time.time()
            
            # Mock experimental results
            accuracy = np.random.normal(0.85, 0.05)
            f1_score = np.random.normal(0.83, 0.04)
            
            runtime = time.time() - start_time
            
            return {
                'model_config': model_config,
                'data_config': data_config,
                'seed': seed,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'runtime': runtime,
                'worker_id': ray.get_runtime_context().worker.worker_id
            }
    
    def run_parallel_experiments(self, 
                               experiment_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run experiments in parallel using Ray."""
        
        if not self.ray_initialized:
            self.initialize_ray()
        
        if not self.ray_initialized:
            logger.error("Ray not available, falling back to sequential execution")
            return self._run_sequential_experiments(experiment_configs)
        
        # Create remote experiment workers
        remote_experiments = []
        for config in experiment_configs:
            remote_exp = self.RemoteExperiment.remote(config)
            remote_experiments.append(remote_exp)
        
        # Submit all experiments
        futures = []
        for i, (remote_exp, config) in enumerate(zip(remote_experiments, experiment_configs)):
            future = remote_exp.run_experiment.remote(
                config.get('model_config', {}),
                config.get('data_config', {}),
                config.get('seed', 42)
            )
            futures.append(future)
        
        # Collect results
        logger.info(f"Running {len(futures)} experiments in parallel...")
        results = ray.get(futures)
        
        logger.info(f"Completed {len(results)} parallel experiments")
        return results
    
    def _run_sequential_experiments(self, 
                                  experiment_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback sequential execution."""
        results = []
        
        for i, config in enumerate(experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment_configs)}")
            
            # Create local experiment instance
            local_exp = self.RemoteExperiment.__new__(self.RemoteExperiment)
            local_exp.__init__(config)
            
            result = local_exp.run_experiment(
                config.get('model_config', {}),
                config.get('data_config', {}),
                config.get('seed', 42)
            )
            results.append(result)
        
        return results


class DaskDistributedFramework:
    """Dask-based distributed framework for data processing."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.client = None
        
    def initialize_dask(self):
        """Initialize Dask cluster."""
        try:
            if self.config.dask_scheduler_address:
                self.client = Client(self.config.dask_scheduler_address)
            else:
                # Local cluster
                self.client = Client(n_workers=self.config.dask_n_workers)
            
            logger.info(f"Dask client initialized: {self.client}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize Dask: {e}")
            self.client = None
            return False
    
    def shutdown_dask(self):
        """Shutdown Dask client."""
        if self.client:
            self.client.close()
            self.client = None
    
    def parallel_data_processing(self, 
                               data_chunks: List[np.ndarray],
                               processing_func: Callable) -> List[Any]:
        """Process data chunks in parallel."""
        
        if not self.client:
            if not self.initialize_dask():
                return self._sequential_processing(data_chunks, processing_func)
        
        # Submit processing jobs
        futures = []
        for chunk in data_chunks:
            future = self.client.submit(processing_func, chunk)
            futures.append(future)
        
        # Collect results
        results = self.client.gather(futures)
        return results
    
    def distributed_cross_validation(self, 
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   model_func: Callable,
                                   cv_folds: int = 5) -> Dict[str, Any]:
        """Distributed cross-validation."""
        
        from sklearn.model_selection import StratifiedKFold
        
        if not self.client:
            if not self.initialize_dask():
                return self._sequential_cv(X, y, model_func, cv_folds)
        
        # Create cross-validation splits
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
        
        # Submit fold processing jobs
        futures = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            future = self.client.submit(
                self._process_cv_fold,
                X[train_idx], y[train_idx],
                X[test_idx], y[test_idx],
                model_func, fold
            )
            futures.append(future)
        
        # Collect results
        fold_results = self.client.gather(futures)
        
        # Aggregate results
        aggregated = self._aggregate_cv_results(fold_results)
        return aggregated
    
    def _process_cv_fold(self, 
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        model_func: Callable, fold: int) -> Dict[str, Any]:
        """Process single CV fold."""
        
        # Create and train model
        model = model_func()
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        return {
            'fold': fold,
            'accuracy': accuracy,
            'f1_score': f1,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        
        accuracies = [result['accuracy'] for result in fold_results]
        f1_scores = [result['f1_score'] for result in fold_results]
        
        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'fold_results': fold_results
        }
    
    def _sequential_processing(self, 
                             data_chunks: List[np.ndarray],
                             processing_func: Callable) -> List[Any]:
        """Fallback sequential processing."""
        return [processing_func(chunk) for chunk in data_chunks]
    
    def _sequential_cv(self, 
                      X: np.ndarray, y: np.ndarray,
                      model_func: Callable, cv_folds: int) -> Dict[str, Any]:
        """Fallback sequential cross-validation."""
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, f1_score
        
        model = model_func()
        
        # Calculate scores
        accuracy_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        f1_scorer = make_scorer(f1_score, average='weighted')
        f1_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=f1_scorer)
        
        return {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'fold_results': [
                {'fold': i, 'accuracy': acc, 'f1_score': f1} 
                for i, (acc, f1) in enumerate(zip(accuracy_scores, f1_scores))
            ]
        }


class HyperparameterOptimization:
    """Distributed hyperparameter optimization."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.ray_framework = RayDistributedFramework(config)
        
    def bayesian_optimization(self, 
                            objective_func: Callable,
                            space: Dict[str, Any],
                            n_trials: int = 100) -> Dict[str, Any]:
        """Bayesian hyperparameter optimization using Ray Tune."""
        
        try:
            from ray import tune
            from ray.tune.search.bayesopt import BayesOptSearch
            
            # Initialize Ray if not already done
            self.ray_framework.initialize_ray()
            
            # Define search algorithm
            search_alg = BayesOptSearch(space)
            
            # Run optimization
            analysis = tune.run(
                objective_func,
                search_alg=search_alg,
                num_samples=n_trials,
                resources_per_trial={"cpu": 1, "gpu": 0.5 if torch.cuda.is_available() else 0}
            )
            
            # Get best configuration
            best_config = analysis.best_config
            best_score = analysis.best_result
            
            return {
                'best_config': best_config,
                'best_score': best_score,
                'all_results': analysis.results_df,
                'analysis': analysis
            }
            
        except ImportError:
            logger.warning("Ray Tune not available, using grid search")
            return self.grid_search(objective_func, space, n_trials)
    
    def grid_search(self, 
                   objective_func: Callable,
                   space: Dict[str, Any],
                   max_trials: int = 100) -> Dict[str, Any]:
        """Grid search hyperparameter optimization."""
        
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(space.keys())
        param_values = list(space.values())
        
        combinations = list(product(*param_values))[:max_trials]
        
        # Create experiment configurations
        experiment_configs = []
        for i, combination in enumerate(combinations):
            config = dict(zip(param_names, combination))
            experiment_configs.append({
                'trial_id': i,
                'hyperparameters': config
            })
        
        # Run experiments in parallel
        results = self.ray_framework.run_parallel_experiments(experiment_configs)
        
        # Find best configuration
        best_result = max(results, key=lambda x: x.get('accuracy', 0))
        
        return {
            'best_config': best_result.get('hyperparameters', {}),
            'best_score': best_result.get('accuracy', 0),
            'all_results': results
        }


class DistributedExperimentManager:
    """Manage large-scale distributed experiments."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.trainer = DistributedTrainer(config)
        self.ray_framework = RayDistributedFramework(config)
        self.dask_framework = DaskDistributedFramework(config)
        self.hyperopt = HyperparameterOptimization(config)
        
    def run_distributed_benchmark(self, 
                                 models: List[str],
                                 datasets: List[str],
                                 hyperparams: List[Dict[str, Any]],
                                 seeds: List[int]) -> Dict[str, Any]:
        """Run comprehensive distributed benchmark."""
        
        logger.info(f"Starting distributed benchmark:")
        logger.info(f"  Models: {len(models)}")
        logger.info(f"  Datasets: {len(datasets)}")
        logger.info(f"  Hyperparameter sets: {len(hyperparams)}")
        logger.info(f"  Seeds: {len(seeds)}")
        
        total_experiments = len(models) * len(datasets) * len(hyperparams) * len(seeds)
        logger.info(f"  Total experiments: {total_experiments}")
        
        # Generate experiment configurations
        experiment_configs = []
        experiment_id = 0
        
        for model in models:
            for dataset in datasets:
                for hyperparam in hyperparams:
                    for seed in seeds:
                        config = {
                            'experiment_id': experiment_id,
                            'model': model,
                            'dataset': dataset,
                            'hyperparameters': hyperparam,
                            'seed': seed
                        }
                        experiment_configs.append(config)
                        experiment_id += 1
        
        # Run experiments in parallel
        start_time = time.time()
        results = self.ray_framework.run_parallel_experiments(experiment_configs)
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_benchmark_results(results)
        
        benchmark_summary = {
            'total_experiments': total_experiments,
            'successful_experiments': len(results),
            'total_runtime_seconds': total_time,
            'average_runtime_per_experiment': total_time / len(results) if results else 0,
            'analysis': analysis,
            'raw_results': results
        }
        
        logger.info(f"Distributed benchmark completed:")
        logger.info(f"  Successful experiments: {len(results)}/{total_experiments}")
        logger.info(f"  Total runtime: {total_time:.1f} seconds")
        logger.info(f"  Average per experiment: {total_time/len(results):.2f} seconds")
        
        return benchmark_summary
    
    def _analyze_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        
        df = pd.DataFrame(results)
        
        analysis = {
            'overall_stats': {
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'best_accuracy': df['accuracy'].max(),
                'worst_accuracy': df['accuracy'].min()
            },
            'model_comparison': df.groupby('model_config')['accuracy'].agg(['mean', 'std']).to_dict(),
            'dataset_performance': df.groupby('data_config')['accuracy'].agg(['mean', 'std']).to_dict(),
            'seed_stability': df.groupby('seed')['accuracy'].agg(['mean', 'std']).to_dict()
        }
        
        return analysis
    
    def optimize_hyperparameters(self, 
                                model_type: str,
                                dataset_name: str,
                                search_space: Dict[str, Any],
                                n_trials: int = 50) -> Dict[str, Any]:
        """Distributed hyperparameter optimization."""
        
        logger.info(f"Starting hyperparameter optimization:")
        logger.info(f"  Model: {model_type}")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Search space: {search_space}")
        logger.info(f"  Trials: {n_trials}")
        
        def objective(config):
            # Mock objective function (replace with actual implementation)
            import random
            accuracy = random.uniform(0.7, 0.9)
            return {"accuracy": accuracy, "config": config}
        
        # Run optimization
        optimization_result = self.hyperopt.bayesian_optimization(
            objective_func=objective,
            space=search_space,
            n_trials=n_trials
        )
        
        logger.info(f"Hyperparameter optimization completed:")
        logger.info(f"  Best score: {optimization_result['best_score']}")
        logger.info(f"  Best config: {optimization_result['best_config']}")
        
        return optimization_result
    
    def cleanup(self):
        """Clean up distributed resources."""
        self.trainer.cleanup_distributed()
        self.ray_framework.shutdown_ray()
        self.dask_framework.shutdown_dask()


# Example usage and validation
if __name__ == "__main__":
    # Configuration for distributed execution
    config = DistributedConfig(
        use_ray=True,
        use_dask=True,
        max_workers=4,
        batch_size_per_gpu=16
    )
    
    # Create distributed experiment manager
    manager = DistributedExperimentManager(config)
    
    # Example: Distributed benchmark
    benchmark_result = manager.run_distributed_benchmark(
        models=["SimpleTransformer", "CNN", "LSTM"],
        datasets=["SyntheticEEG", "RealEEG"],
        hyperparams=[
            {"learning_rate": 0.001, "hidden_dim": 128},
            {"learning_rate": 0.01, "hidden_dim": 256}
        ],
        seeds=[42, 123, 456]
    )
    
    print("Distributed Benchmark Results:")
    print(f"Total experiments: {benchmark_result['total_experiments']}")
    print(f"Runtime: {benchmark_result['total_runtime_seconds']:.1f}s")
    print(f"Best accuracy: {benchmark_result['analysis']['overall_stats']['best_accuracy']:.3f}")
    
    # Example: Hyperparameter optimization
    search_space = {
        "learning_rate": [0.001, 0.01, 0.1],
        "hidden_dim": [64, 128, 256, 512],
        "n_layers": [2, 4, 6]
    }
    
    optimization_result = manager.optimize_hyperparameters(
        model_type="SimpleTransformer",
        dataset_name="SyntheticEEG",
        search_space=search_space,
        n_trials=20
    )
    
    print("Hyperparameter Optimization Results:")
    print(f"Best configuration: {optimization_result['best_config']}")
    print(f"Best score: {optimization_result['best_score']}")
    
    # Cleanup
    manager.cleanup()
    
    print("Distributed computation framework validation completed!")