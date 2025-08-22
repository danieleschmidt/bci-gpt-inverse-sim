"""Reproducibility framework for BCI-GPT research ensuring scientific rigor."""

import torch
import numpy as np
import random
import os
import hashlib
import json
import pickle
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import git
import logging
from datetime import datetime
import platform
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for ensuring reproducible experiments."""
    random_seed: int = 42
    torch_deterministic: bool = True
    cuda_deterministic: bool = True
    python_hash_seed: str = "0"
    track_git_state: bool = True
    track_system_info: bool = True
    save_environment: bool = True
    checksum_data: bool = True
    
    def __post_init__(self):
        """Apply reproducibility settings."""
        self.set_global_seeds()
        self.configure_determinism()
        if self.save_environment:
            os.environ['PYTHONHASHSEED'] = self.python_hash_seed
    
    def set_global_seeds(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
    
    def configure_determinism(self):
        """Configure PyTorch for deterministic operations."""
        if self.torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        if self.cuda_deterministic and torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)


@dataclass
class ExperimentState:
    """Complete state information for experiment reproducibility."""
    experiment_id: str
    timestamp: datetime
    reproducibility_config: ReproducibilityConfig
    git_state: Optional[Dict[str, str]] = None
    system_info: Optional[Dict[str, Any]] = None
    environment_info: Optional[Dict[str, str]] = None
    data_checksums: Optional[Dict[str, str]] = None
    model_checksums: Optional[Dict[str, str]] = None
    dependencies: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp.isoformat(),
            'reproducibility_config': self.reproducibility_config.__dict__,
            'git_state': self.git_state,
            'system_info': self.system_info,
            'environment_info': self.environment_info,
            'data_checksums': self.data_checksums,
            'model_checksums': self.model_checksums,
            'dependencies': self.dependencies
        }


class ReproducibilityTracker:
    """Track and ensure reproducibility of BCI-GPT experiments."""
    
    def __init__(self, 
                 config: Optional[ReproducibilityConfig] = None,
                 base_path: str = "./reproducibility"):
        self.config = config or ReproducibilityConfig()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Apply reproducibility settings
        self.config.set_global_seeds()
        self.config.configure_determinism()
        
    def capture_experiment_state(self, experiment_id: str) -> ExperimentState:
        """Capture complete experiment state for reproducibility."""
        
        state = ExperimentState(
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            reproducibility_config=self.config
        )
        
        # Capture git state
        if self.config.track_git_state:
            state.git_state = self._capture_git_state()
        
        # Capture system information
        if self.config.track_system_info:
            state.system_info = self._capture_system_info()
        
        # Capture environment
        if self.config.save_environment:
            state.environment_info = self._capture_environment()
        
        # Capture dependencies
        state.dependencies = self._capture_dependencies()
        
        return state
    
    def _capture_git_state(self) -> Dict[str, str]:
        """Capture git repository state."""
        try:
            repo = git.Repo(search_parent_directories=True)
            
            git_state = {
                'commit_hash': repo.head.commit.hexsha,
                'branch': repo.active_branch.name,
                'remote_url': repo.remotes.origin.url if repo.remotes else None,
                'is_dirty': repo.is_dirty(),
                'untracked_files': [str(f) for f in repo.untracked_files],
                'commit_message': repo.head.commit.message.strip(),
                'commit_author': str(repo.head.commit.author),
                'commit_date': repo.head.commit.committed_datetime.isoformat()
            }
            
            # Capture diff if dirty
            if repo.is_dirty():
                git_state['diff'] = repo.git.diff()
            
            return git_state
            
        except Exception as e:
            logger.warning(f"Could not capture git state: {e}")
            return {'error': str(e)}
    
    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture system and hardware information."""
        try:
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_info': self._get_gpu_info(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
            }
            
            return system_info
            
        except Exception as e:
            logger.warning(f"Could not capture system info: {e}")
            return {'error': str(e)}
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information if available."""
        gpu_info = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'device': i,
                    'name': props.name,
                    'memory_total_mb': props.total_memory / (1024**2),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        
        return gpu_info
    
    def _capture_environment(self) -> Dict[str, str]:
        """Capture environment variables."""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES',
            'PYTHONPATH',
            'PYTHONHASHSEED',
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS'
        ]
        
        env_info = {}
        for var in relevant_vars:
            if var in os.environ:
                env_info[var] = os.environ[var]
        
        return env_info
    
    def _capture_dependencies(self) -> Dict[str, str]:
        """Capture Python package versions."""
        try:
            import pkg_resources
            
            dependencies = {}
            for package in pkg_resources.working_set:
                dependencies[package.project_name] = package.version
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Could not capture dependencies: {e}")
            return {'error': str(e)}
    
    def calculate_file_checksum(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 checksum of a file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def calculate_data_checksums(self, data_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Calculate checksums for all data files."""
        checksums = {}
        
        for path in data_paths:
            path = Path(path)
            if path.is_file():
                checksums[str(path)] = self.calculate_file_checksum(path)
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        checksums[str(file_path)] = self.calculate_file_checksum(file_path)
        
        return checksums
    
    def calculate_model_checksum(self, model: torch.nn.Module) -> str:
        """Calculate checksum of model parameters."""
        model_bytes = pickle.dumps(model.state_dict())
        return hashlib.sha256(model_bytes).hexdigest()
    
    def save_experiment_state(self, state: ExperimentState) -> Path:
        """Save experiment state to file."""
        state_path = self.base_path / f"{state.experiment_id}_state.json"
        
        with open(state_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved experiment state: {state_path}")
        return state_path
    
    def load_experiment_state(self, experiment_id: str) -> ExperimentState:
        """Load experiment state from file."""
        state_path = self.base_path / f"{experiment_id}_state.json"
        
        if not state_path.exists():
            raise FileNotFoundError(f"Experiment state not found: {state_path}")
        
        with open(state_path, 'r') as f:
            state_dict = json.load(f)
        
        # Reconstruct ExperimentState object
        config_dict = state_dict['reproducibility_config']
        config = ReproducibilityConfig(**config_dict)
        
        state = ExperimentState(
            experiment_id=state_dict['experiment_id'],
            timestamp=datetime.fromisoformat(state_dict['timestamp']),
            reproducibility_config=config,
            git_state=state_dict.get('git_state'),
            system_info=state_dict.get('system_info'),
            environment_info=state_dict.get('environment_info'),
            data_checksums=state_dict.get('data_checksums'),
            model_checksums=state_dict.get('model_checksums'),
            dependencies=state_dict.get('dependencies')
        )
        
        return state
    
    def verify_reproducibility(self, 
                             experiment_id: str,
                             current_data_paths: Optional[List[Union[str, Path]]] = None,
                             current_model: Optional[torch.nn.Module] = None) -> Dict[str, bool]:
        """Verify current environment matches saved experiment state."""
        
        saved_state = self.load_experiment_state(experiment_id)
        verification_results = {}
        
        # Verify git state
        if saved_state.git_state and self.config.track_git_state:
            current_git = self._capture_git_state()
            verification_results['git_commit_match'] = (
                current_git.get('commit_hash') == saved_state.git_state.get('commit_hash')
            )
            verification_results['git_clean'] = not current_git.get('is_dirty', True)
        
        # Verify system compatibility
        if saved_state.system_info:
            current_system = self._capture_system_info()
            verification_results['python_version_match'] = (
                current_system.get('python_version') == saved_state.system_info.get('python_version')
            )
            verification_results['torch_version_match'] = (
                current_system.get('torch_version') == saved_state.system_info.get('torch_version')
            )
            verification_results['cuda_available_match'] = (
                current_system.get('cuda_available') == saved_state.system_info.get('cuda_available')
            )
        
        # Verify data checksums
        if current_data_paths and saved_state.data_checksums:
            current_checksums = self.calculate_data_checksums(current_data_paths)
            verification_results['data_checksums_match'] = (
                current_checksums == saved_state.data_checksums
            )
        
        # Verify model checksum
        if current_model and saved_state.model_checksums:
            current_model_checksum = self.calculate_model_checksum(current_model)
            saved_model_checksum = saved_state.model_checksums.get('main_model')
            verification_results['model_checksum_match'] = (
                current_model_checksum == saved_model_checksum
            )
        
        # Overall reproducibility score
        verified_checks = sum(verification_results.values())
        total_checks = len(verification_results)
        verification_results['reproducibility_score'] = verified_checks / total_checks if total_checks > 0 else 1.0
        
        return verification_results
    
    def create_reproducibility_package(self, 
                                     experiment_id: str,
                                     data_paths: List[Union[str, Path]],
                                     model: torch.nn.Module,
                                     additional_files: Optional[List[Union[str, Path]]] = None) -> Path:
        """Create complete reproducibility package."""
        
        # Capture experiment state
        state = self.capture_experiment_state(experiment_id)
        
        # Calculate data checksums
        if self.config.checksum_data:
            state.data_checksums = self.calculate_data_checksums(data_paths)
        
        # Calculate model checksum
        state.model_checksums = {'main_model': self.calculate_model_checksum(model)}
        
        # Save experiment state
        state_path = self.save_experiment_state(state)
        
        # Create reproducibility report
        report_path = self._create_reproducibility_report(state)
        
        # Create requirements.txt
        requirements_path = self._create_requirements_file(experiment_id, state.dependencies)
        
        logger.info(f"Created reproducibility package for experiment: {experiment_id}")
        return self.base_path / experiment_id
    
    def _create_reproducibility_report(self, state: ExperimentState) -> Path:
        """Create human-readable reproducibility report."""
        
        report_path = self.base_path / f"{state.experiment_id}_reproducibility_report.md"
        
        report_content = f"""# Reproducibility Report

**Experiment ID:** {state.experiment_id}  
**Timestamp:** {state.timestamp}  
**Random Seed:** {state.reproducibility_config.random_seed}

## Git State
"""
        
        if state.git_state:
            report_content += f"""
- **Commit Hash:** {state.git_state.get('commit_hash', 'N/A')}
- **Branch:** {state.git_state.get('branch', 'N/A')}
- **Repository Clean:** {not state.git_state.get('is_dirty', True)}
- **Commit Message:** {state.git_state.get('commit_message', 'N/A')}
"""
        
        report_content += "\n## System Information\n"
        
        if state.system_info:
            report_content += f"""
- **Platform:** {state.system_info.get('platform', 'N/A')}
- **Python Version:** {state.system_info.get('python_version', 'N/A')}
- **PyTorch Version:** {state.system_info.get('torch_version', 'N/A')}
- **CUDA Available:** {state.system_info.get('cuda_available', 'N/A')}
- **GPU Count:** {len(state.system_info.get('gpu_info', []))}
"""
        
        report_content += "\n## Dependencies\n"
        
        if state.dependencies:
            key_packages = ['torch', 'numpy', 'scipy', 'scikit-learn', 'transformers']
            for package in key_packages:
                if package in state.dependencies:
                    report_content += f"- **{package}:** {state.dependencies[package]}\n"
        
        report_content += f"""
## Data Integrity

- **Data Files Checksummed:** {len(state.data_checksums or {})}
- **Model Checksum:** {list((state.model_checksums or {}).values())[0] if state.model_checksums else 'N/A'}

## Reproducibility Configuration

- **Deterministic Operations:** {state.reproducibility_config.torch_deterministic}
- **CUDA Deterministic:** {state.reproducibility_config.cuda_deterministic}
- **Python Hash Seed:** {state.reproducibility_config.python_hash_seed}

## Instructions for Reproduction

1. Clone the repository and checkout commit: `{state.git_state.get('commit_hash', 'N/A') if state.git_state else 'N/A'}`
2. Install dependencies: `pip install -r {state.experiment_id}_requirements.txt`
3. Set environment variables as specified in the experiment state
4. Run with random seed: {state.reproducibility_config.random_seed}

## Data Files

"""
        
        if state.data_checksums:
            for file_path, checksum in state.data_checksums.items():
                report_content += f"- `{file_path}`: {checksum[:16]}...\n"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Created reproducibility report: {report_path}")
        return report_path
    
    def _create_requirements_file(self, 
                                experiment_id: str, 
                                dependencies: Optional[Dict[str, str]]) -> Path:
        """Create requirements.txt file for the experiment."""
        
        requirements_path = self.base_path / f"{experiment_id}_requirements.txt"
        
        if dependencies:
            with open(requirements_path, 'w') as f:
                for package, version in sorted(dependencies.items()):
                    f.write(f"{package}=={version}\n")
        else:
            with open(requirements_path, 'w') as f:
                f.write("# Dependencies could not be captured\n")
        
        logger.info(f"Created requirements file: {requirements_path}")
        return requirements_path


# Example usage and validation
if __name__ == "__main__":
    # Create reproducibility tracker
    config = ReproducibilityConfig(
        random_seed=42,
        torch_deterministic=True,
        cuda_deterministic=True
    )
    
    tracker = ReproducibilityTracker(config)
    
    # Create a simple model for demonstration
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Simulate experiment
    experiment_id = "reproducibility_test_20250122"
    
    # Create reproducibility package
    package_path = tracker.create_reproducibility_package(
        experiment_id=experiment_id,
        data_paths=["./bci_gpt/"],  # Use existing code as "data"
        model=model
    )
    
    print(f"Created reproducibility package: {package_path}")
    
    # Verify reproducibility
    verification = tracker.verify_reproducibility(
        experiment_id=experiment_id,
        current_data_paths=["./bci_gpt/"],
        current_model=model
    )
    
    print(f"Reproducibility verification:")
    for check, passed in verification.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    print(f"Overall reproducibility score: {verification['reproducibility_score']*100:.1f}%")