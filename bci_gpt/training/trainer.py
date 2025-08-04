"""Main training framework for BCI-GPT models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, List, Tuple, Any, Callable
import os
import json
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    warnings.warn("TensorBoard not available")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("Weights & Biases not available")

from ..core.models import BCIGPTModel
from ..core.inverse_gan import InverseSimulator
from .losses import BCILoss, ReconstructionLoss
from .augmentation import EEGAugmenter


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    language_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.1
    gan_loss_weight: float = 0.01
    
    # Validation
    validation_interval: int = 500  # steps
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_interval: int = 1000  # steps
    max_checkpoints: int = 5
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Logging
    log_interval: int = 100  # steps
    use_tensorboard: bool = True
    use_wandb: bool = False


class BCIDataset(Dataset):
    """Dataset for BCI-GPT training."""
    
    def __init__(self,
                 eeg_data: List[np.ndarray],
                 text_data: List[str],
                 tokenizer: Any,
                 max_length: int = 512,
                 eeg_length: int = 1000):
        """Initialize dataset.
        
        Args:
            eeg_data: List of EEG arrays (channels x samples)
            text_data: List of corresponding text strings
            tokenizer: Tokenizer for text processing
            max_length: Maximum text sequence length
            eeg_length: EEG sequence length
        """
        self.eeg_data = eeg_data
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eeg_length = eeg_length
        
        assert len(eeg_data) == len(text_data), "EEG and text data must have same length"
        
    def __len__(self) -> int:
        return len(self.eeg_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eeg = self.eeg_data[idx]
        text = self.text_data[idx]
        
        # Process EEG data
        if eeg.shape[1] != self.eeg_length:
            # Resample or pad/crop to target length
            if eeg.shape[1] > self.eeg_length:
                # Crop to center
                start = (eeg.shape[1] - self.eeg_length) // 2
                eeg = eeg[:, start:start + self.eeg_length]
            else:
                # Pad with reflection
                pad_samples = self.eeg_length - eeg.shape[1]
                eeg = np.pad(eeg, ((0, 0), (0, pad_samples)), mode='reflect')
        
        # Tokenize text
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            # Dummy tokens if no tokenizer
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        
        return {
            'eeg_data': torch.FloatTensor(eeg),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text
        }


class BCIGPTTrainer:
    """Trainer for BCI-GPT models with inverse simulation."""
    
    def __init__(self,
                 model: BCIGPTModel,
                 inverse_simulator: Optional[InverseSimulator] = None,
                 config: Optional[TrainingConfig] = None,
                 device: str = "cuda"):
        """Initialize trainer.
        
        Args:
            model: BCI-GPT model to train
            inverse_simulator: Optional inverse GAN for data augmentation
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.inverse_simulator = inverse_simulator
        self.config = config or TrainingConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Move models to device
        self.model.to(self.device)
        if self.inverse_simulator:
            self.inverse_simulator.to(self.device)
        
        # Initialize optimizers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.bci_loss = BCILoss()
        self.reconstruction_loss = ReconstructionLoss()
        
        # Initialize augmentation
        if self.config.use_augmentation:
            self.augmenter = EEGAugmenter()
        else:
            self.augmenter = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.logger = self._setup_logging()
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for different components."""
        param_groups = [
            {
                'params': self.model.eeg_encoder.parameters(),
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': self.model.fusion_layer.parameters(),
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            }
        ]
        
        # Lower learning rate for pretrained language model
        if hasattr(self.model, 'language_model'):
            param_groups.append({
                'params': self.model.language_model.parameters(),
                'lr': self.config.learning_rate * 0.1,  # 10x lower
                'weight_decay': self.config.weight_decay * 0.1
            })
        
        return optim.AdamW(param_groups)
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - self.config.warmup_steps) / max(1, 50000 - self.config.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_logging(self) -> Optional[Any]:
        """Setup logging with TensorBoard or Weights & Biases."""
        logger = None
        
        if self.config.use_wandb and HAS_WANDB:
            wandb.init(
                project="bci-gpt",
                config=asdict(self.config),
                name=f"bci-gpt-{int(time.time())}"
            )
            logger = wandb
            
        elif self.config.use_tensorboard and HAS_TENSORBOARD:
            log_dir = Path("logs") / f"bci-gpt-{int(time.time())}"
            log_dir.mkdir(parents=True, exist_ok=True)
            logger = SummaryWriter(log_dir)
            
        return logger
    
    def fit(self,
            train_data: str,
            val_data: Optional[str] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            **kwargs) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_data: Path to training data or DataLoader
            val_data: Path to validation data or DataLoader
            epochs: Number of epochs (overrides config)
            batch_size: Batch size (overrides config)
            **kwargs: Additional training arguments
            
        Returns:
            Training history dictionary
        """
        if epochs:
            self.config.epochs = epochs
        if batch_size:
            self.config.batch_size = batch_size
        
        # Load data
        train_loader = self._create_dataloader(train_data, shuffle=True)
        val_loader = self._create_dataloader(val_data, shuffle=False) if val_data else None
        
        # Training history
        history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }
        
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_perplexity'].append(train_metrics['perplexity'])
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader and (epoch + 1) % (self.config.validation_interval // len(train_loader) + 1) == 0:
                val_metrics = self._validate_epoch(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_perplexity'].append(val_metrics['perplexity'])
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Checkpoint saving
            if (epoch + 1) % (self.config.save_interval // len(train_loader) + 1) == 0:
                self._save_checkpoint()
            
            # Log epoch summary
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Perplexity: {train_metrics['perplexity']:.2f}")
            if val_loader:
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
        
        # Final checkpoint
        self._save_checkpoint(is_final=True)
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.inverse_simulator:
            self.inverse_simulator.train()
            
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss, metrics = self._compute_loss(batch)
            else:
                loss, metrics = self._compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            total_tokens += metrics.get('num_tokens', 1)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/perplexity': metrics.get('perplexity', np.exp(loss.item())),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/grad_norm': self._get_grad_norm(),
                }, step=self.global_step)
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_tokens': total_tokens
        }
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        if self.inverse_simulator:
            self.inverse_simulator.eval()
            
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        loss, metrics = self._compute_loss(batch)
                else:
                    loss, metrics = self._compute_loss(batch)
                
                total_loss += loss.item()
                total_tokens += metrics.get('num_tokens', 1)
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)
        
        # Log validation metrics
        self._log_metrics({
            'val/loss': avg_loss,
            'val/perplexity': perplexity,
        }, step=self.global_step)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_tokens': total_tokens
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute training loss."""
        eeg_data = batch['eeg_data']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Data augmentation
        if self.augmenter and self.training:
            if torch.rand(1) < self.config.augmentation_prob:
                eeg_data = self.augmenter.augment(eeg_data)
        
        # Forward pass through BCI-GPT
        outputs = self.model(eeg_data, input_ids, attention_mask)
        
        # Language modeling loss
        logits = outputs['logits']
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padded tokens
        
        language_loss = self.bci_loss.compute_language_loss(logits, labels)
        
        total_loss = self.config.language_loss_weight * language_loss
        
        # Reconstruction loss (if EEG features available)
        if 'eeg_features' in outputs:
            recon_loss = self.reconstruction_loss(outputs['eeg_features'], eeg_data)
            total_loss += self.config.reconstruction_loss_weight * recon_loss
        
        # GAN loss (if inverse simulator available and training)
        if self.inverse_simulator and self.training:
            gan_loss = self._compute_gan_loss(outputs, eeg_data)
            total_loss += self.config.gan_loss_weight * gan_loss
        
        # Calculate metrics
        with torch.no_grad():
            perplexity = torch.exp(language_loss).item()
            num_tokens = (attention_mask == 1).sum().item()
        
        metrics = {
            'perplexity': perplexity,
            'num_tokens': num_tokens,
            'language_loss': language_loss.item()
        }
        
        return total_loss, metrics
    
    def _compute_gan_loss(self, outputs: Dict[str, torch.Tensor], 
                         real_eeg: torch.Tensor) -> torch.Tensor:
        """Compute GAN loss for inverse simulation."""
        if 'text_features' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        text_features = outputs['text_features']
        batch_size = text_features.shape[0]
        
        # Average text features across sequence length
        text_embeddings = torch.mean(text_features, dim=1)  # (batch, hidden_dim)
        
        # Generate synthetic EEG
        fake_eeg = self.inverse_simulator.generate(text_embeddings)
        
        # Discriminator outputs
        real_outputs = self.inverse_simulator.discriminate(real_eeg, text_embeddings)
        fake_outputs = self.inverse_simulator.discriminate(fake_eeg, text_embeddings)
        
        # Generator loss (want fake to be classified as real)
        gen_losses = self.inverse_simulator.compute_generator_loss(
            fake_outputs, real_eeg, fake_eeg
        )
        
        return gen_losses['total_loss']
    
    def _create_dataloader(self, data_path: str, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from data path."""
        if isinstance(data_path, DataLoader):
            return data_path
        
        # For now, create dummy dataset - in practice, would load from files
        # This is a placeholder implementation
        dummy_eeg_data = [np.random.randn(9, 1000) for _ in range(100)]
        dummy_text_data = [f"Sample text {i}" for i in range(100)]
        
        dataset = BCIDataset(
            eeg_data=dummy_eeg_data,
            text_data=dummy_text_data,
            tokenizer=getattr(self.model, 'tokenizer', None),
            max_length=512,
            eeg_length=1000
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
    
    def _get_grad_norm(self) -> float:
        """Calculate gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to configured logger."""
        if self.logger is None:
            return
        
        if HAS_WANDB and isinstance(self.logger, type(wandb)):
            self.logger.log(metrics, step=step)
        elif HAS_TENSORBOARD:
            for key, value in metrics.items():
                self.logger.add_scalar(key, value, step)
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.inverse_simulator:
            checkpoint['inverse_simulator_state_dict'] = self.inverse_simulator.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        elif is_final:
            torch.save(checkpoint, checkpoint_dir / "final_model.pt")
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Remove old checkpoints
            self._cleanup_checkpoints(checkpoint_dir)
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path) -> None:
        """Keep only the most recent checkpoints."""
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        while len(checkpoints) > self.config.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'inverse_simulator_state_dict' in checkpoint and self.inverse_simulator:
            self.inverse_simulator.load_state_dict(checkpoint['inverse_simulator_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from step {self.global_step}")
    
    def save_model(self, save_path: str) -> None:
        """Save trained model for inference."""
        self.model.save_pretrained(save_path)
        
        # Save training config
        config_path = Path(save_path).parent / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)