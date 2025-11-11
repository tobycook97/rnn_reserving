import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Callable
import wandb
from torch.utils.data import DataLoader

from dataclasses import asdict
from tqdm import tqdm
from rnn_reserving.config import TrainingConfig
from rnn_reserving.utils import save_checkpoint, load_checkpoint, setup_logging

# ============================================================================
# TRAINER, plus some useless boilerplate from my man Claude
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self) -> Callable:
        if self.mode == "min":
            return lambda curr, best: curr < best - self.min_delta
        else:
            return lambda curr, best: curr > best + self.min_delta
    
    def __call__(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.compare(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ============================================================================
# Metrics Tracker
# ============================================================================

class MetricsTracker:
    """Track and compute running metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * count
            self.counts[key] += count
    
    def compute(self) -> Dict[str, float]:
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Production-grade training pipeline."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics_fn: Optional[Callable] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.metrics_fn = metrics_fn
        
        # Setup
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min"
        )
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # Logging
        self.logger = setup_logging(
            log_file=f"{config.checkpoint_dir}/training.log"
        )
        
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=asdict(config),
                name=f"{config.model_name}_run"
            )
            wandb.watch(model, log="all", log_freq=100)
    
    def loss_metrics(self, output, target) -> Dict[str, float]:
        loss = self.criterion(output, target)
        return {'loss': loss.item()}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [Train]"
        )
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
           
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.optimizer.step()
        
            # Compute metrics
            batch_metrics = self.metrics_fn(output.detach(), target)
            batch_metrics['loss'] = loss.item()
            
            metrics_tracker.update(batch_metrics, data.size(0))
            
            # Update progress bar
            pbar.set_postfix(batch_metrics)
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    f"train/{k}": v
                    for k, v in batch_metrics.items()
                })
        
        return metrics_tracker.compute()
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [Val]"
        )
        
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Compute metrics
            batch_metrics = self.metrics_fn(output, target)
            batch_metrics['loss'] = loss.item()
            
            metrics_tracker.update(batch_metrics, data.size(0))
            pbar.set_postfix(batch_metrics)
        
        return metrics_tracker.compute()
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        # Resume from checkpoint if provided
        if resume_from:
            self.start_epoch, metrics = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.config.device
            )
            self.best_val_loss = metrics.get('val_loss', float('inf'))
            self.logger.info(f"Resumed from epoch {self.start_epoch}")
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Config: {asdict(self.config)}")
        
        try:
            for epoch in range(self.start_epoch + 1, self.config.epochs + 1):
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate(epoch)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"LR: {current_lr:.6f}"
                )
                
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'learning_rate': current_lr,
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()}
                    })
                
                # Save checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        {'train': train_metrics, 'val': val_metrics},
                        self.config,
                        'best_model.pt'
                    )
                    self.logger.info(f"Saved best model at epoch {epoch}")
                
                if not self.config.save_best_only:
                    if epoch % self.config.save_frequency == 0:
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            {'train': train_metrics, 'val': val_metrics},
                            self.config,
                            f'checkpoint_epoch_{epoch}.pt'
                        )
                
                # Early stopping
                if self.early_stopping(val_metrics['loss']):
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}"
                    )
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                {'train': train_metrics, 'val': val_metrics},
                self.config,
                'interrupted_checkpoint.pt'
            )
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            if self.config.use_wandb:
                wandb.finish()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")



def metrics_fn(output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute metrics given model output and target."""
    mse = nn.MSELoss()(output, target).item()
    mae = nn.L1Loss()(output, target).item()
    return {
        'mse': mse,
        'mae': mae
    }
