from dataclasses import dataclass, field
import torch
import logging 
# ============================================================================
# Configuration
# ============================================================================
logger = logging.getLogger(__name__)

@dataclass
class BaseConfig:
    """Base configuration parameters."""
    data_path: str = "./data/raw/ppauto_pos.csv"
    log_file: str = "./logs/training.log"

    # Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration parameters."""
    # Model
    model_name: str = "gru_model"
    experiment_name: str = "rnn_reserving_experiment"
    run_name: str = "run_1"
    
    # columns
    feature_cols: list[str] = field(default_factory=list)  # to be set externally. Not including our target which is also a feature!
    target_cols: list[str] = field(default_factory=lambda: ["paid_loss_ratio"])
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    shuffle_train: bool = True

    # Optimization
    grad_clip: float = 1.0 # helps avoid exploding gradients
    use_mixed_precision: bool = True # not added yet
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, step, plateau # what is hti
    scheduler_patience: int = 10 # what is this 
    scheduler_factor: float = 0.5 # what is this
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs
    
    # Logging
    log_interval: int = 10  # Log every N batches
    use_wandb: bool = True
    wandb_project: str = "timeseries-forecasting"
    
    deterministic: bool = False
    
    # Device
    num_workers: int = 4


@dataclass
class ModelConfig(BaseConfig):
    """Model configuration parameters."""
    input_size: int = 1  # to be set externally
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if self.num_layers == 1 and self.dropout > 0:
            logger.warning(
                f"dropout={self.dropout} ignored because num_layers=1. "
                "GRU only applies dropout between layers."
            )