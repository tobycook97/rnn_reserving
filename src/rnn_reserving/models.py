import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple

from rnn_reserving.config import ModelConfig
import logging

logger = logging.getLogger(__name__)

class GRUInsurance(nn.Module):
    """
    GRU model for insurance time series forecasting.
    
    Supports:
    - Packed sequences for efficient variable-length processing
    - Multiple output activations - lol nice thanks Claude  

    - Removed weight initialization methods for simplicity (we can add them back if needed)  
    Args:
        config: GRUModelConfig with architecture parameters
    """
    
    def __init__(self, config: ModelConfig):
        
        super(GRUInsurance, self).__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # Output layer
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        
        # Initialize weights
        self._initialize_weights(config.init_method)
        
        # Log model info
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"GRU Model Architecture:")
        logger.info(f"  Input size: {self.config.input_size}")
        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Num layers: {self.config.num_layers}")
        logger.info(f"  Bidirectional: {self.config.bidirectional}")
        logger.info(f"  Dropout: {self.config.dropout}")
        logger.info(f"  Output size: {self.config.output_size}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with packed sequences.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            lengths: Actual sequence lengths (batch,)
            return_hidden: If True, also return final hidden state
        
        Returns:
            predictions: (batch, seq_len, output_size) or (batch, seq_len) if output_size=1
            hidden: (optional) Final hidden state (num_layers * num_directions, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pack sequences for efficient processing
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),  # lengths must be on CPU
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through GRU
        packed_output, hidden = self.gru(packed)
        
        # Unpack output
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len  # Ensure same length as input
        )
        # output: (batch, seq_len, hidden_size)
        
        # Apply output layer
        predictions = self.fc(output)
        # predictions: (batch, seq_len, output_size)
        
        # Squeeze if single output
        if self.config.output_size == 1:
            predictions = predictions.squeeze(-1)  # (batch, seq_len)
        
        if return_hidden:
            return predictions, hidden
        return predictions
    
    def predict_next_step(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict only the next timestep (useful for autoregressive forecasting).
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            lengths: Actual sequence lengths (batch,)
        
        Returns:
            predictions: (batch, output_size) - prediction at last timestep
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, lengths)
            
            # Get predictions at actual sequence end (not padding)
            batch_indices = torch.arange(len(lengths))
            last_predictions = predictions[batch_indices, lengths - 1]
            
            return last_predictions
    
