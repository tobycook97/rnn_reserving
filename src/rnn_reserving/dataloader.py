import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 
from torch.nn import functional as F

from typing import List, Any
import logging



from .data_import import read_and_process_data
from .config import TrainingConfig

logger = logging.getLogger(__name__)

class InsuranceForecastDataset(Dataset):
    """
    Dataset for insurance time series forecasting.
    
    Input: periods 0..(seq_len-2)
    Target: periods 1..(seq_len-1) (next-step prediction)
    
    Args:
        input_seqs: List of input sequences (variable length)
        target_seqs: List of target sequences (variable length)
        lengths: Original sequence lengths before padding
        ids: Sequence identifiers
    """
    
    def __init__(
        self,
        input_seqs: List[np.ndarray],
        target_seqs: List[np.ndarray],
        lengths: List[int],
        ids: List[Any],
    ):
        
        assert len(input_seqs) == len(target_seqs) == len(lengths) == len(ids), \
            "All inputs must have the same length"
        
        self.n_samples = len(input_seqs)

        logger.info("Caching dataset in memory...")
        self.input_seqs = [torch.FloatTensor(seq) for seq in input_seqs]
        self.target_seqs = [torch.FloatTensor(seq) for seq in target_seqs]

        self.lengths = lengths
        self.ids = ids

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.input_seqs[idx]),
            self.lengths[idx],
            torch.FloatTensor(self.target_seqs[idx]),
            self.ids[idx]
        )


def collate_fn(
    batch,
    pad_value: float = -1.0,
):
    """ Pad both inputs and outputs to max of batch length """
    inputs, lengths, targets, ids = zip(*batch)

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_value)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)
    lengths_tensor = torch.LongTensor(lengths)

    batch = {
        'inputs': padded_inputs,
        'lengths': lengths_tensor,
        'targets': padded_targets,
        'ids': ids
    }

    return batch


def collate_fn(batch, pad_value: float = -1.0):
    """Pad both inputs and outputs to max of batch length
    
    This function pads both the input and target sequences in the batch to the same maximum length,
    We did have the issue where the targets and the inputs were being separately padded. This was then an issue for the val data where we aren't fillign in whole sequence
    """
    inputs, lengths, targets, ids = zip(*batch)

    # Find the global max length across both inputs and targets
    max_len = max(
        max(x.shape[0] for x in inputs),
        max(t.shape[0] for t in targets)
    )

    def pad_to_length(tensors, max_len, pad_value):
        padded = []
        for t in tensors:
            pad_size = max_len - t.shape[0]
            # Works for both 1D and 2D (seq_len, features) tensors
            pad_dims = (0, 0) * (t.dim() - 1) + (0, pad_size) # this tells the dims e.g. (0, 1) for 1D and (0, 0, 0, 1) for 2D. 
            padded.append(F.pad(t, pad_dims, value=pad_value))
        return torch.stack(padded)

    padded_inputs = pad_to_length(inputs, max_len, pad_value)
    padded_targets = pad_to_length(targets, max_len, pad_value)
    lengths_tensor = torch.LongTensor(lengths)
    print(padded_inputs)
    print(padded_targets)
    return {
        'inputs': padded_inputs,
        'lengths': lengths_tensor,
        'targets': padded_targets,
        'ids': ids
    }

class CollateFn:
    def __init__(self, pad_value: float = -1.0):
        self.pad_value = pad_value
    def __call__(self, batch):
        return collate_fn(batch, pad_value=self.pad_value)
    


def make_loaders(
    config: TrainingConfig
):
    """Create DataLoaders for training and validation datasets."""
    all_cols = config.target_cols + config.feature_cols if config.feature_cols else config.target_cols
    train_data, validation_data, _ = read_and_process_data(all_cols, data_debug=config.data_debug)
    
    train_data = InsuranceForecastDataset(train_data['inputs'], train_data['targets'], train_data['lengths'], train_data['ids'])
    val_data = InsuranceForecastDataset(
        validation_data['inputs'],
        validation_data['targets'],
        validation_data['lengths'],
        validation_data['ids']
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        collate_fn=CollateFn(pad_value=config.pad_value),
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CollateFn(pad_value=config.pad_value),
        num_workers=config.num_workers
    )

    return train_loader, val_loader