import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 
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
    pad_value: float = 0.0,
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
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )


    return train_loader, val_loader