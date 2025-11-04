import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence 


class InsuranceForecastDataset(Dataset):
    """
    Input: periods 0..(seq_len-2)
    Target: periods 1..(seq_len-1)
    """
    def __init__(self, input_seqs, target_seqs, train_lengths, train_ids):

        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.train_lengths = train_lengths
        self.train_ids = train_ids
        # self.company_code_input = [id[1] for id in train_ids]

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.input_seqs[idx]),
            self.train_lengths[idx],
            torch.FloatTensor(self.target_seqs[idx]),
            self.train_ids[idx]
        )

def collate_fn(batch):
  """ Pad both inputs and outputs to max of batch length """
  inputs, lengths, targets, _ = zip(*batch)

  padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
  padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
  lengths_tensor = torch.LongTensor(lengths)

  return padded_inputs, lengths_tensor, padded_targets


