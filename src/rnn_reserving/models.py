from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUInsurance(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
    super(GRUInsurance, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.gru = nn.GRU(
        input_size,
        hidden_size,
        num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0
    )

    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x, lengths):
    """
    x: (batch, seq_len, input_size)
    lengths: (batch,)

    Returns: (batch, seq_len) - prediction at each timestep
    """

    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    packed_output, hidden = self.gru(packed)
    # unpack!
    output, _ = pad_packed_sequence(packed_output, batch_first=True)
    # output: (batch, seq_len, hidden_size)

    # predict at each timestep
    predictions = self.fc(output)
    # predictions: (batch, seq_len, 1)

    return predictions.squeeze(-1) # (batch, seq_len)