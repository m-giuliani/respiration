import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ThresholdEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.hidden2estimate = nn.Linear(hidden_size, 3)

    def forward(self, input, lengths):
        packed_sequence = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed_sequence)
        padded_outs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        out = self.hidden2estimate(padded_outs)
        return out