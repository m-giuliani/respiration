import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forget_gate_1 = nn.Linear(input_size+hidden_size, 1)
        self.hidden = torch.randn(hidden_size)
        self.forget_gate_2 = nn.Linear(input_size+hidden_size, 1)
        self.cell_state = torch.randn(hidden_size)
        self.cell_gate = nn.Linear(input_size+hidden_size, 1)


    def forward(self, x, h):
        xh = torch.cat([x, h], dim=2)
        fg = F.sigmoid(self.forget_gate_1(xh))
        c_now = h * fg
        c_new = F.tanh(self.cell_gate(xh))
        c_now += F.sigmoid(self.forget_gate_2(xh)) * c_new
        ...
        

