import torch
from torch import nn
from .RNN import RNN

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sigma=0.01):
        super().__init__()

        self.input_size = input_size

        self.rnn_f = RNN(input_size, hidden_size, sigma)
        self.rnn_b = RNN(input_size, hidden_size, sigma)
        
        self.hidden_size *= 2

    def forward(self, X, state=None):

        if state is None:
            state = (None, None)
        
        outputs_f, state_f = self.rnn_f(X, state[0])
        outputs_b, state_b = self.rnn_b(torch.flip(X, dims=[0]), state[1])

        outputs_b = torch.flip(outputs_b, dims=[0])

        outputs = torch.cat([outputs_f, outputs_b], dim=2)

        return outputs, (state_f, state_b)



