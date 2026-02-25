import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, sigma=0.01):
        # batch has shape [batch_size, num_steps, emb_dims]
        # since RNN is a layer, and processes words at some time step t
        # in all sentences, the input shape to this layer will be [batch_size, emb_dims]

        # input_size - embeddings dimensionality
        # hidden_size - the number of neurons in the hidden state 

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(
            data=torch.randn((input_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.W_hh = nn.Parameter(
            data=torch.randn((hidden_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.b_h = nn.Parameter(
            data=torch.zeros(hidden_size),
            requires_grad=True
        )

    def forward(self, X, state=None):

        # X shape: (num_steps, batch_size, emb_dims)

        if state is None:
            state = torch.zeros((X.shape[1], self.hidden_size), device=X.device)

        outputs = []

        for t in range(X.shape[0]):
            state = F.tanh(
                torch.matmul(X[t], self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h
            )
            outputs.append(state)

        return torch.stack(outputs), state