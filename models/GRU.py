import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, sigma=0.01):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (
            init_weight(input_size, hidden_size),
            init_weight(hidden_size, hidden_size),
            nn.Parameter(torch.zeros(hidden_size)),
        )

        self.W_xr, self.W_hr, self.b_r = triple()
        self.W_xz, self.W_hz, self.b_z = triple()
        self.W_xh, self.W_hh, self.b_h = triple()

    def forward(self, X, state=None):

        if state is None:
            state = torch.zeros((X.shape[1], self.hidden_size), device=X.device)

        outputs = []

        for t in range(X.shape[0]):
            R = torch.sigmoid(
                torch.matmul(X[t], self.W_xr) + torch.matmul(state, self.W_hr) + self.b_r
            )

            Z = torch.sigmoid(
                torch.matmul(X[t], self.W_xz) + torch.matmul(state, self.W_hz) + self.b_z
            )

            H_hat = torch.tanh(
                torch.matmul(X[t], self.W_xh) + torch.matmul((R * state), self.W_hh) + self.b_h
            )

            state = Z * state + (1 - Z) * H_hat

            outputs.append(state)

        return torch.stack(outputs, dim=0), state   

