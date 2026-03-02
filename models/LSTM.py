import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sigma=0.01):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_xf = nn.Parameter(
            data=torch.randn((input_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.W_hf = nn.Parameter(
            data=torch.randn((hidden_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.b_f = nn.Parameter(torch.ones((hidden_size)), requires_grad=True)

        self.W_xi = nn.Parameter(
            data=torch.randn((input_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.W_hi = nn.Parameter(
            data=torch.randn((hidden_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.b_i = nn.Parameter(torch.zeros((hidden_size)), requires_grad=True)
        
        self.W_xo = nn.Parameter(
            data=torch.randn((input_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.W_ho = nn.Parameter(
            data=torch.randn((hidden_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.b_o = nn.Parameter(torch.zeros((hidden_size)), requires_grad=True)

        self.W_xc = nn.Parameter(
            data=torch.randn((input_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.W_hc = nn.Parameter(
            data=torch.randn((hidden_size, hidden_size)) * sigma,
            requires_grad=True
        )

        self.b_c = nn.Parameter(torch.zeros((hidden_size)), requires_grad=True)

    def forward(self, X, state=None, C=None):
        
        # X shape: (num_steps, batch_size, emb_dims)

        if state is None:
            state = torch.zeros((X.shape[1], self.hidden_size), device=X.device)

        if C is None:
            C = torch.zeros((X.shape[1], self.hidden_size), device=X.device)

        outputs = []

        for t in range(X.shape[0]):
            F = torch.sigmoid(
                torch.matmul(X[t], self.W_xf) + torch.matmul(state, self.W_hf) + self.b_f
            )

            I = torch.sigmoid(
                torch.matmul(X[t], self.W_xi) + torch.matmul(state, self.W_hi) + self.b_i
            )

            O = torch.sigmoid(
                torch.matmul(X[t], self.W_xo) + torch.matmul(state, self.W_ho) + self.b_o
            )

            C_hat = torch.tanh(
                torch.matmul(X[t], self.W_xc) + torch.matmul(state, self.W_hc) + self.b_c
            )

            C = F * C + I * C_hat
            state = O * torch.tanh(C)

            outputs.append(state)
        
        outputs = torch.stack(outputs, dim=0) 
        
        return outputs, (state, C)