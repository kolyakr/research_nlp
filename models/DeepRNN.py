import torch
from torch import nn
from .RNN import RNN

class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sigma=0.01):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnns = nn.ModuleList([
            RNN(input_size if i == 0 else hidden_size, hidden_size, sigma=sigma)
            for i in range(num_layers)
        ])
        
    def forward(self, X, state=None):

        if state is None:
            state = [torch.zeros((X.shape[1], self.hidden_size), device=X.device) 
                     for _ in range(self.num_layers)]
        
        curr_input = X
        new_states = []

        for l in range(self.num_layers):
            
            outputs, updated_state = self.rnns[l](curr_input, state[l])
            
            curr_input = outputs
            new_states.append(updated_state)

        return outputs, new_states