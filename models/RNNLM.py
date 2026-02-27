import torch
from torch import nn
from utils import get_device
from .RNN import RNN
import torch.nn.functional as F

class RNNLM(nn.Module):
    def __init__(self, rnn: RNN, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn = rnn
        self.output = nn.Linear(in_features=rnn.hidden_size, out_features=vocab_size)

    def forward(self, X, state=None):
        X = self.vectorize(X)

        states, last_state = self.rnn(X, state)

        logits = self.output(states).transpose(0, 1)

        return logits, last_state

    def vectorize(self, X):
        return F.one_hot(X, num_classes=self.vocab_size).transpose(0, 1).float()
    
    def predict(self, prefix, num_preds, vocab, device):
        state = None
        actual_sequence = []

        for i in range(len(prefix)):
            char_to_feed = prefix[i] 
            
            X = torch.tensor([[vocab[char_to_feed]]], device=device)
            logits, state = self.forward(X, state)
            
            actual_sequence.append(char_to_feed)

        for _ in range(num_preds):
            last_char_idx = vocab[actual_sequence[-1]]
            X = torch.tensor([[last_char_idx]], device=device)
            
            logits, state = self.forward(X, state)
            
            idx = torch.argmax(logits, dim=2).item()
            actual_sequence.append(vocab[idx])

        return "".join(actual_sequence)



