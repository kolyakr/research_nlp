import torch
from torch import nn
from utils import get_device
from .RNN import RNN

class RNNLM(nn.Module):
    def __init__(self, rnn: RNN, vocab_size, embedding_dims=300):
        super().__init__()
        self.rnn = rnn
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dims)
        self.output = nn.Linear(in_features=rnn.hidden_size, out_features=vocab_size)

    def forward(self, X):
        X = self.vectorize(X)

        outputs, state = self.rnn(X)

    def vectorize(self, X):
        return self.emb(torch.tensor(X, device=get_device())).transpose(0, 1)
