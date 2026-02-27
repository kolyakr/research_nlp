import re
from .Vocabulary import Vocabulary
from .Tokenizer import Tokenizer
import torch
import torch.nn.functional as F
import random

class TextLoader():
    def __init__(self, filename, batch_size, num_steps):
        self.raw_text = self._read_file(filename)
        tokenizer = Tokenizer()
        self.tokens = tokenizer.split_by_character(self._preprocess(self.raw_text))
        self.vocab = Vocabulary(self.tokens)
        self.corpus = self.vocab[self.tokens]
        self.vocab_size = len(self.vocab.stats)
        self.batch_size = batch_size
        self.num_steps = num_steps

    def _read_file(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
        
    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()
    
    def get_batches(self, train=True):
        split_index = int(len(self.corpus) * 0.9)

        if train:
            source_corpus = self.corpus[:split_index]
            d = random.randint(0, self.num_steps)
        else:
            source_corpus = self.corpus[split_index:]
            d = 0

        sliced_corpus = source_corpus[d:]

        batch = []
        for i in range(0, len(sliced_corpus) - self.num_steps):
            
            x = sliced_corpus[i : i + self.num_steps]
            y = sliced_corpus[i + 1 : i + self.num_steps + 1]
            
            batch.append((x, y))
            
            if len(batch) == self.batch_size:
                yield ( 
                    [item[0] for item in batch], 
                    [item[1] for item in batch] 
                )
                batch = [] 