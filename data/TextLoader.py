import re
from .Vocabulary import Vocabulary
from .Tokenizer import Tokenizer
import torch
import torch.nn.functional as F

class TextLoader():
    def __init__(self, filename):
        self.raw_text = self._read_file(filename)
        tokenizer = Tokenizer()
        self.tokens = tokenizer.split_by_word(self._preprocess(self.raw_text))
        self.vocab = Vocabulary(self.tokens)
        self.vocab_size = len(self.vocab.stats)

    def _read_file(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
        
    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()
    
    # READ DATA SPLIT INTO BATCHES!!
    