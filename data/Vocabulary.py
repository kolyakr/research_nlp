import collections

class Vocabulary():
    def __init__(self, text):
        
        self.stats = collections.Counter(text)
        self.idx_to_token = [item for item, _ in self.stats.items()]
        self.token_to_idx = {item:i for i, item in enumerate(self.idx_to_token)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx_to_token[key]
            
        elif isinstance(key, str):
            return self.token_to_idx[key]
            
        elif isinstance(key, (list, tuple)):
            return [self[item] for item in key]
            
        raise ValueError(f"Key must be an int, str, list, or tuple. Got {type(key)}")