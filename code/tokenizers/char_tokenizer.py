class CharTokenizer:
    def __init__(self):
        self.chars = None
        self.vocab_size = None
        self.stoi = None
        self.itos = None
        
    def train_tokenizer(self, texts):
        text = ' '.join(texts)
        self.chars = set(text)
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i, ch in enumerate(self.chars)}
        self.itos = {i:ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])