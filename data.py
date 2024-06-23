import tiktoken
import torch

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # load data and tokenize
        with open('input.txt', 'r') as f:
            text = f.read()
            
        # get tokenizer
        enc = tiktoken.get_encoding('gpt2')
        # tokenize
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        
        # state of the dataloader
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        # get tokens for one batch
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T)  # targets
        
        self.current_position += B * T
        
        if self.current_position + (B * T + 1) >= len(self.tokens):
            self.current_position = 0
            
        return x, y