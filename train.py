import torch
import tiktoken

from gpt2 import GPT
from config import GPTConfig

# seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# constants
B = 2   # batch size
T = 4  # tokens per sample

# detect available devices
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    
device = "cpu" # override device due to out of memory. Running on 2GB GPU
    
print(f"Using device: {device}")

# load data and tokenize
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
    
text = text[:1000]
tokens = enc.encode(text)
buf = torch.tensor(tokens[:B*T + 1], device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# create model
model = GPT(GPTConfig())
model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for i in range(50):
    optimizer.zero_grad()
    # get logits and loss
    logits, loss = model(x, y)
    # backward and step
    loss.backward()
    optimizer.step()
    
    print(f"step {i}, loss: {loss.item()}")
    
