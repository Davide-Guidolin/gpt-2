import torch
import time

from gpt2 import GPT
from config import GPTConfig
from data import DataLoaderLite

# constants
B = 4   # batch size
T = 32  # tokens per sample

# seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# detect available devices
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    
device = "cpu" # override device due to out of memory. Running on 2GB GPU
    
print(f"Using device: {device}")

train_loader = DataLoaderLite(B, T)

# create model
model = GPT(GPTConfig())
model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # get logits and loss
    logits, loss = model(x, y)
    # backward and step
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in ms
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
    
