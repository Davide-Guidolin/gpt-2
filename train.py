import torch

from gpt2 import GPT
from config import GPTConfig
from data import DataLoaderLite

# seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# constants
B = 4   # batch size
T = 32  # tokens per sample

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
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # get logits and loss
    logits, loss = model(x, y)
    # backward and step
    loss.backward()
    optimizer.step()
    
    print(f"step {i}, loss: {loss.item()}")
    
