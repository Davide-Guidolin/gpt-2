import torch
import time

from gpt2 import GPT
from config import GPTConfig
from data import DataLoaderLite

# constants
B = 4   # batch size
T = 128  # tokens per sample

# seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# detect available devices
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    
print(f"Using device: {device}")

train_loader = DataLoaderLite(B, T)

# use tf32 if available
torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
if device == "cuda":
    cuda_cap = torch.cuda.get_device_capability()
    if cuda_cap[0] >= 7:
        model = torch.compile(model)
    else:
        print(f"Cannot compile the model. Cuda capability {cuda_cap[0]}.{cuda_cap[1]} < 7.0")

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

model.to(torch.bfloat16)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
    
