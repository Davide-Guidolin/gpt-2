import torch
import time
import math

from gpt2 import GPT
from config import GPTConfig
from data import DataLoaderLite

# constants
B = 2   # batch size
T = 256  # tokens per sample
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

# cosine decaying lr with linear warmup
def get_lr(it):
    # linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (max_lr - min_lr)

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
# optimizer = torch.optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

model.to(torch.bfloat16)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # get logits and loss
        logits, loss = model(x, y)
    # backward and step
    loss.backward()
    # clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine lr
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in ms
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | grad_norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
