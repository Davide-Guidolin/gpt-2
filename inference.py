import torch
import torch.nn.functional as F

from gpt2 import GPT
from config import GPTConfig
import tiktoken

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    
print(f"Using device: {device}")

prompt = "Hello, I'm a language model and"
num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig()) # GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # get logits (B, T, vocab_size)
        logits = logits[:, -1, :] # take logits at last position
        probs = F.softmax(logits, dim=-1) # get probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # get top 50 probabilities
        ix = torch.multinomial(topk_probs, 1) # sample a token from the top-k probabilities
        x_col = torch.gather(topk_indices, -1, ix) # gather the corresponding indices
        x = torch.cat((x, x_col), dim=1) # append to sequence

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)