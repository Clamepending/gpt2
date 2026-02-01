"""Load checkpoint from wandb (online) and sample. Uses only the downloaded artifact, not local files."""
from pathlib import Path
import torch
from torch.nn import functional as F
import tiktoken
import wandb
from model import GPT2, GPT2config, load_model



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = load_model().to(device)
tokenizer = tiktoken.get_encoding("gpt2")

prompt, max_len = "Hello, I'm a language model,", 200

def sample_from_model_stream(model, prompt, max_len):
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    next_tok = None
    with torch.inference_mode():
        while x.size(1) < max_len and (next_tok is None or next_tok.item() != 50256):
            logits, _ = model(x)
            probs = F.softmax(logits[:, -1], dim=-1)
            top_p, top_idx = torch.topk(probs, 50)
            next_tok = top_idx.gather(-1, torch.multinomial(top_p, 1))
            x = torch.cat([x, next_tok], dim=1)
            print(tokenizer.decode([next_tok.item()]), end="", flush=True)


while True:
    user_input = input("\nEnter a prompt: ")
    if user_input == "exit":
        break
    print("\nmodel response:")
    sample_from_model_stream(model, user_input, max_len)

