"""Load checkpoint from wandb (online) and sample. Uses only the downloaded artifact, not local files."""
from pathlib import Path
import torch
from torch.nn import functional as F
import tiktoken
import wandb
from model import GPT2, GPT2config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
config = GPT2config(vocab_size=50304)
model = GPT2(config).to(device)

# Download from wandb (online) only; do not use local checkpoints/
artifact = wandb.Api().artifact("gpt2/model-checkpoint-step-90:v0")
root = artifact.download(root=".")  # returns path to downloaded contents
state = torch.load(Path(root) / "model_step_90.pt", map_location=device, weights_only=True)
# Checkpoint was saved from torch.compile() model; strip "module._orig_mod." prefix
prefix = "module._orig_mod."
state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
model.load_state_dict(state, strict=True)

prompt, max_len = "Hello, I'm a language model,", 50
x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
with torch.inference_mode():
    while x.size(1) < max_len:
        logits, _ = model(x)
        probs = F.softmax(logits[:, -1], dim=-1)
        top_p, top_idx = torch.topk(probs, 50)
        next_tok = top_idx.gather(-1, torch.multinomial(top_p, 1))
        x = torch.cat([x, next_tok], dim=1)
print(tokenizer.decode(x[0].tolist()))

