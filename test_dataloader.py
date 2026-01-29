"""Test script: load first 5 training examples from FineWebDataLoader and decode with GPT-2 tokenizer."""

import tiktoken
from fineweb_dataloader import FineWebDataLoader

# Use same tokenizer as train_gpt2
tokenizer = tiktoken.get_encoding("gpt2")

# Small batch: 5 examples, short sequence length for readable output
B, T = 5, 256
dataloader = FineWebDataLoader(B=B, T=T, ddp_rank=0, ddp_world_size=1)

# Get first batch: x (input), y (targets, i.e. next token per position)
x, y = dataloader.get_batch()

# x, y are (B, T) tensors of token ids
print(f"Batch shape: x {x.shape}, y {y.shape}\n")
print("=" * 80)

for i in range(B):
    input_ids = x[i].tolist()
    target_ids = y[i].tolist()

    input_text = tokenizer.decode(input_ids)
    target_text = tokenizer.decode(target_ids)

    print(f"--- Example {i + 1} ---")
    print("Input (x):")
    print(repr(input_text))
    print()
    print("Target (y, next-token):")
    print(repr(target_text))
    print()
    # For causal LM, y is x shifted by 1; show first/last few tokens to verify
    print(f"  First 5 token ids: x={input_ids[:5]}, y={target_ids[:5]}")
    print("=" * 80)
