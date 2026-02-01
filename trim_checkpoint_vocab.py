#!/usr/bin/env python3
"""
Trim checkpoint vocab from 50304 to 50257 by surgically slicing
transformer.wte.weight and lm_head.weight. Saves a new .pt file (does not overwrite).
50257 = standard GPT-2 vocab size (tiktoken "gpt2"; max token id 50256).
"""
import argparse
from pathlib import Path

import torch

TARGET_VOCAB = 50257
KEYS_TO_TRIM = ("transformer.wte.weight", "lm_head.weight")


def trim_checkpoint(
    input_path: Path,
    output_path: Path | None = None,
    target_vocab: int = TARGET_VOCAB,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.parent / (
        input_path.stem + f"_vocab{target_vocab}" + input_path.suffix
    )

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f'Checkpoint has no "model" key. Top-level keys: {list(ckpt.keys())}')

    model_state = ckpt["model"]
    for key in list(model_state.keys()):
        for suffix in KEYS_TO_TRIM:
            if key.endswith(suffix) or key == suffix:
                t = model_state[key]
                if t.shape[0] > target_vocab:
                    model_state[key] = t[:target_vocab].clone()
                    print(f"  {key}: {t.shape} -> {model_state[key].shape}")
                break

    torch.save(ckpt, output_path)
    print(f"Saved trimmed checkpoint to {output_path}")
    print("Replace the original with this file, or point load_model() at it.")


def main() -> None:
    p = argparse.ArgumentParser(description="Trim checkpoint vocab to 50257 (standard GPT-2)")
    p.add_argument("input", nargs="?", default="checkpoint_step_18000.pt", help="Input .pt path")
    p.add_argument("--vocab-size", type=int, default=TARGET_VOCAB, help="Target vocab size (default 50257)")
    p.add_argument("-o", "--output", help="Output .pt path (default: input_stem_vocab50257.pt)")
    args = p.parse_args()
    trim_checkpoint(args.input, args.output, target_vocab=args.vocab_size)


if __name__ == "__main__":
    main()
