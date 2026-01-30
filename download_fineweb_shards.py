#!/usr/bin/env python3
"""
Download FineWeb sample-10B (sample-10BT) from Hugging Face and convert to .npy shards
compatible with FineWebDataLoader (training_shard_*.npy, validation_shard_*.npy).

Source: asquirous/fineweb_sample_10B_np_bin
- train.bin ~23.5 GB (uint16 token IDs)
- val.bin ~12 MB

Note: The source .bin files were tokenized with TinyLlama tokenizer. For GPT-2
training you may want to use tiktoken; in that case consider downloading
HuggingFaceFW/fineweb (sample-10BT parquet) and tokenizing locally with tiktoken.

Usage:
  python download_fineweb_shards.py [--output-dir DIR] [--train-shard-tokens N]
"""

import argparse
import os
import numpy as np
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("Install huggingface_hub: uv add huggingface_hub  # or pip install huggingface_hub")

REPO_ID = "asquirous/fineweb_sample_10B_np_bin"
REPO_TYPE = "dataset"
# .bin files are raw uint16 (one token id per 2 bytes)
BIN_DTYPE = np.uint16
# Default: ~500M tokens per training shard (~1 GB per shard) -> ~24 shards for ~11.7B tokens
DEFAULT_TRAIN_SHARD_TOKENS = 500_000_000


def main():
    p = argparse.ArgumentParser(description="Download FineWeb sample-10B and create .npy shards for FineWebDataLoader")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fineweb_sample_10B"),
        help="Directory to write training_shard_*.npy and validation_shard_*.npy",
    )
    p.add_argument(
        "--train-shard-tokens",
        type=int,
        default=DEFAULT_TRAIN_SHARD_TOKENS,
        help=f"Max tokens per training shard (default {DEFAULT_TRAIN_SHARD_TOKENS:,})",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face cache dir for downloads (default: HF_HOME or ~/.cache/huggingface)",
    )
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = str(args.cache_dir) if args.cache_dir else None

    # 1) Download train.bin and val.bin
    print("Downloading train.bin (~23.5 GB)...")
    train_bin = hf_hub_download(
        repo_id=REPO_ID,
        filename="train.bin",
        repo_type=REPO_TYPE,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
    )
    print("Downloading val.bin (~12 MB)...")
    val_bin = hf_hub_download(
        repo_id=REPO_ID,
        filename="val.bin",
        repo_type=REPO_TYPE,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
    )

    # 2) Convert train.bin -> training_shard_0.npy, training_shard_1.npy, ...
    print("Splitting train.bin into .npy shards...")
    train_data = np.memmap(train_bin, dtype=BIN_DTYPE, mode="r")
    n_train = len(train_data)
    n_shards = (n_train + args.train_shard_tokens - 1) // args.train_shard_tokens
    for i in range(n_shards):
        start = i * args.train_shard_tokens
        end = min(start + args.train_shard_tokens, n_train)
        shard = np.array(train_data[start:end], dtype=np.int64)  # dataloader uses .long()
        path = out / f"training_shard_{i}.npy"
        np.save(path, shard, allow_pickle=False)
        print(f"  {path.name}  {len(shard):,} tokens")
    del train_data

    # 3) Convert val.bin -> validation_shard_0.npy
    print("Writing validation_shard_0.npy...")
    val_data = np.memmap(val_bin, dtype=BIN_DTYPE, mode="r")
    val_arr = np.array(val_data, dtype=np.int64)
    np.save(out / "validation_shard_0.npy", val_arr, allow_pickle=False)
    print(f"  validation_shard_0.npy  {len(val_arr):,} tokens")
    del val_data, val_arr

    # Optional: remove .bin files to save disk (keep only .npy)
    # Uncomment if you want to free ~23.5 GB after conversion:
    # for f in (out / "train.bin", out / "val.bin"):
    #     if f.exists():
    #         f.unlink()
    #         print(f"Removed {f}")

    print(f"\nDone. Use with FineWebDataLoader: dataset_path={out!s}")
    print("  Example: FineWebDataLoader(B=4, T=1024, dataset_path=%r)" % str(out))


if __name__ == "__main__":
    main()
