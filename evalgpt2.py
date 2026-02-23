import json
import os
from urllib.request import urlretrieve

import tiktoken
import torch
from torch.nn import functional as F


HELLASWAG_VAL_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
HELLASWAG_VAL_PATH = os.path.join(os.path.dirname(__file__), "eval", "hellaswag", "hellaswag_val.jsonl")
ENC = tiktoken.get_encoding("gpt2")


def _ensure_val_file() -> str:
    os.makedirs(os.path.dirname(HELLASWAG_VAL_PATH), exist_ok=True)
    if not os.path.exists(HELLASWAG_VAL_PATH):
        urlretrieve(HELLASWAG_VAL_URL, HELLASWAG_VAL_PATH)
    return HELLASWAG_VAL_PATH


def _render(example: dict):
    ctx = ENC.encode(example["ctx"])
    label = int(example["label"])

    rows = [ctx + ENC.encode(" " + end) for end in example["endings"]]
    masks = [[0] * len(ctx) + [1] * (len(r) - len(ctx)) for r in rows]

    n = max(len(r) for r in rows)
    tokens = torch.zeros((4, n), dtype=torch.long)
    mask = torch.zeros((4, n), dtype=torch.long)
    for i in range(4):
        tokens[i, :len(rows[i])] = torch.tensor(rows[i], dtype=torch.long)
        mask[i, :len(masks[i])] = torch.tensor(masks[i], dtype=torch.long)
    return tokens, mask, label


@torch.no_grad()
def get_hellaswag_estimates(model, batch_size=8, device=None, limit=100, ddp_rank=0, ddp_world_size=1):
    del batch_size  # compatibility with existing call-site
    if device is None:
        device = next(model.parameters()).device

    eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    was_training = eval_model.training
    eval_model.to(device).eval()

    correct = 0
    correct_norm = 0
    total = 0

    with open(_ensure_val_file(), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if i % ddp_world_size != ddp_rank:
                continue
            tokens, mask, label = _render(json.loads(line))
            tokens = tokens.to(device)
            mask = mask.to(device)

            logits, _ = eval_model(tokens)  # [4, T, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tokens = tokens[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()

            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_tokens.reshape(-1),
                reduction="none",
            ).view(4, -1)

            sum_loss = (losses * shift_mask).sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            pred = int(sum_loss.argmin().item())
            pred_norm = int(avg_loss.argmin().item())

            correct += int(pred == label)
            correct_norm += int(pred_norm == label)
            total += 1

    if ddp_world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor([correct, correct_norm, total], dtype=torch.long, device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        correct, correct_norm, total = [int(x) for x in stats.tolist()]

    if was_training:
        eval_model.train()
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    return correct / total, correct_norm / total, 0.0, 0.0