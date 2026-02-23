import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import tiktoken
import wandb

from fineweb_dataloader import FineWebDataLoader
from model import GPT2, GPT2config
from evalgpt2 import get_hellaswag_estimates

# DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = ddp_rank == 0

        torch.cuda.set_device(f"cuda:{ddp_local_rank}")
        device = torch.device(f"cuda:{ddp_local_rank}")
        init_process_group(backend="nccl")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def sample_from_model(model, prompt, num_return_sequences, max_length, tokenizer, device) -> list[str]:
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    with torch.inference_mode():
        while x.size(1) < max_length:
            logits, _ = raw_model(x) # B T V - use raw_model for inference to avoid DDP collectives
            logits = logits[:,-1,:] # B V
            
            probs = F.softmax(logits, dim=-1) # B V
            
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # B 50
            
            ix = torch.multinomial(topk_probs, 1) # B 1
            
            x_next = torch.gather(topk_indices, 1, ix) # B 1
            
            x = torch.cat((x, x_next), dim=1) # B T+1
            
    return tokenizer.decode_batch(x.tolist())

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: Path | None = None,
    rng_states: dict | None = None,
) -> Path:
    """Save full training state (model, optimizer, step, RNG) for resumption."""
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model.pt"

    raw = model.module if hasattr(model, "module") else model
    state = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if rng_states is not None:
        state["rng"] = rng_states

    torch.save(state, checkpoint_path)
    return checkpoint_path


def log_checkpoint_artifact(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, checkpoint_dir: Path, rng_states: dict | None = None) -> None:
    checkpoint_path = save_checkpoint(model, optimizer, step, checkpoint_dir, rng_states)

    if wandb.run is not None:
        artifact = wandb.Artifact(name="model", type="model", metadata={"step": step})
        artifact.add_file(checkpoint_path.as_posix(), name=checkpoint_path.name)
        wandb.log_artifact(artifact)


def get_lr(it, cfg_train):
    if it < cfg_train.warmup_steps:
        return cfg_train.max_lr * (it + 1) / cfg_train.warmup_steps
    if it > cfg_train.max_steps:
        return cfg_train.min_lr

    decay_ratio = (it - cfg_train.warmup_steps) / (cfg_train.max_steps - cfg_train.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg_train.min_lr + coeff * (cfg_train.max_lr - cfg_train.min_lr)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_distributed()

    try:
        tokenizer = tiktoken.get_encoding("gpt2")

        torch.manual_seed(int(cfg.runtime.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(cfg.runtime.seed))

        grad_accumulation_steps = cfg.train.total_batch_size // (cfg.train.B * cfg.train.T * ddp_world_size)
        assert cfg.train.total_batch_size % (cfg.train.B * cfg.train.T * ddp_world_size) == 0

        if master_process and cfg.runtime.use_wandb:
            wandb.init(
                project=cfg.runtime.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.experiment_name,
            )
            print(f"Total batch size: {cfg.train.total_batch_size}, Grad accumulation steps: {grad_accumulation_steps}")

        train_dataloader = FineWebDataLoader(
            B=cfg.train.B,
            T=cfg.train.T,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            split="training",
            dataset_path=cfg.data.dataset_path,
            buffer_size=cfg.data.buffer_size,
        )
        if master_process:
            validation_dataloader = FineWebDataLoader(
                B=cfg.train.B,
                T=cfg.train.T,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                split="validation",
                dataset_path=cfg.data.dataset_path,
                buffer_size=cfg.data.buffer_size,
            )

        torch.set_float32_matmul_precision("high")

        model_cfg = GPT2config(
            vocab_size=cfg.model.vocab_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embed=cfg.model.n_embed,
            block_size=cfg.train.T,
            ffn_type=cfg.model.ffn_type,
        )
        model = GPT2(model_cfg).to(device)
        if cfg.runtime.compile:
            model = torch.compile(model)

        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module if ddp else model

        tokens_processed = 0
        checkpoint_dir = Path(cfg.runtime.checkpoint_dir)
        optimizer = raw_model.configure_optimizers(
            weight_decay=cfg.train.weight_decay,
            learning_rate=cfg.train.max_lr,
            device=device,
        )

        for step in range(1, cfg.train.max_steps + 1):
            start_time = time.time()
            optimizer.zero_grad()
            loss_accumulated = 0
            for micro_step in range(grad_accumulation_steps):
                x, y = train_dataloader.get_batch()
                x = x.to(device)
                y = y.to(device)
                if ddp and micro_step != grad_accumulation_steps - 1:
                    with model.no_sync():
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            _, loss = model(x, y)
                        loss = loss / grad_accumulation_steps
                        loss.backward()
                else:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss = loss / grad_accumulation_steps
                    loss.backward()
                loss_accumulated += loss.detach()

            if ddp:
                torch.distributed.all_reduce(loss_accumulated, op=torch.distributed.ReduceOp.AVG)
            loss = loss_accumulated.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, cfg.train)
            for param in optimizer.param_groups:
                param["lr"] = lr
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            if master_process:
                tokens_processed += cfg.train.B * cfg.train.T * grad_accumulation_steps * ddp_world_size
                tokens_per_second = cfg.train.B * cfg.train.T * grad_accumulation_steps * ddp_world_size / (end_time - start_time)
                print(
                    f"Step {step}: loss {loss:.6f}, lr {lr:.6f}, time {end_time - start_time}, "
                    f"norm {norm: .4f}, tokens per second {tokens_per_second:.2f}"
                )
                if cfg.runtime.use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/lr": lr,
                            "train/norm": norm,
                            "train/tokens_per_second": tokens_per_second,
                            "train/tokens_processed": tokens_processed,
                            "train/current_shard_idx": train_dataloader.current_shard_local_idx,
                            "train/num_shards": len(train_dataloader.shard_files),
                            "train/current_shard_progress": train_dataloader.get_current_shard_progress(),
                        },
                        step=step,
                    )

                if step % cfg.train.sample_interval == 0:
                    with torch.inference_mode():
                        x, y = validation_dataloader.get_batch()
                        x = x.to(device)
                        y = y.to(device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            _, val_loss = raw_model(x, y)
                        if cfg.runtime.use_wandb:
                            wandb.log({"validation/loss": val_loss.item()}, step=step)

                        responses = sample_from_model(
                            raw_model,
                            cfg.train.sample_prompt,
                            cfg.train.num_samples_per_interval,
                            cfg.train.sample_max_length,
                            tokenizer,
                            device,
                        )
                        if cfg.runtime.use_wandb:
                            table_data = [(step, response) for response in responses]
                            table = wandb.Table(data=table_data, columns=["step", "sample"])
                            wandb.log({"samples": table}, step=step)

                if step % cfg.train.save_interval == 0:
                    rng_states = {
                        "cpu": torch.get_rng_state(),
                    }
                    if torch.cuda.is_available():
                        rng_states["cuda"] = torch.cuda.get_rng_state_all()
                    if cfg.runtime.use_wandb:
                        log_checkpoint_artifact(model, optimizer, step, checkpoint_dir, rng_states)
                    else:
                        save_checkpoint(model, optimizer, step, checkpoint_dir, rng_states)

            if step % cfg.train.hellaswag_eval_interval == 0 or step == 1:
                acc, acc_norm, acc_stderr, acc_norm_stderr = get_hellaswag_estimates(
                    raw_model,
                    batch_size=cfg.train.B,
                    device=device,
                    limit=cfg.train.hellaswag_eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                if master_process and cfg.runtime.use_wandb:
                    wandb.log(
                        {
                            "validation/hellaswag/acc": acc,
                            "validation/hellaswag/acc_norm": acc_norm,
                            "validation/hellaswag/acc_stderr": acc_stderr,
                            "validation/hellaswag/acc_norm_stderr": acc_norm_stderr,
                        },
                        step=step,
                    )

        if master_process and cfg.runtime.use_wandb:
            wandb.finish()
    finally:
        if ddp:
            destroy_process_group()


if __name__ == "__main__":
    main()
