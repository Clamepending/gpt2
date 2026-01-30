from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from fineweb_dataloader import FineWebDataLoader
import wandb
from pathlib import Path
from model import GPT2, GPT2config

# DDP
from torch.distributed import init_process_group, destroy_process_group

import os
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    
    master_process = ddp_rank == 0
    
    torch.cuda.set_device(f'cuda:{ddp_local_rank}')
    device = torch.device(f'cuda:{ddp_local_rank}')
    init_process_group(backend="nccl")
else:
    ddp_rank = 0
    ddp_local_rank =0 
    ddp_world_size = 1
    master_process = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
    
@dataclass
class TrainingConfig:
    total_batch_size: int = 524288
    B: int = 32 # 4
    T: int = 2048 # 1024
    grad_accumulation_steps: int = total_batch_size // (B * T * ddp_world_size)
    max_steps: int = 10_000 # total number of training steps
    max_lr: float = 2e-3 # maximum learning rate for cosine schedule (original 6e-4)
    min_lr: float = max_lr * 0.1 # minimum learning rate for cosine schedule
    warmup_steps: int = 10 # number of warmup steps
    weight_decay: float = 0.1 # weight decay (no bias decay)
    sample_interval: int = 20 # interval to sample from the model
    num_samples_per_interval: int = 5 # number of samples to generate per interval
    sample_max_length: int = 50 # maximum length of the generated samples including prompt
    save_interval: int = 2000 # interval to save the model checkpoint
    resume_path: str | None = None  # path to checkpoint .pt to resume from (e.g. checkpoints/model_step_2000.pt)

train_config = TrainingConfig()
if not ddp or master_process:
    wandb.init(project="gpt2", config=train_config)
    assert train_config.total_batch_size % (train_config.B * train_config.T * ddp_world_size) == 0
    print(f"Total batch size: {train_config.total_batch_size}, Grad accumulation steps: {train_config.grad_accumulation_steps}")

    
    

# --------------------------------------------------

def sample_from_model(model, prompt, num_return_sequences, max_length) -> list[str]:
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype = torch.long)
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
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

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


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict | None]:
    """Load checkpoint; returns (step to resume from, rng_states or None)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    rng = ckpt.get("rng")
    if rng is not None:
        if "cpu" in rng:
            torch.set_rng_state(rng["cpu"])
        if "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(ckpt["step"]), rng


def log_checkpoint_artifact(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng_states: dict | None = None,
) -> None:
    # Save full checkpoint to project-level checkpoints/
    checkpoint_dir = Path.cwd() / "checkpoints"
    checkpoint_path = save_checkpoint(model, optimizer, step, checkpoint_dir, rng_states)

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-step-{step}",
            type="model",
            metadata={"step": step},
        )
        artifact.add_file(checkpoint_path.as_posix(), name=checkpoint_path.name)
        wandb.log_artifact(artifact)

def get_lr(it):
    if it < train_config.warmup_steps:
        return train_config.max_lr * (it + 1)/train_config.warmup_steps
    if it > train_config.max_steps:
        return train_config.min_lr
    
    decay_ratio = (it - train_config.warmup_steps) / (train_config.max_steps - train_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.max_lr - train_config.min_lr)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")


# shakespeare_dataset = DataloaderLite(4, 1024, ddp_rank, ddp_world_size)
train_dataloader = FineWebDataLoader(B=train_config.B, T=train_config.T, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size, split="training")
if not ddp or master_process:
    validation_dataloader = FineWebDataLoader(B=train_config.B, T=train_config.T, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size, split="validation")


torch.set_float32_matmul_precision("high")

model = GPT2(GPT2config(vocab_size = 50304)).to(device)
model = torch.compile(model)

from torch.nn.parallel import DistributedDataParallel as DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model



import time
losses = torch.zeros(train_config.max_steps, device = device)
# perplexity = torch.zeros(train_config.max_steps, device = device)
tokens_processed = 0

optimizer = raw_model.configure_optimizers(weight_decay = train_config.weight_decay, learning_rate = train_config.max_lr, device = device)

# Resume from checkpoint if requested
start_step = 0
if train_config.resume_path:
    resume_path = Path(train_config.resume_path)
    if resume_path.exists():
        start_step, _ = load_checkpoint(resume_path, model, optimizer, device)
        start_step += 1  # next step to run
        if not ddp or master_process:
            print(f"Resumed from {resume_path}, starting at step {start_step}")
    else:
        raise FileNotFoundError(f"Resume path not found: {resume_path}")

for step in range(start_step, train_config.max_steps):
    start_time = time.time()
    optimizer.zero_grad()
    loss_accumulated = 0
    for micro_step in range(train_config.grad_accumulation_steps):
        x, y = train_dataloader.get_batch()
        x = x.to(device)
        y = y.to(device)
        if ddp and micro_step != train_config.grad_accumulation_steps - 1: # don't sync gradients until the last micro_step
            with model.no_sync():
                with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / train_config.grad_accumulation_steps
                loss.backward()
        else:
            with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / train_config.grad_accumulation_steps
            loss.backward()
        loss_accumulated += loss.detach()
    
    if ddp:
        torch.distributed.all_reduce(loss_accumulated, op = torch.distributed.ReduceOp.AVG) # average the loss across all devices
    losses[step] = loss_accumulated.item()
    # perplexity[step] = torch.exp(losses[step])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param in optimizer.param_groups:
        param["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    if not ddp or master_process:
        tokens_processed += train_config.B*train_config.T*train_config.grad_accumulation_steps*ddp_world_size
        print(f"Step {step}: loss {loss_accumulated.item():.6f}, lr {lr:.6f}, time {end_time - start_time}, norm {norm: .4f}, tokens per second {train_config.B*train_config.T*train_config.grad_accumulation_steps*ddp_world_size/(end_time - start_time):.2f}")
        wandb.log({"train/loss": losses[step],
                   "train/lr": lr,
                   "train/norm": norm,
                   "train/tokens_per_second": train_config.B*train_config.T*train_config.grad_accumulation_steps*ddp_world_size/(end_time - start_time),
                #    "train/perplexity": perplexity[step],
                   "train/tokens_processed": tokens_processed},
                    step=step)
        if step % train_config.sample_interval == 0:
            with torch.inference_mode():
                x, y = validation_dataloader.get_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
                    logits, loss = raw_model(x, y) # deadlocks if use ddp model here because only master process validates
                wandb.log({"validation/loss": loss.item()}, step=step)
                        # "validation/perplexity": torch.exp(loss)}, step=step)
            
                responses = sample_from_model(model, "Hello, I'm a language model,", train_config.num_samples_per_interval, train_config.sample_max_length)
                table_data = [(step, response) for response in responses]
                table = wandb.Table(data=table_data, columns=["step", "sample"])
                wandb.log({"samples": table}, step=step)
                
        # save checkpoint (only on master when using DDP)
        if step > 0 and step % train_config.save_interval == 0:
            rng_states = {
                "cpu": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                rng_states["cuda"] = torch.cuda.get_rng_state_all()
            log_checkpoint_artifact(model, optimizer, step, rng_states)

if not ddp or master_process:
    wandb.finish()

# print samples to terminal
if not ddp or master_process:
    decoded = sample_from_model(model, "Hello, I'm a language model,", 5, 30)
    for d in decoded:
        print(d)

if ddp:
    destroy_process_group()
