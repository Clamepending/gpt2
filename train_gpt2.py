from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from fineweb_dataloader import FineWebDataLoader
import wandb


# class DataloaderLite:
#     def __init__(self, B, T, ddp_rank, ddp_world_size):
#         #downlaod tiny shakespeare dataset from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#         import requests
#         url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#         response = requests.get(url)
#         text = response.text
#         self.text = torch.tensor(tokenizer.encode(text), dtype = torch.long)
#         self.current_pos = ddp_rank * B * T
#         self.B = B
#         self.T = T
#         self.ddp_world_size = ddp_world_size
        
#     def get_batch(self):
#         buf = self.text[self.current_pos:self.current_pos+self.B*self.T+1]
#         x = buf[:-1].view(self.B, self.T)
#         y = buf[1:].view(self.B, self.T)
        
#         self.current_pos += self.B*self.T * self.ddp_world_size
#         if self.current_pos + self.B*self.T*self.ddp_world_size >= len(self.text):
#             self.current_pos = 0
#         return x, y

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
class GPT2config:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    block_size: int = 1024
    
    
@dataclass
class TrainingConfig:
    total_batch_size: int = 524288
    B: int = 4
    T: int = 1024
    grad_accumulation_steps: int = total_batch_size // (B * T * ddp_world_size)
    max_steps: int = 100 # total number of training steps
    max_lr: float = 6e-4 # maximum learning rate for cosine schedule
    min_lr: float = max_lr * 0.1 # minimum learning rate for cosine schedule
    warmup_steps: int = 10 # number of warmup steps
    weight_decay: float = 0.1 # weight decay (no bias decay)
    sample_interval: int = 20 # interval to sample from the model
    num_samples_per_interval: int = 5 # number of samples to generate per interval
    sample_max_length: int = 50 # maximum length of the generated samples including prompt

train_config = TrainingConfig()
if not ddp or master_process:
    wandb.init(project="gpt2", config=train_config)
    assert train_config.total_batch_size % (train_config.B * train_config.T * ddp_world_size) == 0
    print(f"Total batch size: {train_config.total_batch_size}, Grad accumulation steps: {train_config.grad_accumulation_steps}")

    
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.c_proj.RESIDUAL_SCALE_FLAG = True
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # for batch, num_heads, bloxksize, blocksize
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2) # (B, T, 3 * C)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hc
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hc
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hc
        # attn_scores = (q@k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = attn_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # att = att @ v
        att = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        att = att.transpose(1,2).contiguous().view(B, T, C)
        att = self.c_proj(att)
        return att
        
        

class MLP(nn.Module):
    def __init__(self, config: GPT2config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.RESIDUAL_SCALE_FLAG = True
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config: GPT2config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
            
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)
        
        
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self.init_weights)
        
        
    def init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "RESIDUAL_SCALE_FLAG"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == "cuda"
        
        if not ddp or master_process:
            print(f"num_decay_params: {num_decay_params}, num_nodecay_params: {num_nodecay_params}")
            print(f"Using fused AdamW: {use_fused}")
        
        return torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        
    def forward(self, idx, targets = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos = self.transformer.wpe(pos) # T C
        tok_emb = self.transformer.wte(idx) # B T C
        x = tok_emb + pos # B T C
        for block in self.transformer.h:
            x = block(x) # B T C
        x = self.transformer.ln_f(x) # B T C
        x = self.lm_head(x) # B T V
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss
        

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
fineweb_dataset = FineWebDataLoader(B=4, T=1024, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size)



torch.set_float32_matmul_precision("high")

model = GPT2(GPT2config(vocab_size = 50304)).to(device)
model = torch.compile(model)

from torch.nn.parallel import DistributedDataParallel as DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model



import time
losses = torch.zeros(train_config.max_steps, device = device)
perplexity = torch.zeros(train_config.max_steps, device = device)
tokens_processed = 0

optimizer = raw_model.configure_optimizers(weight_decay = train_config.weight_decay, learning_rate = train_config.max_lr, device = device)
for step in range(train_config.max_steps):
    start_time = time.time()
    optimizer.zero_grad()
    loss_accumulated = 0
    for micro_step in range(train_config.grad_accumulation_steps):
        x, y = fineweb_dataset.get_batch()
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
    perplexity[step] = torch.exp(losses[step])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param in optimizer.param_groups:
        param["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    if not ddp or master_process:
        tokens_processed += fineweb_dataset.B*fineweb_dataset.T*train_config.grad_accumulation_steps*ddp_world_size
        print(f"Step {step}: loss {loss_accumulated.item():.6f}, lr {lr:.6f}, time {end_time - start_time}, norm {norm: .4f}, tokens per second {fineweb_dataset.B*fineweb_dataset.T*train_config.grad_accumulation_steps*ddp_world_size/(end_time - start_time):.2f}, perplexity {perplexity[step]:.2f}")
        wandb.log({"train/loss": losses[step],
                   "train/lr": lr,
                   "train/norm": norm,
                   "train/tokens_per_second": fineweb_dataset.B*fineweb_dataset.T*train_config.grad_accumulation_steps*ddp_world_size/(end_time - start_time),
                   "train/perplexity": perplexity[step],
                   "train/tokens_processed": tokens_processed},
                    step=step)
        if step % train_config.sample_interval == 0:
            responses = sample_from_model(model, "Hello, I'm a language model,", train_config.num_samples_per_interval, train_config.sample_max_length)
            table_data = [(step, response) for response in responses]
            table = wandb.Table(data=table_data, columns=["step", "sample"])
            wandb.log({"samples": table}, step=step)
            # sample from the model

if not ddp or master_process:
    wandb.finish()

# print samples to terminal
if not ddp or master_process:
    decoded = sample_from_model(model, "Hello, I'm a language model,", 5, 30)
    for d in decoded:
        print(d)

if ddp:
    destroy_process_group()
