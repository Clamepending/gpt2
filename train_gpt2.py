from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPT2config:
    vocab_size = 50257
    n_layer = 12
    n_head = 12
    n_embed = 768
    block_size = 1024
    
    
    
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # for batch, num_heads, bloxksize, blocksize
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2) # (B, T, 3 * C)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hc
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hc
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hc
        attn_scores = (q@k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        att = attn_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att @ v
        att = att.transpose(1,2).contiguous().view(B, T, C)
        att = self.c_proj(att)
        return att
        
        

class MLP(nn.Module):
    def __init__(self, config: GPT2config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(4 * config.n_embed, config.n_embed)
    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
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
        
        
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos = self.transformer.wpe(pos) # T C
        tok_emb = self.transformer.wte(idx) # B T C
        x = tok_emb + pos # B T C
        for block in self.transformer.h:
            x = block(x) # B T C
        x = self.transformer.ln_f(x) # B T C
        x = self.lm_head(x) # B T V
        return x
        

# --------------------------------------------------

num_return_sequences = 5
max_length = 30
device = "mps"


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

tokens = tokenizer.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

model = GPT2(GPT2config()).to(device)
model = model.eval()

torch.manual_seed(42)

with torch.no_grad():
    while x.size(1) < max_length:
        logits = model(x) # B T V
        logits = logits[:,-1,:] # B V
        
        probs = F.softmax(logits, dim=-1) # B V
        
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # B 50
        
        ix = torch.multinomial(topk_probs, 1) # B 1
        
        x_next = torch.gather(topk_indices, 1, ix) # B 1
        
        x = torch.cat((x, x_next), dim=1) # B T+1
        
decoded = tokenizer.decode_batch(x.tolist())
for d in decoded:
    print(d)
        
        
        


