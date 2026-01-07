from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math





class DataloaderLite:
    def __init__(self, B, T):
        #downlaod tiny shakespeare dataset from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        import requests
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        text = response.text
        self.text = torch.tensor(tokenizer.encode(text), dtype = torch.long)
        self.current_pos = 0
        self.B = B
        self.T = T
        
    def get_batch(self):
        buf = self.text[self.current_pos:self.current_pos+self.B*self.T+1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        self.current_pos += self.B*self.T
        if self.current_pos + self.B*self.T >= len(self.text):
            self.current_pos = 0
        return x, y
        
        


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
        self.c_proj.RESIDUAL_SCALE_FLAG = True
        
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


B, T = 4, 32
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

shakespeare_dataset = DataloaderLite(B, T)


optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
for i in range(50):
    optimizer.zero_grad()
    x, y = shakespeare_dataset.get_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)
    loss.item()
    loss.backward()
    optimizer.step()
    print(f"Step {i}: loss {loss.item()}")
    






# with torch.no_grad():
#     while x.size(1) < max_length:
#         logits = model(x) # B T V
#         logits = logits[:,-1,:] # B V
        
#         probs = F.softmax(logits, dim=-1) # B V
        
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # B 50
        
#         ix = torch.multinomial(topk_probs, 1) # B 1
        
#         x_next = torch.gather(topk_indices, 1, ix) # B 1
        
#         x = torch.cat((x, x_next), dim=1) # B T+1
        
# decoded = tokenizer.decode_batch(x.tolist())
# for d in decoded:
#     print(d)
        
        
        


