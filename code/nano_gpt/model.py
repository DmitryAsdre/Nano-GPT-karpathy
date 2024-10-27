import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, 
                 head_size,
                 n_embed, 
                 block_size,
                 dropout):
        super().__init__()
        
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.dropout = dropout
        
        self.key = nn.Linear(self.n_embed, head_size, bias=False)
        self.query = nn.Linear(self.n_embed, head_size, bias=False)
        self.value = nn.Linear(self.n_embed, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_heads, 
                 head_size,
                 n_embed,
                 block_size, 
                 dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.dropout = dropout
        
        self.heads = nn.ModuleList([Head(self.head_size, 
                                         self.n_embed, 
                                         self.block_size, 
                                         self.dropout) for _ in range(self.num_heads)])
        self.proj = nn.Linear(self.n_embed, self.n_embed)
        self.dropout_ = nn.Dropout(self.dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout_(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, 
                 n_embed,
                 ffwd_coef, 
                 dropout):
        super().__init__()
        
        self.n_embed = n_embed
        self.ffwd_coef = ffwd_coef
        self.dropout = dropout
        
        self.net = nn.Sequential(
            nn.Linear(self.n_embed, self.ffwd_coef*self.n_embed),
            nn.ReLU(),
            nn.Linear(self.ffwd_coef*self.n_embed, self.n_embed),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,
                 n_embed, 
                 num_heads, 
                 block_size, 
                 dropout,
                 ffwd_coef):
        super().__init__()
        
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.ffwd_coef = ffwd_coef
        
        self.head_size = self.n_embed // self.num_heads
        
        self.mha = MultiHeadAttention(self.num_heads, 
                                      self.head_size,
                                      self.n_embed,
                                      self.block_size, 
                                      self.dropout)
        self.ffwd = FeedForward(self.n_embed,
                                self.ffwd_coef,
                                self.dropout)
        self.ln1 = nn.LayerNorm(self.n_embed)
        self.ln2 = nn.LayerNorm(self.n_embed)
        
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class NanoGPT(nn.Module):
    def __init__(self, 
                 vocab_size,
                 block_size, 
                 n_embed,
                 num_heads,
                 dropout,
                 ffwd_coef,
                 num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffwd_coef = ffwd_coef
        self.num_layers = num_layers
        
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embed)
        
        self.blocks = nn.Sequential(*[Block(n_embed=self.n_embed, 
                                            num_heads=self.num_heads, 
                                            block_size=self.block_size,
                                            dropout=self.dropout,
                                            ffwd_coef=self.ffwd_coef) for _ in range(self.num_layers)])
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.ffwd = FeedForward(self.n_embed,
                                self.ffwd_coef,
                                self.dropout)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.position_embedding_table.weight.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx