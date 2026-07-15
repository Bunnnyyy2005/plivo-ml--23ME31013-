"""A small GPT in plain PyTorch. Yours to modify or replace entirely —
attention, SSM, whatever — as long as evaluate.py still works and the
parameter cap holds.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 1024
    block_size = 256
    n_layer = 4
    n_head = 6
    n_embd = 192
    dropout = 0.0
    tie_weights = True

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm_x + self.eps))

def apply_rotary_emb(x, freqs_cis):
    B, T, H, D = x.shape
    x_reshaped = x.view(B, T, H, D // 2, 2)
    freqs_cis = freqs_cis.view(1, T, 1, D // 2, 2)
    x_r, x_i = x_reshaped[..., 0], x_reshaped[..., 1]
    f_r, f_i = freqs_cis[..., 0], freqs_cis[..., 1]
    out_r = x_r * f_r - x_i * f_i
    out_i = x_r * f_i + x_i * f_r
    out = torch.stack([out_r, out_i], dim=-1).view(B, T, H, D)
    return out

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.register_buffer("freqs_cis", precompute_freqs_cis(cfg.n_embd // cfg.n_head, cfg.block_size * 2), persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        freqs_cis = self.freqs_cis[:T]
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, freqs_cis)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # We use a set to not double-count tied weights
        return sum(p.numel() for p in set(self.parameters()))
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # byte-level tokenizer default
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = False   # <- one of many things worth questioning


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        # baseline init: plain normal, one std for everything
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
