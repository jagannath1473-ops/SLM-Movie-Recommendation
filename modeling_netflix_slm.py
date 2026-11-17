from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rope_params(
    head_dim,
    theta_base=10000,
    context_length=4096,
    dtype=torch.float32
):
    assert head_dim % 2 == 0, "Head dimension must be even"

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, head_dim, 2, dtype=dtype) / head_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)

    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape

    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)

    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6
        )
        return (x_norm * self.weight).to(x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_groups == 0

        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(
            d_model, n_kv_groups * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            d_model, n_kv_groups * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape

        q = self.q_proj(x).view(
            b, t, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            b, t, self.n_kv_groups, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            b, t, self.n_kv_groups, self.head_dim
        ).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        repeat = self.n_heads // self.n_kv_groups
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(
            F.silu(self.gate(x)) * self.up(x)
        )

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups, d_ff):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_groups
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

class Qwen3Model(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads,
        n_kv_groups, n_layers, d_ff, max_seq_len
    ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, n_kv_groups, d_ff
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.out_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

        # Share embedding and output weights
        self.out_head.weight = self.tok_emb.weight
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        cos, sin = compute_rope_params(
            head_dim=d_model // n_heads,
            context_length=max_seq_len
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.max_seq_len = max_seq_len

    def forward(self, input_ids, targets=None):
        assert input_ids.shape[1] <= self.max_seq_len

        x = self.tok_emb(input_ids)

        for block in self.blocks:
            x = block(x, self.cos, self.sin)

        logits = self.out_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

