"""Llama-style decoder components, written from scratch (docs/ARCHITECTURE.md).

Parameter names follow HF Llama (weight, gate_proj, up_proj, down_proj) so the
Phase 4.8 export, and loading TinyLlama in Phase 4.9, need no renaming.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.config import ModelConfig


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight

    Like LayerNorm, but with no mean subtraction and no bias: cheaper, and it
    works just as well in LLMs. The statistic is computed in fp32 even under
    bf16, because a sum of squares loses precision in bf16.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables.

    Each pair of channels (i, i + head_dim/2) in q and k is rotated by the angle
    pos * theta^(-2i/head_dim). Then q·k depends only on the *relative* distance
    between positions, and no position parameters are learned.
    This is the "rotate_half" layout that HF Llama uses, so TinyLlama weights
    work unchanged.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)  # (D/2,)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(pos, inv_freq)              # (T, D/2)
        angles = torch.cat([angles, angles], dim=-1)     # (T, D): same angle for i and i + D/2
        # Not saved in checkpoints (persistent=False): they're derived from the config.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """cos/sin for positions offset .. offset+seq_len-1. The offset is for KV-cached decode."""
        end = offset + seq_len
        if end > self.cos.shape[0]:
            raise ValueError(f"position {end} exceeds max_seq_len {self.cos.shape[0]}")
        return self.cos[offset:end], self.sin[offset:end]


def rotate_half(x: Tensor) -> Tensor:
    """[x1, x2] -> [-x2, x1], with x1, x2 the two halves of the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)."""
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)): a gated MLP, used by Llama, Mistral, Qwen.

    It has 3 matrices instead of 2, so ffn_hidden is ~(8/3)·d rather than 4·d
    to keep the param count similar.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
