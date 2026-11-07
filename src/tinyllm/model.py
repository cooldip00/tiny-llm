"""Llama-style decoder components, written from scratch (docs/ARCHITECTURE.md).

Parameter names follow HF Llama (weight, gate_proj, up_proj, down_proj) so the
Phase 4.8 export, and loading TinyLlama in Phase 4.9, need no renaming.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.config import ModelConfig


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight

    Like LayerNorm, but with no mean subtraction and no bias: cheaper, and it
    works just as well in LLMs. The statistic is computed in fp32 even under
    bf16, because a sum of squares loses precision in bf16.

    Backend "triton" (step 4.10a) runs tinyllm.kernels.rmsnorm: one kernel instead of about seven,
    and add_forward fuses the residual add before it. It is used only for CUDA inputs when no
    gradient is needed (the kernel has no backward); otherwise this PyTorch code runs.
    """

    def __init__(self, dim: int, eps: float = 1e-5, backend: str = "torch"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.backend = _check_norm_backend(backend)

    def _kernel(self, x: Tensor):
        if self.backend != "triton" or not x.is_cuda:
            return None
        if torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad):
            return None
        return _norm_kernel()

    def forward(self, x: Tensor) -> Tensor:
        if (k := self._kernel(x)) is not None:
            return k.rmsnorm(x, self.weight, self.eps)
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight

    def add_forward(self, x: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """(self(x + a), x + a): the residual add and the norm after it, fused on backend "triton"."""
        if (k := self._kernel(x)) is not None and not (torch.is_grad_enabled() and a.requires_grad):
            return k.add_rmsnorm(x, a, self.weight, self.eps)
        h = x + a
        return self(h), h

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}, backend={self.backend}"


NORM_BACKENDS = ("torch", "triton")


def _check_norm_backend(backend: str) -> str:
    if backend not in NORM_BACKENDS:
        raise ValueError(f"unknown RMSNorm backend {backend!r}; one of {NORM_BACKENDS}")
    return backend


_NORM_KERNEL: list = []


def _norm_kernel():
    """tinyllm.kernels.rmsnorm, or None without Triton (imported once, on first CUDA use)."""
    if not _NORM_KERNEL:
        try:
            from tinyllm.kernels import rmsnorm
        except ImportError:
            rmsnorm = None
        _NORM_KERNEL.append(rmsnorm)
    return _NORM_KERNEL[0]


def set_norm_backend(model: nn.Module, backend: str) -> int:
    """Switch every RMSNorm in `model` to `backend` (step 4.10a). Returns how many."""
    _check_norm_backend(backend)
    norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
    for m in norms:
        m.backend = backend
    return len(norms)


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

    def at(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """cos/sin for per-row positions (B, T) → (B, 1, T, head_dim), broadcasting over heads.
        Used when rows sit at different positions (continuous batching, 4.4).
        No bounds check here (it would force a GPU sync every step): the caller checks."""
        return self.cos[positions][:, None], self.sin[positions][:, None]


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


class Block(nn.Module):
    """Pre-norm decoder layer: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        # Imported here, not at the top: the attention modules import apply_rope from this file.
        from tinyllm.attention import build_attention

        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = build_attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], cache=None) -> Tensor:
        a = self.self_attn(self.input_layernorm(x), rope, cache)
        y, x = self.post_attention_layernorm.add_forward(x, a)  # x + a, then its norm (one kernel, 4.10a)
        return x + self.mlp(y)


# Projections that write into the residual stream. Their init is scaled down so the
# stream's variance doesn't grow with depth (GPT-2 / nanoGPT recipe).
RESIDUAL_PROJ = ("o_proj.weight", "down_proj.weight")


class Transformer(nn.Module):
    """Token ids (B, T) -> logits (B, T, vocab), plus the next-token loss when targets are given."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # One matrix for both: saves V·d params (4.2M in main, 14% of the model).
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)  # 2 residual writes per layer
        for name, p in self.named_parameters():
            if name.endswith(RESIDUAL_PROJ):
                nn.init.normal_(p, mean=0.0, std=resid_std)
        # RMSNorm weights stay at 1.

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # a tied matrix is counted once

    def forward(
        self, idx: Tensor, targets: Tensor | None = None, cache=None,
    ) -> tuple[Tensor, Tensor | None]:
        """cache: a tinyllm.kv_cache cache for incremental decoding (None in training).
        With a cache, idx holds only the new tokens; the cache supplies their positions."""
        _, T = idx.shape
        rope = self.rope(T) if cache is None else cache.begin_step(T, self.rope)
        x = self.embed_tokens(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, rope, None if cache is None else cache.layer(i))
        if cache is not None:
            cache.end_step(T)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            # Loss in fp32: log-softmax over 8192 classes is imprecise in bf16.
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
