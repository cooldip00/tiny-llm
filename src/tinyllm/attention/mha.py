"""Baseline multi-head causal self-attention (D-010: PyTorch SDPA)."""

import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.attention.base import Attention, register
from tinyllm.config import ModelConfig
from tinyllm.model import apply_rope


@register("mha")
class MHA(Attention):
    """softmax(q·kᵀ / sqrt(head_dim) + causal mask) · v, with every head having its own k and v."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__(cfg, layer_idx)
        if cfg.n_kv_heads != cfg.n_heads:
            raise ValueError("mha needs n_kv_heads == n_heads; use 'gqa' for fewer KV heads")
        d, h, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.n_heads, self.head_dim = h, hd
        # Separate projections with HF Llama names (export / TinyLlama loading).
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, h * hd, bias=False)
        self.v_proj = nn.Linear(d, h * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], cache=None) -> Tensor:
        """cache: a kv_cache.LayerKV. x then holds only the new tokens (positions cache.start..)."""
        B, T, _ = x.shape
        # (B, T, h*hd) -> (B, h, T, hd)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = {"is_causal": True}
        if cache is not None:
            k, v = cache.update(k, v)                    # all cached positions, new ones included
            mask = cache.attn_kwargs                     # the cache knows positions and padding
        # Fused kernel (FlashAttention / memory-efficient), never materialises the T×T matrix.
        y = F.scaled_dot_product_attention(q, k, v, **mask)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return 2 * self.n_heads * self.head_dim * dtype_bytes  # one K and one V vector per head
