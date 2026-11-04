"""Baseline multi-head causal self-attention (D-010: PyTorch SDPA)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.attention.base import Attention, register
from tinyllm.config import ModelConfig
from tinyllm.model import apply_rope


def causal_kwargs(start: int, T: int, device) -> dict:
    """SDPA masking for T new queries at positions start..start+T-1 over start+T keys.

    - No cache / prefill (start 0): standard causal.
    - One new token: it may see every cached key, so no mask.
    - Several new tokens on a non-empty cache (chunked prefill, speculative verify):
      SDPA's is_causal aligns to the top-left, which is wrong here, so pass an explicit
      mask where query i sees keys 0..start+i.
    """
    if start == 0:
        return {"is_causal": True}
    if T == 1:
        return {}
    q_pos = torch.arange(start, start + T, device=device)[:, None]
    k_pos = torch.arange(start + T, device=device)[None, :]
    return {"attn_mask": k_pos <= q_pos}


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
        start = 0
        if cache is not None:
            start = cache.start
            k, v = cache.update(k, v)                    # now all start+T cached positions
        # Fused kernel (FlashAttention / memory-efficient), never materialises the T×T matrix.
        y = F.scaled_dot_product_attention(q, k, v, **causal_kwargs(start, T, q.device))
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return 2 * self.n_heads * self.head_dim * dtype_bytes  # one K and one V vector per head
