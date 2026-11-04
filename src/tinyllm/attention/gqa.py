"""Grouped-query attention (Ainslie et al. 2023), step 4.9a. TinyLlama: 32 query heads, 4 KV heads.

Query heads are split into n_kv_heads groups of g = n_heads / n_kv_heads; every head in group j
attends with the same key/value head j. So q_proj stays d -> n_heads·hd while k_proj / v_proj
shrink to d -> n_kv_heads·hd, and the KV cache stores n_kv_heads heads: g× less decode state.

Head order matches HF Llama's `repeat_kv` (query head i uses KV head i // g), which is what
TinyLlama's weights expect (4.9b / 4.9c).

With n_kv_heads == n_heads this is exactly `mha` (same parameter names and shapes, so an `mha`
state dict loads into it).
"""

import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.attention.base import Attention, register
from tinyllm.config import ModelConfig
from tinyllm.model import apply_rope


def repeat_kv(x: Tensor, groups: int) -> Tensor:
    """(B, kv, T, hd) -> (B, kv·groups, T, hd): KV head j is copied to query heads j·g .. j·g+g-1.

    The explicit form of HF's grouping. Used as the reference in tests; forward uses enable_gqa.
    """
    if groups == 1:
        return x
    B, kv, T, hd = x.shape
    return x[:, :, None].expand(B, kv, groups, T, hd).reshape(B, kv * groups, T, hd)


@register("gqa")
class GQA(Attention):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__(cfg, layer_idx)
        d, h, kv, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_heads, self.n_kv_heads, self.head_dim = h, kv, hd
        self.groups = h // kv
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, kv * hd, bias=False)
        self.v_proj = nn.Linear(d, kv * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], cache=None) -> Tensor:
        """cache: a kv_cache.LayerKV / SlotLayerKV / paged layer view. It stores n_kv_heads heads."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = {"is_causal": True}
        if cache is not None:
            k, v = cache.update(k, v)                    # (B, kv, S, hd): the cache holds only kv heads
            mask = cache.attn_kwargs
        # enable_gqa: the kernel maps query head i to KV head i // g itself, without copying K/V.
        # Measured in 4.9a (torch 2.14, TinyLlama heads, bf16): 0.07-0.74x the time of
        # repeat_kv + SDPA, masks included. repeat_kv at decode copies the whole cache every
        # step (B=32, 512 positions: 1.09 ms vs 0.08 ms). repeat_kv stays as the test reference.
        y = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.groups > 1, **mask)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes  # one K and one V vector per KV head
