"""Step 4.3: a preallocated KV cache for incremental decoding (docs/INFERENCE.md).

Each layer keeps K and V buffers of shape [B, n_kv_heads, max_len, head_dim].
Prefill writes the whole prompt at once; each decode step writes one position.
Buffers are allocated on first use with the layer's own dtype, device and head
count, so bf16 on GPU, fp32 on CPU and future GQA layers all work unchanged.

`pos` is shared by every row (static batch: all sequences have the same length).
Per-sequence lengths arrive with continuous batching (4.4) and paging (4.5).

No reference cycles: the cache owns plain tensor lists, and `layer(i)` returns a
short-lived view. A cycle (cache ↔ layer objects) would keep every finished cache's
GPU memory alive until Python's cycle collector happens to run.
"""

import torch
from torch import Tensor


class KVCache:
    def __init__(self, n_layers: int, max_len: int):
        self.max_len = max_len
        self.pos = 0
        self.k: list[Tensor | None] = [None] * n_layers
        self.v: list[Tensor | None] = [None] * n_layers

    def layer(self, i: int) -> "LayerKV":
        return LayerKV(self, i)

    def advance(self, n: int) -> None:
        self.pos += n

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class LayerKV:
    """One layer's view of the cache. Attention calls update(); the model advances pos."""

    def __init__(self, cache: KVCache, layer_idx: int):
        self.cache = cache
        self.i = layer_idx

    @property
    def start(self) -> int:
        """Position of the first new token = number of tokens already cached."""
        return self.cache.pos

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store k, v (B, H, T, D) at positions start..start+T-1; return all cached K, V so far."""
        c, i = self.cache, self.i
        B, H, T, D = k.shape
        if c.k[i] is None:
            shape = (B, H, c.max_len, D)
            c.k[i] = torch.zeros(shape, dtype=k.dtype, device=k.device)
            c.v[i] = torch.zeros(shape, dtype=v.dtype, device=v.device)
        start, end = self.start, self.start + T
        if end > c.max_len:
            raise ValueError(f"KV cache full: position {end} > max_len {c.max_len}")
        c.k[i][:, :, start:end] = k
        c.v[i][:, :, start:end] = v
        return c.k[i][:, :, :end], c.v[i][:, :, :end]
