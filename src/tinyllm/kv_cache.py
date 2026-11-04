"""KV caches for incremental decoding (docs/INFERENCE.md).

The model talks to a cache through three calls per forward pass:
    rope = cache.begin_step(T, rope_module)   # positions of the T new tokens; builds the mask
    view = cache.layer(i)                     # per layer: view.update(k, v), view.attn_kwargs
    cache.end_step(T)                         # the new tokens are now stored

- KVCache (4.3, 4.4 static): one shared position for every row. Optional left padding
  (`pad_lens`) lets prompts of different lengths share a batch: pad keys are masked.
- SlotKVCache (4.4 continuous): S independent slots, each with its own length, so
  requests can join and leave between decode steps.

Buffers are [B, n_kv_heads, max_len, head_dim] per layer. No reference cycles: the
caches own plain tensor lists and `layer(i)` returns a short-lived view. (A cycle once
kept every finished cache's GPU memory alive until Python's cycle collector ran.)

Left padding needs no per-row RoPE positions: RoPE makes q·k depend only on the
distance between two positions, and padding shifts every real token of a row by the
same amount. Pad *queries* may attend to themselves, so no row is fully masked (a fully
masked softmax row is NaN, and NaN would leak through 0 × NaN in later layers).
"""

import torch
from torch import Tensor


def causal_kwargs(start: int, T: int, device) -> dict:
    """SDPA masking for T new queries at positions start..start+T-1 over start+T keys, no padding.

    - start 0 (prefill / training): standard causal.
    - one new token: it may see every cached key, so no mask.
    - several new tokens on a non-empty cache (chunked prefill, speculative verify):
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


def padded_causal_mask(start: int, T: int, first_valid: Tensor) -> Tensor:
    """(B, 1, T, start+T) bool: causal, keys before first_valid[b] hidden, diagonal always kept."""
    dev = first_valid.device
    q_pos = torch.arange(start, start + T, device=dev)[:, None]          # (T, 1)
    k_pos = torch.arange(start + T, device=dev)[None, :]                 # (1, K)
    causal = k_pos <= q_pos
    valid = k_pos[None] >= first_valid[:, None, None]                    # (B, 1, K)
    return (causal[None] & (valid | (k_pos == q_pos)[None]))[:, None]    # (B, 1, T, K)


class KVCache:
    """Shared-position cache. pad_lens (B,): left-pad count per row, or None for no padding."""

    def __init__(self, n_layers: int, max_len: int, pad_lens: Tensor | None = None):
        self.max_len = max_len
        self.pos = 0
        self.pad_lens = pad_lens
        self.k: list[Tensor | None] = [None] * n_layers
        self.v: list[Tensor | None] = [None] * n_layers
        self.attn_kwargs: dict = {}

    def begin_step(self, T: int, rope) -> tuple[Tensor, Tensor]:
        cos, sin = rope(T, self.pos)
        if self.pad_lens is None:
            self.attn_kwargs = causal_kwargs(self.pos, T, cos.device)
        else:
            self.attn_kwargs = {"attn_mask": padded_causal_mask(self.pos, T, self.pad_lens)}
        return cos, sin

    def end_step(self, T: int) -> None:
        self.pos += T

    def layer(self, i: int) -> "LayerKV":
        return LayerKV(self, i)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class LayerKV:
    """One layer's view of a KVCache."""

    def __init__(self, cache: KVCache, layer_idx: int):
        self.cache = cache
        self.i = layer_idx

    @property
    def start(self) -> int:
        """Position of the first new token = number of tokens already cached."""
        return self.cache.pos

    @property
    def attn_kwargs(self) -> dict:
        return self.cache.attn_kwargs

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


class SlotKVCache:
    """S slots, each holding one sequence with its own length (continuous batching, 4.4).

    Slot b holds keys at positions first[b] .. lens[b]-1 (positions before first[b] are
    left padding from a batched prefill). A decode step feeds one token per slot, at
    position lens[b]. Free slots still run (their output is ignored) so shapes stay fixed.

    Every slot reserves max_len positions whatever its request needs: the waste that
    paged KV (4.5) removes. `used_tokens()` vs `reserved_tokens()` measures it.
    """

    def __init__(self, n_layers: int, n_slots: int, n_kv_heads: int, head_dim: int,
                 max_len: int, dtype: torch.dtype, device):
        shape = (n_slots, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.n_slots, self.max_len = n_slots, max_len
        # CPU copies drive scheduling without GPU syncs; device copies feed RoPE and masks.
        self.first = [0] * n_slots
        self.lens = [0] * n_slots
        self.active = [False] * n_slots
        self._lens_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._first_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._active_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._rows = torch.arange(n_slots, device=device)
        self.attn_kwargs: dict = {}
        self._end = 0

    # -- scheduling --------------------------------------------------------
    def free_slots(self) -> list[int]:
        return [s for s in range(self.n_slots) if not self.active[s]]

    def admit(self, slots: list[int], prefill: KVCache, pad_lens: list[int]) -> None:
        """Copy a finished batched prefill (row j → slot slots[j]) into the slots."""
        L = prefill.pos
        if L > self.max_len:
            raise ValueError(f"prompt length {L} > slot max_len {self.max_len}")
        idx = torch.tensor(slots, device=self._lens_dev.device)
        for i in range(len(self.k)):
            self.k[i][idx, :, :L] = prefill.k[i][:, :, :L]
            self.v[i][idx, :, :L] = prefill.v[i][:, :, :L]
        for s, p in zip(slots, pad_lens):
            self.first[s], self.lens[s], self.active[s] = p, L, True
        self._sync()

    def release(self, slot: int) -> None:
        self.active[slot] = False
        self.first[slot] = self.lens[slot] = 0
        self._sync()

    def _sync(self) -> None:
        """Push the CPU bookkeeping to the device (only on admit / release, not every step)."""
        self._lens_dev.copy_(torch.tensor(self.lens))
        self._first_dev.copy_(torch.tensor(self.first))
        self._active_dev.copy_(torch.tensor(self.active, dtype=torch.long))

    def can_admit(self, lengths: list[int]) -> bool:
        return True                                          # a free slot is all a request needs

    def reserve_step(self) -> list[int]:
        return []                                            # never preempts: capacity is per slot

    def used_tokens(self) -> int:
        return sum(n - f for n, f, a in zip(self.lens, self.first, self.active) if a)

    def allocated_tokens(self) -> int:
        """KV positions claimed by running sequences (a whole slot each)."""
        return sum(self.active) * self.max_len

    def reserved_tokens(self) -> int:
        return self.n_slots * self.max_len

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    # -- model protocol (decode only: one token per slot) ------------------
    def begin_step(self, T: int, rope) -> tuple[Tensor, Tensor]:
        if T != 1:
            raise ValueError("SlotKVCache decodes one token per slot; prefill goes through KVCache")
        if max(self.lens) >= self.max_len:
            raise ValueError(f"a slot is full (max_len {self.max_len})")
        self._end = max(self.lens) + 1                       # attend only as far as the longest slot
        k_pos = torch.arange(self._end, device=self._lens_dev.device)
        valid = (k_pos >= self._first_dev[:, None]) & (k_pos <= self._lens_dev[:, None])
        self.attn_kwargs = {"attn_mask": valid[:, None, None, :]}   # (S, 1, 1, end)
        return rope.at(self._lens_dev[:, None])

    def end_step(self, T: int) -> None:
        for s in range(self.n_slots):
            if self.active[s]:
                self.lens[s] += 1
        self._lens_dev += self._active_dev                   # same update on the device, no copy

    def layer(self, i: int) -> "SlotLayerKV":
        return SlotLayerKV(self, i)


class SlotLayerKV:
    def __init__(self, cache: SlotKVCache, layer_idx: int):
        self.cache = cache
        self.i = layer_idx

    @property
    def attn_kwargs(self) -> dict:
        return self.cache.attn_kwargs

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """k, v (S, H, 1, D): write each slot's token at its own position lens[s]."""
        c, i = self.cache, self.i
        c.k[i][c._rows, :, c._lens_dev] = k[:, :, 0]
        c.v[i][c._rows, :, c._lens_dev] = v[:, :, 0]
        return c.k[i][:, :, :c._end], c.v[i][:, :, :c._end]
