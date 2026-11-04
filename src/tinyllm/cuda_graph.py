"""CUDA-graph decode (4.11, docs/INFERENCE.md).

A decode step of a small model is launch-bound: ~10 kernels per layer, each a few µs of
GPU work, and the CPU needs about as long to launch each one. A CUDA graph records the
whole step once and replays it with a single launch.

A graph replays fixed kernels on fixed addresses, so the step must have static shapes:
- StaticKVCache keeps the position as a device tensor. K/V are written with index_copy_
  at that position, RoPE is gathered at it, and every decode step attends over a fixed
  window of W keys, with keys past the position masked out.
- W comes from length buckets (128, 256, ..., max_len). Each bucket has its own graph;
  the step uses the smallest bucket that holds pos + 1 keys. All graphs share one memory
  pool and the same static buffers (cache, input token, position).
- Prefill stays eager: it runs once per request and has a different shape each time.
  Sampling stays eager too.

Not bit-identical to eager generate(): eager decode attends over exactly pos + 1 keys
with no mask; the graph attends over W keys with a mask, and SDPA picks a different
kernel for that. Graph replay *is* bit-identical to the same static step run eagerly.

Autocast runs with cache_enabled=False: its weight-cast cache doesn't survive capture.
"""

import torch
from torch import Tensor

from tinyllm.kv_cache import causal_kwargs, padded_causal_mask
from tinyllm.model import Transformer


def default_buckets(max_len: int, first: int = 128) -> tuple[int, ...]:
    """first, 2*first, ... below max_len, then max_len itself."""
    out, b = [], first
    while b < max_len:
        out.append(b)
        b *= 2
    return (*out, max_len)


class StaticKVCache:
    """Shared-position cache with fixed shapes (B rows, max_len positions).

    Same model protocol as kv_cache.KVCache. Two modes:
    - prefill (eager, position 0): as KVCache, writes and returns positions 0..T-1.
    - decode (`window` set, T=1): position from the device tensor `pos`; returns the first
      `window` keys with a mask, so shapes don't depend on the position.
    first_valid (B,) is the left-pad count per row; zeros when unpadded.
    """

    def __init__(self, n_layers: int, batch: int, max_len: int, device):
        self.n_layers, self.batch, self.max_len = n_layers, batch, max_len
        self.k: list[Tensor | None] = [None] * n_layers
        self.v: list[Tensor | None] = [None] * n_layers
        self.pos = torch.zeros(1, dtype=torch.long, device=device)      # device copy: graph input
        self.first_valid = torch.zeros(batch, dtype=torch.long, device=device)
        self.padded = False
        self.n = 0                       # host copy of pos, for bounds and bucket choice (no sync)
        self.window: int | None = None   # None = prefill mode
        self.attn_kwargs: dict = {}

    def reset(self, pad_lens: Tensor | None = None) -> None:
        self.n, self.window = 0, None
        self.pos.zero_()
        self.padded = pad_lens is not None
        self.first_valid.copy_(pad_lens if self.padded else torch.zeros_like(self.first_valid))

    def begin_step(self, T: int, rope) -> tuple[Tensor, Tensor]:
        if self.window is None:                          # eager prefill from position 0
            if self.n != 0:
                raise ValueError("StaticKVCache prefills only from position 0")
            cos, sin = rope(T, 0)
            self.attn_kwargs = ({"attn_mask": padded_causal_mask(0, T, self.first_valid)}
                                if self.padded else causal_kwargs(0, T, cos.device))
            return cos, sin
        if T != 1:
            raise ValueError("decode mode takes one token per row")
        k_pos = torch.arange(self.window, device=self.pos.device)
        valid = (k_pos <= self.pos) & (k_pos >= self.first_valid[:, None])        # (B, W)
        self.attn_kwargs = {"attn_mask": valid[:, None, None, :]}                 # (B, 1, 1, W)
        return rope.cos.index_select(0, self.pos), rope.sin.index_select(0, self.pos)

    def end_step(self, T: int) -> None:
        if self.window is None:
            self.n = T
            self.pos.fill_(T)
        else:
            self.pos.add_(1)             # captured into the graph; the host copy moves in step()

    def layer(self, i: int) -> "StaticLayerKV":
        return StaticLayerKV(self, i)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class StaticLayerKV:
    def __init__(self, cache: StaticKVCache, layer_idx: int):
        self.cache = cache
        self.i = layer_idx

    @property
    def attn_kwargs(self) -> dict:
        return self.cache.attn_kwargs

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        c, i = self.cache, self.i
        B, H, T, D = k.shape
        if c.k[i] is None:
            shape = (c.batch, H, c.max_len, D)
            c.k[i] = torch.zeros(shape, dtype=k.dtype, device=k.device)
            c.v[i] = torch.zeros(shape, dtype=v.dtype, device=v.device)
        elif c.k[i].dtype != k.dtype:
            raise ValueError(f"cache holds {c.k[i].dtype}, got {k.dtype}")
        if c.window is None:
            if T > c.max_len:
                raise ValueError(f"KV cache full: position {T} > max_len {c.max_len}")
            c.k[i][:, :, :T] = k
            c.v[i][:, :, :T] = v
            return c.k[i][:, :, :T], c.v[i][:, :, :T]
        c.k[i].index_copy_(2, c.pos, k)
        c.v[i].index_copy_(2, c.pos, v)
        return c.k[i][:, :, :c.window], c.v[i][:, :, :c.window]


class GraphDecoder:
    """Prefill eagerly, then decode one token per row with one CUDA-graph replay per step.

    Reusable: prefill() resets the cache, and graphs captured for one request serve the
    next (same batch size). use_graph=False runs the identical static step eagerly, the
    reference that replay must match bit for bit.
    """

    def __init__(self, model: Transformer, batch: int, max_len: int | None = None,
                 buckets: tuple[int, ...] | None = None, use_graph: bool = True):
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA model")
        self.model = model
        self.max_len = max_len or model.cfg.max_seq_len
        if self.max_len > model.cfg.max_seq_len:
            raise ValueError(f"max_len {self.max_len} > max_seq_len {model.cfg.max_seq_len}")
        self.buckets = tuple(sorted(buckets)) if buckets else default_buckets(self.max_len)
        if self.buckets[-1] < self.max_len:
            raise ValueError(f"largest bucket {self.buckets[-1]} < max_len {self.max_len}")
        self.cache = StaticKVCache(model.cfg.n_layers, batch, self.max_len, device)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.use_graph = use_graph
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, Tensor]] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)

    @torch.no_grad()
    def prefill(self, idx: Tensor, pad_lens: Tensor | None = None) -> Tensor:
        """idx (B, T), left-padded by pad_lens (B,) if given. Returns logits (B, T, V)."""
        if idx.shape[0] != self.cache.batch:
            raise ValueError(f"batch {idx.shape[0]} != decoder batch {self.cache.batch}")
        self.cache.reset(pad_lens)
        with self._autocast():
            logits, _ = self.model(idx, cache=self.cache)
        return logits

    def _forward(self) -> Tensor:
        with self._autocast():
            logits, _ = self.model(self.tok, cache=self.cache)
        return logits

    def _capture(self, window: int) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        # Warm up on a side stream (lazy cuBLAS / SDPA setup must not happen during capture).
        # The warm-up runs the real step: its K/V writes are the ones replay will make, and
        # the position it advances is put back.
        saved = self.cache.pos.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._forward()
        torch.cuda.current_stream().wait_stream(side)
        self.cache.pos.copy_(saved)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._forward()
        return graph, out

    def bucket(self, n_keys: int) -> int:
        return next(b for b in self.buckets if b >= n_keys)

    @torch.no_grad()
    def step(self, tok: Tensor) -> Tensor:
        """tok (B, 1) → logits (B, 1, V). With graphs, the result is a static buffer that
        the next step overwrites: read it (or sample from it) first."""
        c = self.cache
        if c.n >= self.max_len:
            raise ValueError(f"KV cache full: position {c.n + 1} > max_len {self.max_len}")
        c.window = self.bucket(c.n + 1)
        self.tok.copy_(tok)
        if not self.use_graph:
            out = self._forward()
        else:
            if c.window not in self.graphs:
                self.graphs[c.window] = self._capture(c.window)
            graph, out = self.graphs[c.window]
            graph.replay()
        c.n += 1
        return out
