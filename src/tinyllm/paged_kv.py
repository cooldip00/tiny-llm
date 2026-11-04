"""Step 4.5: paged KV cache (vLLM-style blocks), a drop-in for SlotKVCache in run_continuous.

The analogy is virtual memory:
    block table  = page table      (one per sequence: logical block j -> physical block)
    block        = page            (block_size tokens of K and V, every layer)
    token t      = (table[t // block_size], t % block_size)

All layers share one block table: block b holds the same token positions in every layer's
pool. Blocks are allocated as a sequence grows, not reserved up front, and every block has
the same size, so a freed block fits any sequence: no external fragmentation, and at most
block_size - 1 wasted slots per sequence (its last, partly filled block).

Left padding from the batched prefill is not stored: only real tokens go into blocks.
The pad count becomes a RoPE offset (`first`), so a token at storage index j keeps its
rotation position first + j and the next query lands at first + len, as in SlotKVCache.

Attention reads the blocks by gathering each sequence's blocks into a contiguous
(S, H, n_blocks * block_size, D) view and calling SDPA as before. vLLM's PagedAttention
kernel reads the blocks in place instead; the gather is this toy's extra copy (measured
in docs/INFERENCE.md).

When decode needs more blocks than are free, the newest sequence is preempted: its blocks
are freed and the engine requeues it to be recomputed (prompt + tokens so far), which
reproduces the same greedy tokens. This is vLLM's default recompute policy.
"""

import math

import torch
from torch import Tensor

from tinyllm.kv_cache import KVCache


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Fixed-size blocks with a free list. Lowest ids are handed out first (reused LIFO)."""

    def __init__(self, n_blocks: int, block_size: int):
        self.n_blocks, self.block_size = n_blocks, block_size
        self._free = list(range(n_blocks - 1, -1, -1))     # stack: pop() gives block 0 first
        self._in_use = [False] * n_blocks
        self.peak_used = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.n_blocks - len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise OutOfBlocks(f"all {self.n_blocks} KV blocks in use")
        b = self._free.pop()
        self._in_use[b] = True
        self.peak_used = max(self.peak_used, self.num_used)
        return b

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if not self._in_use[b]:
                raise ValueError(f"block {b} freed twice (or never allocated)")
            self._in_use[b] = False
            self._free.append(b)


class PagedKVCache:
    """S sequence slots over a shared pool of n_blocks KV blocks.

    Same model protocol as SlotKVCache (begin_step / layer / end_step, one token per slot)
    and the same scheduling calls (free_slots / admit / release), plus can_admit() and
    reserve_step() so the engine can respect the block budget.
    """

    def __init__(self, n_layers: int, n_slots: int, n_kv_heads: int, head_dim: int,
                 n_blocks: int, block_size: int, max_len: int, dtype: torch.dtype, device):
        self.alloc = BlockAllocator(n_blocks, block_size)
        self.bs, self.n_slots, self.max_len = block_size, n_slots, max_len
        self.max_blocks = math.ceil(max_len / block_size)
        self.scratch = n_blocks           # extra block, never allocated: free slots write here
        shape = (n_blocks + 1, n_kv_heads, block_size, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        # CPU bookkeeping (scheduling, no GPU syncs) and its device mirror (indexing, masks).
        self.tables: list[list[int]] = [[] for _ in range(n_slots)]
        self.lens = [0] * n_slots         # tokens stored (no padding)
        self.first = [0] * n_slots        # RoPE offset = left-pad count at prefill
        self.active = [False] * n_slots
        self._admit_no = [0] * n_slots    # admission order, for picking preemption victims
        self._admits = 0
        self._table_dev = torch.full((n_slots, self.max_blocks), self.scratch,
                                     dtype=torch.long, device=device)
        self._lens_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._first_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._active_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._rows = torch.arange(n_slots, device=device)
        self.attn_kwargs: dict = {}
        self._n_read_blocks = 0
        self._blk = self._off = None

    # -- scheduling --------------------------------------------------------
    def free_slots(self) -> list[int]:
        return [s for s in range(self.n_slots) if not self.active[s]]

    def can_admit(self, lengths: list[int]) -> bool:
        """Enough free blocks for these prompts, plus room for each one's first decode token."""
        return sum(math.ceil((n + 1) / self.bs) for n in lengths) <= self.alloc.num_free

    def admit(self, slots: list[int], prefill: KVCache, pad_lens: list[int]) -> None:
        """Copy row j of a padded prefill (real tokens only) into fresh blocks for slots[j]."""
        L = prefill.pos
        dev = self._lens_dev.device
        rows, src, blk, off = [], [], [], []
        for j, (s, p) in enumerate(zip(slots, pad_lens)):
            n = L - p
            if n >= self.max_len:
                raise ValueError(f"prompt length {n} >= max_len {self.max_len}")
            blocks = [self.alloc.alloc() for _ in range(math.ceil(n / self.bs))]
            self.tables[s] = blocks
            self.lens[s], self.first[s], self.active[s] = n, p, True
            self._admits += 1
            self._admit_no[s] = self._admits
            self._table_dev[s].fill_(self.scratch)
            self._table_dev[s, :len(blocks)] = torch.tensor(blocks, device=dev)
            t = torch.arange(n)
            rows.append(torch.full((n,), j))
            src.append(t + p)
            blk.append(torch.tensor(blocks, dtype=torch.long)[t // self.bs])
            off.append(t % self.bs)
        rows, src, blk, off = (torch.cat(x).to(dev) for x in (rows, src, blk, off))
        for i in range(len(self.k)):
            self.k[i][blk, :, off] = prefill.k[i][rows, :, src]     # (N, H, D) both sides
            self.v[i][blk, :, off] = prefill.v[i][rows, :, src]
        self._sync()

    def release(self, slot: int) -> None:
        self.alloc.free(self.tables[slot])
        self.tables[slot] = []
        self.active[slot] = False
        self.lens[slot] = self.first[slot] = 0
        self._table_dev[slot].fill_(self.scratch)
        self._sync()

    def reserve_step(self) -> list[int]:
        """Make sure the next decode step's new blocks can be allocated. Preempts the newest
        sequences (frees their blocks) until they can; returns the preempted slots, newest first."""
        victims = []
        while True:
            need = sum(1 for s in range(self.n_slots)
                       if self.active[s] and len(self.tables[s]) * self.bs <= self.lens[s])
            if need <= self.alloc.num_free or not any(self.active):
                return victims
            v = max((s for s in range(self.n_slots) if self.active[s]), key=self._admit_no.__getitem__)
            self.release(v)
            victims.append(v)

    def _sync(self) -> None:
        self._lens_dev.copy_(torch.tensor(self.lens))
        self._first_dev.copy_(torch.tensor(self.first))
        self._active_dev.copy_(torch.tensor(self.active, dtype=torch.long))

    def used_tokens(self) -> int:
        return sum(n for n, a in zip(self.lens, self.active) if a)

    def allocated_tokens(self) -> int:
        return self.alloc.num_used * self.bs

    def reserved_tokens(self) -> int:
        """Pool capacity: the KV memory this cache holds."""
        return self.alloc.n_blocks * self.bs

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    # -- model protocol (decode only: one token per slot) ------------------
    def begin_step(self, T: int, rope) -> tuple[Tensor, Tensor]:
        if T != 1:
            raise ValueError("PagedKVCache decodes one token per slot; prefill goes through KVCache")
        if max(self.lens) >= self.max_len:
            raise ValueError(f"a sequence is full (max_len {self.max_len})")
        new_rows, new_cols, new_blks = [], [], []
        for s in range(self.n_slots):                         # grow sequences that filled a block
            if self.active[s] and len(self.tables[s]) * self.bs <= self.lens[s]:
                b = self.alloc.alloc()                        # reserve_step made room
                new_rows.append(s)
                new_cols.append(len(self.tables[s]))
                new_blks.append(b)
                self.tables[s].append(b)
        if new_rows:
            dev = self._table_dev.device
            self._table_dev[torch.tensor(new_rows, device=dev),
                            torch.tensor(new_cols, device=dev)] = torch.tensor(new_blks, device=dev)
        self._n_read_blocks = math.ceil((max(self.lens) + 1) / self.bs)
        self._blk = self._table_dev[self._rows, self._lens_dev // self.bs]   # where to write
        self._off = self._lens_dev % self.bs
        k_pos = torch.arange(self._n_read_blocks * self.bs, device=self._lens_dev.device)
        self.attn_kwargs = {"attn_mask": (k_pos <= self._lens_dev[:, None])[:, None, None, :]}
        return rope.at((self._first_dev + self._lens_dev)[:, None])

    def end_step(self, T: int) -> None:
        for s in range(self.n_slots):
            if self.active[s]:
                self.lens[s] += 1
        self._lens_dev += self._active_dev

    def layer(self, i: int) -> "PagedLayerKV":
        return PagedLayerKV(self, i)


class PagedLayerKV:
    def __init__(self, cache: PagedKVCache, layer_idx: int):
        self.cache = cache
        self.i = layer_idx

    @property
    def attn_kwargs(self) -> dict:
        return self.cache.attn_kwargs

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """k, v (S, H, 1, D): write each slot's token into its block, then gather its blocks."""
        c, i = self.cache, self.i
        c.k[i][c._blk, :, c._off] = k[:, :, 0]
        c.v[i][c._blk, :, c._off] = v[:, :, 0]
        table = c._table_dev[:, :c._n_read_blocks]                     # (S, nb)
        S, nb = table.shape
        _, H, bs, D = c.k[i].shape

        def gather(pool: Tensor) -> Tensor:                            # (S, nb, H, bs, D) -> (S, H, nb*bs, D)
            return pool[table].transpose(1, 2).reshape(S, H, nb * bs, D)

        return gather(c.k[i]), gather(c.v[i])
