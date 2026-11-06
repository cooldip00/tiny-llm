"""Step 4.4: serving many requests: static batching vs continuous batching.

Workload model: offline, every request is queued at t=0 (no arrival process yet).

- run_static: take requests in order, B at a time. Left-pad the prompts, prefill once,
  decode until the batch's *longest* request is done. Rows that finish early keep
  computing (wasted work), and the next batch waits for the slowest row.
- run_continuous: S slots (SlotKVCache). Every step, queued requests are admitted into
  free slots (one batched, padded prefill for the newcomers), then every slot decodes
  one token together. A finished request frees its slot at once.

Both return per-request Results (generated tokens, TTFT, latency) and Stats (throughput,
wasted decode rows, KV memory reserved vs used). Correctness: each request's tokens must
equal its solo generate() output (tests/test_engine.py).
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tinyllm.generate import sample_next
from tinyllm.kv_cache import KVCache, SlotKVCache
from tinyllm.model import Transformer


@dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int
    eos_id: int | None = None


@dataclass
class Result:
    tokens: list[int]          # generated tokens only (EOS included if it was produced)
    ttft_s: float              # engine start → first generated token
    latency_s: float           # engine start → last generated token


@dataclass
class Stats:
    wall_s: float = 0.0
    useful_tokens: int = 0     # generated tokens that belong to a request
    decode_steps: int = 0
    computed_rows: int = 0     # rows run through decode (incl. finished rows / free slots)
    useful_rows: int = 0       # rows whose output was kept
    kv_reserved_tokens: int = 0            # KV capacity held (max over the run)
    kv_used_frac: list[float] = field(default_factory=list)  # per decode step: real tokens / capacity
    kv_alloc_frac: list[float] = field(default_factory=list)  # per decode step: real tokens / claimed
    kv_bytes_per_token: int = 0
    preemptions: int = 0
    peak_active: int = 0
    kv_allocated_at_end: int = 0           # leak check: must be 0 once every request is done

    def summary(self, results: list[Result]) -> dict:
        lat = sorted(r.latency_s for r in results)
        ttft = sorted(r.ttft_s for r in results)
        p95 = lambda xs: xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]
        return {
            "requests": len(results),
            "wall_s": self.wall_s,
            "useful_tokens": self.useful_tokens,
            "tok_s": self.useful_tokens / self.wall_s,
            "decode_steps": self.decode_steps,
            "row_util": self.useful_rows / max(1, self.computed_rows),
            "latency_mean_s": sum(lat) / len(lat),
            "latency_p95_s": p95(lat),
            "ttft_mean_s": sum(ttft) / len(ttft),
            "ttft_p95_s": p95(ttft),
            "kv_reserved_mib": self.kv_reserved_tokens * self.kv_bytes_per_token / 2**20,
            "kv_used_mean": sum(self.kv_used_frac) / max(1, len(self.kv_used_frac)),
            "kv_used_peak": max(self.kv_used_frac, default=0.0),
            "kv_alloc_used_mean": (sum(self.kv_alloc_frac) / len(self.kv_alloc_frac)
                                   if self.kv_alloc_frac else None),
            "preemptions": self.preemptions,
            "peak_active": self.peak_active,
            "kv_allocated_at_end": self.kv_allocated_at_end,
        }


def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _kv_bytes_per_token(model: Transformer, device: torch.device) -> int:
    elem = 2 if device.type == "cuda" else 4          # bf16 under autocast, else fp32
    return sum(layer.self_attn.cache_bytes_per_token(elem) for layer in model.layers)


@torch.no_grad()
def prefill_padded(model: Transformer, prompts: list[list[int]], max_len: int,
                   pad_id: int = 0) -> tuple[Tensor, KVCache, list[int]]:
    """Left-pad prompts to one length, run one prefill. Returns last-position logits (B, V),
    the filled cache (pos = padded length) and each row's pad count."""
    device = next(model.parameters()).device
    L = max(len(p) for p in prompts)
    pad = [L - len(p) for p in prompts]
    idx = torch.tensor([[pad_id] * n + p for n, p in zip(pad, prompts)], device=device)
    cache = KVCache(model.cfg.n_layers, max_len, pad_lens=torch.tensor(pad, device=device))
    with _autocast(device):
        logits, _ = model(idx, cache=cache)
    return logits[:, -1], cache, pad


def _done(req: Request, toks: list[int]) -> bool:
    return len(toks) >= req.max_new_tokens or (req.eos_id is not None and toks[-1] == req.eos_id)


@torch.no_grad()
def run_static(model: Transformer, requests: list[Request], batch_size: int,
               temperature: float = 0.0, top_k=None, top_p=None, generator=None,
               ) -> tuple[list[Result], dict]:
    model.eval()
    device = next(model.parameters()).device
    pick = dict(temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
    stats = Stats(kv_bytes_per_token=_kv_bytes_per_token(model, device))
    results: list[Result | None] = [None] * len(requests)
    t0 = time.perf_counter()
    for b0 in range(0, len(requests), batch_size):
        batch = requests[b0:b0 + batch_size]
        L = max(len(r.prompt) for r in batch)
        new_max = max(r.max_new_tokens for r in batch)
        if L + new_max > model.cfg.max_seq_len:
            raise ValueError(f"padded prompt {L} + new {new_max} exceeds max_seq_len")
        logits, cache, _ = prefill_padded(model, [r.prompt for r in batch], L + new_max)
        capacity = len(batch) * (L + new_max)
        stats.kv_reserved_tokens = max(stats.kv_reserved_tokens, capacity)
        outs: list[list[int]] = [[] for _ in batch]
        ttft = [0.0] * len(batch)
        done = [False] * len(batch)
        while True:
            nxt = sample_next(logits, **pick)
            toks = nxt.squeeze(-1).tolist()                      # GPU sync: tokens to the host
            now = time.perf_counter() - t0
            for j, (r, t) in enumerate(zip(batch, toks)):
                if done[j]:
                    continue
                outs[j].append(t)
                if len(outs[j]) == 1:
                    ttft[j] = now
                if _done(r, outs[j]):
                    done[j] = True
                    results[b0 + j] = Result(outs[j], ttft[j], now)
            if all(done):
                break
            live = [j for j in range(len(batch)) if not done[j]]
            stats.decode_steps += 1
            stats.computed_rows += len(batch)
            stats.useful_rows += len(live)
            stats.kv_used_frac.append(
                sum(len(batch[j].prompt) + len(outs[j]) for j in live) / capacity)
            with _autocast(device):
                logits, _ = model(nxt, cache=cache)
            logits = logits[:, -1]
    stats.wall_s = time.perf_counter() - t0
    stats.useful_tokens = sum(len(r.tokens) for r in results)
    return results, stats.summary(results)


@torch.no_grad()
def run_continuous(model: Transformer, requests: list[Request], n_slots: int,
                   max_len: int | None = None, temperature: float = 0.0, top_k=None,
                   top_p=None, generator=None, kv: str = "slots", block_size: int = 16,
                   kv_blocks: int | None = None, cuda_graph=False,
                   buckets: tuple[int, ...] | None = None) -> tuple[list[Result], dict]:
    """kv="slots": SlotKVCache, every slot reserves max_len positions (4.4).
    kv="paged": PagedKVCache with kv_blocks blocks of block_size tokens (4.5). Default
    kv_blocks = the same memory as the slot cache. Sequences beyond what the blocks hold are
    preempted and recomputed later.
    cuda_graph (4.11b, CUDA, kv="slots" only): the decode step is replayed as a CUDA graph per
    attention-window bucket (cuda_graph.SlotGraphDecoder); prefill stays eager. True builds a
    decoder for this call; pass a SlotGraphDecoder to reuse its cache and captured graphs
    across calls (benchmarks, a server)."""
    model.eval()
    cfg = model.cfg
    device = next(model.parameters()).device
    max_len = max_len or cfg.max_seq_len
    pick = dict(temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if cuda_graph is not False and kv != "slots":
        raise ValueError("cuda_graph supports kv='slots' only (paged: not yet)")
    decoder = None
    if cuda_graph is not False and not isinstance(cuda_graph, bool):   # a SlotGraphDecoder
        decoder, slots = cuda_graph, cuda_graph.slots
        if slots.n_slots != n_slots or slots.max_len != max_len or any(slots.active):
            raise ValueError(f"decoder cache has {slots.n_slots} slots of {slots.max_len} "
                             f"(active: {sum(slots.active)}); need {n_slots} idle slots of {max_len}")
    elif kv == "slots":
        slots = SlotKVCache(cfg.n_layers, n_slots, cfg.n_kv_heads, cfg.head_dim, max_len, dtype, device)
    elif kv == "paged":
        from tinyllm.paged_kv import PagedKVCache
        n_blocks = kv_blocks or n_slots * math.ceil(max_len / block_size)
        slots = PagedKVCache(cfg.n_layers, n_slots, cfg.n_kv_heads, cfg.head_dim, n_blocks,
                             block_size, max_len, dtype, device)
    else:
        raise ValueError(f"kv must be 'slots' or 'paged', got {kv!r}")
    if cuda_graph is True:
        from tinyllm.cuda_graph import SlotGraphDecoder
        decoder = SlotGraphDecoder(model, slots, buckets=buckets)
    stats = Stats(kv_bytes_per_token=_kv_bytes_per_token(model, device),
                  kv_reserved_tokens=slots.reserved_tokens())
    results: list[Result | None] = [None] * len(requests)
    outs: dict[int, list[int]] = {}
    ttft: dict[int, float] = {}
    slot_req: list[int | None] = [None] * n_slots
    last_tok = torch.zeros(n_slots, 1, dtype=torch.long, device=device)
    queue = deque(range(len(requests)))
    t0 = time.perf_counter()

    def context(rid: int) -> list[int]:
        """Prompt plus tokens generated so far (non-empty only after a preemption)."""
        return requests[rid].prompt + outs.get(rid, [])

    def record(s: int, rid: int, tok: int, now: float) -> None:
        outs[rid].append(tok)
        if len(outs[rid]) == 1:
            ttft[rid] = now
        if _done(requests[rid], outs[rid]):
            results[rid] = Result(outs[rid], ttft[rid], now)
            slot_req[s] = None
            slots.release(s)

    def left(rid: int) -> int:
        return requests[rid].max_new_tokens - len(outs.get(rid, []))

    def fits_padded(rids: list[int]) -> bool:
        """Every row of a padded prefill sits at positions up to L + its remaining tokens, so
        that must fit max_len (the RoPE table, and slot storage). A preempted request comes back
        with a longer context, so this can fail for a batch even when each request fits alone."""
        return max(len(context(r)) for r in rids) + max(left(r) for r in rids) <= max_len

    while queue or any(slots.active):
        free = slots.free_slots()
        rids: list[int] = []
        if free and queue:                                    # admit newcomers: one padded prefill
            if not fits_padded([queue[0]]):
                r = queue[0]
                raise ValueError(f"request {r}: prompt {len(context(r))} + new {left(r)} "
                                 f"exceeds slot max_len {max_len}")
            while (queue and len(rids) < len(free) and fits_padded(rids + [queue[0]])
                   and slots.can_admit([len(context(r)) for r in rids + [queue[0]]])):
                rids.append(queue.popleft())
            if not rids and not any(slots.active):
                raise ValueError(f"request {queue[0]} does not fit in the KV cache even alone")
        if rids:
            prompts = [context(r) for r in rids]
            L = max(len(p) for p in prompts)
            logits, pre, pad = prefill_padded(model, prompts, max_len=L)
            nxt = sample_next(logits, **pick)
            taken = free[:len(rids)]
            slots.admit(taken, pre, pad)
            last_tok[torch.tensor(taken, device=device)] = nxt
            toks = nxt.squeeze(-1).tolist()
            now = time.perf_counter() - t0
            for s, rid, t in zip(taken, rids, toks):
                slot_req[s] = rid
                outs.setdefault(rid, [])
                record(s, rid, t, now)
        for s in slots.reserve_step():                        # paged: out of blocks → preempt newest
            queue.appendleft(slot_req[s])                     # recompute later from its context
            slot_req[s] = None
            stats.preemptions += 1
        if not any(slots.active):
            continue
        live = [s for s in range(n_slots) if slots.active[s]]
        stats.peak_active = max(stats.peak_active, len(live))
        stats.decode_steps += 1
        stats.computed_rows += n_slots
        stats.useful_rows += len(live)
        used = slots.used_tokens()
        stats.kv_used_frac.append(used / slots.reserved_tokens())
        stats.kv_alloc_frac.append(used / max(1, slots.allocated_tokens()))
        if decoder is not None:
            logits = decoder.step(last_tok)                   # static buffer: sample at once
        else:
            with _autocast(device):
                logits, _ = model(last_tok, cache=slots)      # one token per slot
        last_tok = sample_next(logits[:, -1], **pick)
        toks = last_tok.squeeze(-1).tolist()
        now = time.perf_counter() - t0
        for s in live:
            record(s, slot_req[s], toks[s], now)
    stats.kv_allocated_at_end = slots.allocated_tokens()
    stats.wall_s = time.perf_counter() - t0
    stats.useful_tokens = sum(len(r.tokens) for r in results)
    return results, stats.summary(results)
