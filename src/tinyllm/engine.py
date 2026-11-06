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


@dataclass
class Event:
    """One generated token, as step() reports it (a server streams these)."""
    rid: int
    token: int
    done: bool                 # the request finished with this token (EOS or max_new_tokens)


class ContinuousEngine:
    """Step 4.13: run_continuous's loop as an object, so requests can arrive while others
    decode. add_request() queues a request at any time; each step() admits queued requests
    into free slots (one padded prefill), decodes one token for every active slot and
    returns the new tokens as Events. A finished request's Result is in `results` (pop it).

    Times are seconds on the engine clock (since construction). TTFT and latency count from
    the request's arrival, so with every request added at once they match 4.4's numbers.
    Sampling settings are per engine. Not thread-safe: call add_request / step / abort from
    one thread.

    kv="slots": SlotKVCache, every slot reserves max_len positions (4.4).
    kv="paged": PagedKVCache with kv_blocks blocks of block_size tokens (4.5). Default
    kv_blocks = the same memory as the slot cache. Sequences beyond what the blocks hold are
    preempted and recomputed later.
    cuda_graph (4.11b, CUDA, kv="slots" only): the decode step is replayed as a CUDA graph per
    attention-window bucket (cuda_graph.SlotGraphDecoder); prefill stays eager. True builds a
    decoder for this engine; pass a SlotGraphDecoder to reuse its cache and captured graphs.
    """

    def __init__(self, model: Transformer, n_slots: int, max_len: int | None = None,
                 temperature: float = 0.0, top_k=None, top_p=None, generator=None,
                 kv: str = "slots", block_size: int = 16, kv_blocks: int | None = None,
                 cuda_graph=False, buckets: tuple[int, ...] | None = None):
        model.eval()
        cfg = model.cfg
        self.model = model
        self.device = device = next(model.parameters()).device
        self.n_slots = n_slots
        self.max_len = max_len = max_len or cfg.max_seq_len
        self.pick = dict(temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if cuda_graph is not False and kv != "slots":
            raise ValueError("cuda_graph supports kv='slots' only (paged: not yet)")
        self.decoder = None
        if cuda_graph is not False and not isinstance(cuda_graph, bool):   # a SlotGraphDecoder
            self.decoder, slots = cuda_graph, cuda_graph.slots
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
            self.decoder = SlotGraphDecoder(model, slots, buckets=buckets)
        self.slots = slots
        self.stats = Stats(kv_bytes_per_token=_kv_bytes_per_token(model, device),
                           kv_reserved_tokens=slots.reserved_tokens())
        self.results: dict[int, Result] = {}              # finished, until the caller pops them
        self._req: dict[int, Request] = {}                # unfinished requests only
        self._arrival: dict[int, float] = {}
        self._outs: dict[int, list[int]] = {}
        self._ttft: dict[int, float] = {}
        self._slot_req: list[int | None] = [None] * n_slots
        self._last_tok = torch.zeros(n_slots, 1, dtype=torch.long, device=device)
        self._queue: deque[int] = deque()
        self._next_id = 0
        self._t0 = time.perf_counter()

    # -- public API ------------------------------------------------------------
    def now(self) -> float:
        return time.perf_counter() - self._t0

    def add_request(self, req: Request) -> int:
        """Queue a request; returns its id. Raises at once if it can never fit a slot."""
        if not req.prompt or req.max_new_tokens < 1:
            raise ValueError("a request needs a non-empty prompt and max_new_tokens >= 1")
        if len(req.prompt) + req.max_new_tokens > self.max_len:
            raise ValueError(f"prompt {len(req.prompt)} + new {req.max_new_tokens} "
                             f"exceeds slot max_len {self.max_len}")
        rid, self._next_id = self._next_id, self._next_id + 1
        self._req[rid] = req
        self._arrival[rid] = self.now()
        self._queue.append(rid)
        return rid

    def has_work(self) -> bool:
        return bool(self._queue) or any(self.slots.active)

    def abort(self, rid: int) -> bool:
        """Drop an unfinished request (e.g. its client went away), freeing its slot.
        Returns False if it is unknown or already finished. No Result is kept."""
        if rid not in self._req:
            return False
        if rid in self._queue:
            self._queue.remove(rid)
        if rid in self._slot_req:
            s = self._slot_req.index(rid)
            self._slot_req[s] = None
            self.slots.release(s)
        self._forget(rid)
        return True

    @torch.no_grad()
    def step(self) -> list[Event]:
        """Admit what fits, decode one token per active slot. Returns the new tokens."""
        events: list[Event] = []
        slots = self.slots
        free = slots.free_slots()
        rids: list[int] = []
        if free and self._queue:                              # admit newcomers: one padded prefill
            q = self._queue
            if not self._fits_padded([q[0]]):
                r = q[0]
                raise ValueError(f"request {r}: prompt {len(self._context(r))} + new {self._left(r)} "
                                 f"exceeds slot max_len {self.max_len}")
            while (q and len(rids) < len(free) and self._fits_padded(rids + [q[0]])
                   and slots.can_admit([len(self._context(r)) for r in rids + [q[0]]])):
                rids.append(q.popleft())
            if not rids and not any(slots.active):
                raise ValueError(f"request {q[0]} does not fit in the KV cache even alone")
        if rids:
            prompts = [self._context(r) for r in rids]
            L = max(len(p) for p in prompts)
            logits, pre, pad = prefill_padded(self.model, prompts, max_len=L)
            nxt = sample_next(logits, **self.pick)
            taken = free[:len(rids)]
            slots.admit(taken, pre, pad)
            self._last_tok[torch.tensor(taken, device=self.device)] = nxt
            toks = nxt.squeeze(-1).tolist()
            now = self.now()
            for s, rid, t in zip(taken, rids, toks):
                self._slot_req[s] = rid
                self._outs.setdefault(rid, [])
                events.append(self._record(s, rid, t, now))
        for s in slots.reserve_step():                        # paged: out of blocks → preempt newest
            self._queue.appendleft(self._slot_req[s])         # recompute later from its context
            self._slot_req[s] = None
            self.stats.preemptions += 1
        if not any(slots.active):
            return events
        st = self.stats
        live = [s for s in range(self.n_slots) if slots.active[s]]
        st.peak_active = max(st.peak_active, len(live))
        st.decode_steps += 1
        st.computed_rows += self.n_slots
        st.useful_rows += len(live)
        used = slots.used_tokens()
        st.kv_used_frac.append(used / slots.reserved_tokens())
        st.kv_alloc_frac.append(used / max(1, slots.allocated_tokens()))
        if self.decoder is not None:
            logits = self.decoder.step(self._last_tok)        # static buffer: sample at once
        else:
            with _autocast(self.device):
                logits, _ = self.model(self._last_tok, cache=slots)   # one token per slot
        self._last_tok = sample_next(logits[:, -1], **self.pick)
        toks = self._last_tok.squeeze(-1).tolist()
        now = self.now()
        for s in live:
            events.append(self._record(s, self._slot_req[s], toks[s], now))
        return events

    # -- internals -------------------------------------------------------------
    def _context(self, rid: int) -> list[int]:
        """Prompt plus tokens generated so far (non-empty only after a preemption)."""
        return self._req[rid].prompt + self._outs.get(rid, [])

    def _left(self, rid: int) -> int:
        return self._req[rid].max_new_tokens - len(self._outs.get(rid, []))

    def _fits_padded(self, rids: list[int]) -> bool:
        """Every row of a padded prefill sits at positions up to L + its remaining tokens, so
        that must fit max_len (the RoPE table, and slot storage). A preempted request comes back
        with a longer context, so this can fail for a batch even when each request fits alone."""
        return max(len(self._context(r)) for r in rids) + max(self._left(r) for r in rids) <= self.max_len

    def _record(self, s: int, rid: int, tok: int, now: float) -> Event:
        outs = self._outs[rid]
        outs.append(tok)
        if len(outs) == 1:
            self._ttft[rid] = now
        done = _done(self._req[rid], outs)
        if done:
            t_in = self._arrival[rid]
            self.results[rid] = Result(outs, self._ttft[rid] - t_in, now - t_in)
            self.stats.useful_tokens += len(outs)
            self._slot_req[s] = None
            self.slots.release(s)
            self._forget(rid)
        return Event(rid, tok, done)

    def _forget(self, rid: int) -> None:
        for d in (self._req, self._arrival, self._outs, self._ttft):
            d.pop(rid, None)


@torch.no_grad()
def run_continuous(model: Transformer, requests: list[Request], n_slots: int,
                   max_len: int | None = None, temperature: float = 0.0, top_k=None,
                   top_p=None, generator=None, kv: str = "slots", block_size: int = 16,
                   kv_blocks: int | None = None, cuda_graph=False,
                   buckets: tuple[int, ...] | None = None) -> tuple[list[Result], dict]:
    """Offline continuous batching (4.4): every request queued at t=0, run to completion.
    A thin wrapper over ContinuousEngine; see it for kv / cuda_graph."""
    eng = ContinuousEngine(model, n_slots, max_len, temperature, top_k, top_p, generator,
                           kv, block_size, kv_blocks, cuda_graph, buckets)
    t0 = time.perf_counter()
    rids = [eng.add_request(r) for r in requests]
    while eng.has_work():
        eng.step()
    st = eng.stats
    st.kv_allocated_at_end = eng.slots.allocated_tokens()
    st.wall_s = time.perf_counter() - t0
    results = [eng.results.pop(r) for r in rids]
    return results, st.summary(results)
