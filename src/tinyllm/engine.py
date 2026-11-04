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
    kv_bytes_per_token: int = 0

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
                   top_p=None, generator=None) -> tuple[list[Result], dict]:
    model.eval()
    cfg = model.cfg
    device = next(model.parameters()).device
    max_len = max_len or cfg.max_seq_len
    pick = dict(temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    slots = SlotKVCache(cfg.n_layers, n_slots, cfg.n_kv_heads, cfg.head_dim, max_len, dtype, device)
    stats = Stats(kv_bytes_per_token=_kv_bytes_per_token(model, device),
                  kv_reserved_tokens=slots.reserved_tokens())
    results: list[Result | None] = [None] * len(requests)
    outs: dict[int, list[int]] = {}
    ttft: dict[int, float] = {}
    slot_req: list[int | None] = [None] * n_slots
    last_tok = torch.zeros(n_slots, 1, dtype=torch.long, device=device)
    queue = deque(range(len(requests)))
    t0 = time.perf_counter()

    def record(s: int, rid: int, tok: int, now: float) -> None:
        outs[rid].append(tok)
        if len(outs[rid]) == 1:
            ttft[rid] = now
        if _done(requests[rid], outs[rid]):
            results[rid] = Result(outs[rid], ttft[rid], now)
            slot_req[s] = None
            slots.release(s)

    while queue or any(slots.active):
        free = slots.free_slots()
        if free and queue:                                    # admit newcomers: one padded prefill
            rids = [queue.popleft() for _ in range(min(len(free), len(queue)))]
            prompts = [requests[r].prompt for r in rids]
            L = max(len(p) for p in prompts)
            for r in rids:
                if L + requests[r].max_new_tokens > max_len:
                    raise ValueError(f"request {r}: padded prompt {L} + new "
                                     f"{requests[r].max_new_tokens} exceeds slot max_len {max_len}")
            logits, pre, pad = prefill_padded(model, prompts, max_len=L)
            nxt = sample_next(logits, **pick)
            taken = free[:len(rids)]
            slots.admit(taken, pre, pad)
            last_tok[torch.tensor(taken, device=device)] = nxt
            toks = nxt.squeeze(-1).tolist()
            now = time.perf_counter() - t0
            for s, rid, t in zip(taken, rids, toks):
                slot_req[s] = rid
                outs[rid] = []
                record(s, rid, t, now)
        if not any(slots.active):
            continue
        live = [s for s in range(n_slots) if slots.active[s]]
        stats.decode_steps += 1
        stats.computed_rows += n_slots
        stats.useful_rows += len(live)
        stats.kv_used_frac.append(slots.used_tokens() / slots.reserved_tokens())
        with _autocast(device):
            logits, _ = model(last_tok, cache=slots)          # one token per slot
        last_tok = sample_next(logits[:, -1], **pick)
        toks = last_tok.squeeze(-1).tolist()
        now = time.perf_counter() - t0
        for s in live:
            record(s, slot_req[s], toks[s], now)
    stats.wall_s = time.perf_counter() - t0
    stats.useful_tokens = sum(len(r.tokens) for r in results)
    return results, stats.summary(results)
