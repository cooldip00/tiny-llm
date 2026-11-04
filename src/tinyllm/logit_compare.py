"""Compare next-token logits between models (step 4.9c; reused by 4.8b).

Logits are collected the way inference produces them: a prefill over the first `n_prefill`
tokens, then one token per step through the KV cache (teacher-forced, so every model sees
the same sequence). Nothing here imports transformers: an HF model is passed in and only
called as `hf(input_ids, past_key_values=..., use_cache=True)` (D-015: test reference only).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from tinyllm.kv_cache import KVCache


@torch.no_grad()
def ours_stepwise(model, ids: Tensor, n_prefill: int) -> Tensor:
    """(B, T, V) fp32 CPU logits: prefill ids[:, :n_prefill], then one decode step per token."""
    cache = KVCache(model.cfg.n_layers, model.cfg.max_seq_len)
    autocast = dict(device_type=ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda")
    out = []
    with torch.autocast(**autocast):                      # same setting as generate()
        out.append(model(ids[:, :n_prefill], cache=cache)[0].float().cpu())
        for t in range(n_prefill, ids.shape[1]):
            out.append(model(ids[:, t:t + 1], cache=cache)[0].float().cpu())
    return torch.cat(out, dim=1)


@torch.no_grad()
def hf_stepwise(hf, ids: Tensor, n_prefill: int) -> Tensor:
    """Same as ours_stepwise for an HF causal LM, using its own KV cache."""
    o = hf(ids[:, :n_prefill], use_cache=True)
    out, pkv = [o.logits.float().cpu()], o.past_key_values
    for t in range(n_prefill, ids.shape[1]):
        o = hf(ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
        out.append(o.logits.float().cpu())
        pkv = o.past_key_values
    return torch.cat(out, dim=1)


def logit_stats(ref: Tensor, x: Tensor) -> dict[str, float]:
    """x against a reference, over all positions: abs error, KL(ref || x) in nats, top-1 agreement."""
    ref, x = ref.float().flatten(0, -2), x.float().flatten(0, -2)
    d = (x - ref).abs()
    lp_ref, lp_x = F.log_softmax(ref, -1), F.log_softmax(x, -1)
    kl = (lp_ref.exp() * (lp_ref - lp_x)).sum(-1)
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "top1": (ref.argmax(-1) == x.argmax(-1)).float().mean().item(),
        "positions": ref.shape[0],
    }


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token over the common length, or None if that prefix is equal."""
    for i, (p, q) in enumerate(zip(a, b)):
        if p != q:
            return i
    return None
