"""Text generation.

- sample(): naive, a full forward pass over the context per new token (no cache).
  Used for training samples, and the reference that every faster path must match (4.1).
- generate(): KV-cached, one prefill pass then one new token per forward (4.3).
Both choose tokens with sample_next (4.2).
"""

import torch
from torch import Tensor

from tinyllm.kv_cache import KVCache
from tinyllm.model import Transformer


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pick the next token from last-position logits (B, V) → (B, 1). Step 4.2.

    Order (same as HF's warpers): temperature, then top-k, then top-p.
    - temperature 0 means greedy (argmax); top_k / top_p are then irrelevant.
    - top_k keeps the k most likely tokens.
    - top_p (nucleus) keeps the smallest set of most likely tokens whose total
      probability is >= top_p. The most likely token is always kept.
    """
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = logits.sort(-1, descending=True)
        probs = sorted_logits.softmax(-1)
        mass_before = probs.cumsum(-1) - probs           # probability of all strictly better tokens
        drop_sorted = mass_before >= top_p               # first token has mass_before 0: always kept
        drop = drop_sorted.scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1, generator=generator)


@torch.no_grad()
def sample(
    model: Transformer,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    top_p: float | None = None,
) -> Tensor:
    """Append up to max_new_tokens to idx (1, T). Stops early when eos_id is sampled.

    Token choice is sample_next (temperature, top_k, top_p).
    """
    was_training = model.training
    model.eval()
    max_len = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        context = idx[:, -max_len:]  # no cache: the window slides once the context is full
        with torch.autocast(idx.device.type, dtype=torch.bfloat16, enabled=idx.device.type == "cuda"):
            logits, _ = model(context)
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    model.train(was_training)
    return idx


@torch.no_grad()
def generate(
    model: Transformer,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """KV-cached generation. Append up to max_new_tokens to idx (B, T).

    Prefill runs the prompt once and fills the cache; each decode step feeds only the
    newest token. Stops early once every row has produced eos_id (rows that finish
    first keep generating; the caller trims after EOS).

    Unlike sample(), the context can't slide: RoPE positions are baked into the cached
    keys. So prompt + max_new_tokens must fit in max_seq_len.
    """
    B, T = idx.shape
    max_len = model.cfg.max_seq_len
    if T + max_new_tokens > max_len:
        raise ValueError(f"prompt {T} + new {max_new_tokens} exceeds max_seq_len {max_len}")
    was_training = model.training
    model.eval()
    cache = KVCache(model.cfg.n_layers, max_len)
    autocast = dict(device_type=idx.device.type, dtype=torch.bfloat16,
                    enabled=idx.device.type == "cuda")
    done = torch.zeros(B, dtype=torch.bool, device=idx.device)
    out = [idx]
    with torch.autocast(**autocast):
        logits, _ = model(idx, cache=cache)                    # prefill
    for i in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        out.append(nxt)
        if eos_id is not None:
            done |= nxt.squeeze(-1) == eos_id
            if done.all():
                break
        if i + 1 == max_new_tokens:
            break                                              # no forward for a token we won't use
        with torch.autocast(**autocast):
            logits, _ = model(nxt, cache=cache)                # decode: one token
    model.train(was_training)
    return torch.cat(out, dim=1)
