"""Text sampling. For now this is the simple version used for samples during training:
a full forward pass per new token, with no KV cache. The real inference engine
(KV cache, batching, benchmarks) is built here in Phase 4.
"""

import torch
from torch import Tensor

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
