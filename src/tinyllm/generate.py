"""Text sampling. For now this is the simple version used for samples during training:
a full forward pass per new token, with no KV cache. The real inference engine
(KV cache, batching, benchmarks) is built here in Phase 4.
"""

import torch
from torch import Tensor

from tinyllm.model import Transformer


@torch.no_grad()
def sample(
    model: Transformer,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Append up to max_new_tokens to idx (1, T). Stops early when eos_id is sampled.

    temperature 0 means greedy (argmax). top_k keeps only the k most likely tokens.
    """
    was_training = model.training
    model.eval()
    max_len = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        context = idx[:, -max_len:]  # no cache: the window slides once the context is full
        with torch.autocast(idx.device.type, dtype=torch.bfloat16, enabled=idx.device.type == "cuda"):
            logits, _ = model(context)
        logits = logits[:, -1, :].float()
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    model.train(was_training)
    return idx
