"""Step 4.3: KV-cached generation must match the naive reference (docs/INFERENCE.md)."""

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.generate import generate, sample
from tinyllm.kv_cache import KVCache
from tinyllm.model import Transformer

CFG = ModelConfig(vocab_size=128, max_seq_len=48, d_model=64, n_layers=3, n_heads=4, ffn_hidden=160)
TOL = dict(atol=1e-4, rtol=1e-4)  # fp32 on CPU: only op-order differences


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(CFG).eval()
    # Larger init than training's 0.02 so the logits are far from uniform and
    # greedy tokens are well separated (a real test of matching, not of ties).
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return m


def prompt(b=2, t=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, CFG.vocab_size, (b, t), generator=g)


@torch.no_grad()
def test_prefill_logits_match_full_forward(model):
    idx = prompt()
    cache = KVCache(CFG.n_layers, CFG.max_seq_len)
    torch.testing.assert_close(model(idx, cache=cache)[0], model(idx)[0], **TOL)
    assert cache.pos == idx.shape[1]


@torch.no_grad()
def test_decode_steps_match_full_forward(model):
    idx = prompt(t=20)
    cache = KVCache(CFG.n_layers, CFG.max_seq_len)
    model(idx[:, :8], cache=cache)
    for t in range(8, 20):                                   # one token at a time
        step = model(idx[:, t:t + 1], cache=cache)[0][:, -1]
        full = model(idx[:, :t + 1])[0][:, -1]
        torch.testing.assert_close(step, full, **TOL)


@torch.no_grad()
def test_chunked_prefill_matches_full_forward(model):
    idx = prompt(t=23)
    cache = KVCache(CFG.n_layers, CFG.max_seq_len)
    chunks = [model(idx[:, s:s + 5], cache=cache)[0] for s in range(0, 23, 5)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), model(idx)[0], **TOL)


@pytest.mark.parametrize("b", [1, 3])
def test_greedy_generate_equals_naive_sample(model, b):
    idx = prompt(b=b, t=9)
    assert torch.equal(generate(model, idx, 30, temperature=0), sample(model, idx, 30, temperature=0))


def test_sampled_generate_equals_naive_with_same_generator(model):
    # Same logits (to fp32 noise) and one multinomial draw per step in both → same tokens.
    idx = prompt(b=1, t=5)
    kw = dict(temperature=0.8, top_k=50, top_p=0.9)
    a = generate(model, idx, 25, generator=torch.Generator().manual_seed(7), **kw)
    b = sample(model, idx, 25, generator=torch.Generator().manual_seed(7), **kw)
    assert torch.equal(a, b)


def test_eos_stops_when_every_row_is_done(model):
    idx = prompt(b=1, t=6)
    first = generate(model, idx, 1, temperature=0)[0, -1].item()
    assert generate(model, idx, 20, temperature=0, eos_id=first).shape == (1, 7)
    # Two rows: stops only when both have produced EOS (or at max_new_tokens).
    idx2 = prompt(b=2, t=6)
    full = generate(model, idx2, 20, temperature=0)
    eos = full[0, 6].item()                                  # row 0's first token
    out = generate(model, idx2, 20, temperature=0, eos_id=eos)
    hits = (full[1, 6:] == eos).nonzero()
    expect = 20 if len(hits) == 0 else max(1, hits[0].item() + 1)
    assert out.shape == (2, 6 + expect)
    assert torch.equal(out, full[:, :6 + expect])


def test_generate_restores_train_mode():
    m = Transformer(CFG)
    generate(m, prompt(), 3, temperature=0)
    assert m.training


def test_too_long_raises(model):
    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, prompt(t=40), 9, temperature=0)


@torch.no_grad()
def test_cache_full_raises_and_size(model):
    cache = KVCache(CFG.n_layers, max_len=10)
    model(prompt(b=2, t=10), cache=cache)
    # 3 layers × (K + V) × [2, 4 heads, 10, 16] × 4 bytes (fp32)
    assert cache.nbytes() == CFG.n_layers * 2 * (2 * 4 * 10 * 16) * 4
    with pytest.raises(ValueError, match="full"):
        model(prompt(b=2, t=1), cache=cache)


@torch.no_grad()
def test_cache_is_freed_without_the_cycle_collector(model):
    # A cache ↔ layer reference cycle once kept every finished cache alive until gc ran
    # (seen as ~8 caches' worth of extra peak VRAM in the 4.3 benchmark).
    import gc
    import weakref
    gc.disable()
    try:
        cache = KVCache(CFG.n_layers, CFG.max_seq_len)
        model(prompt(), cache=cache)
        ref = weakref.ref(cache.k[0])
        del cache
        assert ref() is None                                 # freed by refcount alone
    finally:
        gc.enable()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.no_grad()
def test_cuda_bf16_decode_logits_close(model):
    m = Transformer(CFG).cuda().eval()
    m.load_state_dict(model.state_dict())
    idx = prompt(t=30).cuda()
    cache = KVCache(CFG.n_layers, CFG.max_seq_len)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(idx[:, :10], cache=cache)
        steps = torch.cat([m(idx[:, t:t + 1], cache=cache)[0] for t in range(10, 30)], dim=1)
        full = m(idx)[0][:, 10:]
    assert cache.k[0].dtype == torch.bfloat16
    err = (steps.float() - full.float()).abs().max().item()
    scale = full.float().abs().max().item()
    assert err <= 0.02 * scale, (err, scale)                 # bf16: ~3 significant digits
