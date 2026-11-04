"""Step 4.9a: grouped-query attention."""

import dataclasses
import math

import pytest
import torch

from test_engine import CFG as ENGINE_CFG, make_requests, solo
from tinyllm.attention import build_attention
from tinyllm.attention.gqa import GQA, repeat_kv
from tinyllm.attention.mha import MHA
from tinyllm.config import get_config
from tinyllm.engine import run_continuous
from tinyllm.generate import generate, sample
from tinyllm.kv_cache import KVCache
from tinyllm.model import RotaryEmbedding, Transformer, apply_rope
from tinyllm.quant import quantize_model

CFG = get_config("smoke", n_kv_heads=2)          # 6 query heads, 2 KV heads: groups of 3
B, T = 2, 16


@pytest.fixture(scope="module")
def rope():
    return RotaryEmbedding(CFG.head_dim, CFG.max_seq_len, CFG.rope_theta)(T)


def seeded(cls, cfg):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return cls(cfg, 0)


def x_in():
    return torch.randn(B, T, CFG.d_model, generator=torch.Generator().manual_seed(1))


def per_head_reference(attn: GQA, x, cos, sin):
    """Textbook GQA: a loop over query heads, head i reading KV head i // g, explicit causal softmax."""
    h, kv, hd, g = attn.n_heads, attn.n_kv_heads, attn.head_dim, attn.groups
    q = apply_rope(attn.q_proj(x).view(B, T, h, hd).transpose(1, 2), cos, sin)
    k = apply_rope(attn.k_proj(x).view(B, T, kv, hd).transpose(1, 2), cos, sin)
    v = attn.v_proj(x).view(B, T, kv, hd).transpose(1, 2)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    heads = []
    for i in range(h):
        s = q[:, i] @ k[:, i // g].transpose(-2, -1) / math.sqrt(hd)
        heads.append(s.masked_fill(future, float("-inf")).softmax(-1) @ v[:, i // g])
    y = torch.stack(heads, 1)                                         # (B, h, T, hd)
    return attn.o_proj(y.transpose(1, 2).reshape(B, T, -1))


def test_config_defaults_to_gqa_layers():
    assert CFG.layer_types == ("gqa",) * CFG.n_layers
    assert isinstance(build_attention(CFG, 0), GQA)
    assert get_config("smoke").layer_types == ("mha",) * 6            # unchanged without n_kv_heads


def test_projection_shapes_and_names():
    attn = GQA(CFG, 0)
    shapes = {n: tuple(p.shape) for n, p in attn.named_parameters()}
    d, hd = CFG.d_model, CFG.head_dim
    assert shapes == {"q_proj.weight": (6 * hd, d), "k_proj.weight": (2 * hd, d),
                      "v_proj.weight": (2 * hd, d), "o_proj.weight": (d, 6 * hd)}


def test_repeat_kv_order_matches_hf():
    x = torch.arange(3).float().view(1, 3, 1, 1)                      # KV heads 0, 1, 2
    assert repeat_kv(x, 2).flatten().tolist() == [0, 0, 1, 1, 2, 2]   # query head i -> KV head i // g


def test_matches_per_head_reference(rope):
    attn, x = seeded(GQA, CFG), x_in()
    torch.testing.assert_close(attn(x, rope), per_head_reference(attn, x, *rope), atol=1e-5, rtol=1e-4)


def test_equals_mha_when_kv_heads_equal_heads(rope):
    cfg = get_config("smoke")
    mha = seeded(MHA, cfg)
    gqa = GQA(cfg, 0)
    gqa.load_state_dict(mha.state_dict())                             # same names and shapes
    x = x_in()
    assert torch.equal(gqa(x, rope), mha(x, rope))


def test_causal_future_tokens_do_not_leak(rope):
    attn, x = seeded(GQA, CFG), x_in()
    x2 = x.clone()
    x2[:, 10:] = torch.randn(B, T - 10, CFG.d_model)
    y, y2 = attn(x, rope), attn(x2, rope)
    torch.testing.assert_close(y[:, :10], y2[:, :10])
    assert not torch.allclose(y[:, 10:], y2[:, 10:])


def test_cached_chunks_match_full_pass(rope):
    attn, x = seeded(GQA, CFG), x_in()
    cos, sin = rope
    cache = KVCache(n_layers=1, max_len=T)
    rope_mod = RotaryEmbedding(CFG.head_dim, CFG.max_seq_len, CFG.rope_theta)
    outs = []
    for n in (6, 1, 5, 4):
        s = cache.pos
        cache.begin_step(n, rope_mod)
        outs.append(attn(x[:, s:s + n], (cos[s:s + n], sin[s:s + n]), cache.layer(0)))
        cache.end_step(n)
    assert cache.k[0].shape[1] == CFG.n_kv_heads                     # the cache stores KV heads only
    torch.testing.assert_close(torch.cat(outs, dim=1), attn(x, rope), atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_cuda_matches_reference(rope):
    # The fused CUDA kernels with enable_gqa, against the fp32 per-head loop.
    attn, x = seeded(GQA, CFG), x_in()
    ref = per_head_reference(attn, x, *rope)
    attn, x, cos, sin = attn.cuda(), x.cuda(), *(t.cuda() for t in rope)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = attn(x, (cos, sin))
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)


def test_cache_bytes_and_params():
    # main with 2 KV heads: 2 (K,V) × 2 heads × 64 dims × 2 bytes = 512 B per layer, 1/4 of mha's 2048
    assert GQA(get_config("main", n_kv_heads=2), 0).cache_bytes_per_token() == 512
    m = Transformer(CFG)
    assert m.num_params() == CFG.estimate_params()


def test_training_step_backprops():
    cfg = dataclasses.replace(ENGINE_CFG, n_kv_heads=2, layer_types=None)
    m = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 12), generator=torch.Generator().manual_seed(0))
    _, loss = m(idx[:, :-1], idx[:, 1:])
    loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if "k_proj" in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


# --- the whole model with GQA through every decode path ------------------------------------------

@pytest.fixture(scope="module")
def model():
    cfg = dataclasses.replace(ENGINE_CFG, n_kv_heads=2, layer_types=None)   # 4 query heads, 2 KV heads
    assert cfg.layer_types == ("gqa",) * cfg.n_layers
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(cfg).eval()
    with torch.no_grad():                                             # well-separated logits, as in test_engine
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return m


def test_kv_generate_matches_naive(model):
    idx = torch.randint(0, model.cfg.vocab_size, (3, 7), generator=torch.Generator().manual_seed(3))
    assert torch.equal(generate(model, idx, 20, temperature=0), sample(model, idx, 20, temperature=0))


@pytest.mark.parametrize("kv", ["slots", "paged"])
def test_engine_matches_solo(model, kv):
    reqs = make_requests(7)
    results, _ = run_continuous(model, reqs, n_slots=3, kv=kv, block_size=4)
    assert [r.tokens for r in results] == [solo(model, r) for r in reqs]


def test_quantized_gqa_model_runs(model):
    import copy
    q = copy.deepcopy(model)
    assert quantize_model(q) == 7 * model.cfg.n_layers + 1
    idx = torch.randint(0, model.cfg.vocab_size, (2, 5), generator=torch.Generator().manual_seed(4))
    assert torch.equal(generate(q, idx, 10, temperature=0), sample(q, idx, 10, temperature=0))
