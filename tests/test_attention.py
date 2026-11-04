import math

import pytest
import torch

from tinyllm.attention import ATTENTION, Attention, build_attention, register
from tinyllm.attention.mha import MHA
from tinyllm.config import get_config
from tinyllm.model import RotaryEmbedding, apply_rope

torch.manual_seed(0)
CFG = get_config("smoke")
B, T = 2, 16


@pytest.fixture(scope="module")
def rope_tables():
    return RotaryEmbedding(CFG.head_dim, CFG.max_seq_len, CFG.rope_theta)(T)


def naive_attention(attn: MHA, x, cos, sin):
    """Textbook version: explicit T×T scores, causal mask, softmax."""
    h, hd = attn.n_heads, attn.head_dim
    q = attn.q_proj(x).view(B, T, h, hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, h, hd).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, h, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    y = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
    return attn.o_proj(y.transpose(1, 2).reshape(B, T, -1))


def test_registry_builds_mha():
    assert "mha" in ATTENTION
    assert isinstance(build_attention(CFG, 0), MHA)


def test_unknown_variant_raises():
    cfg = get_config("smoke", layer_types=["nope"] * CFG.n_layers)
    with pytest.raises(KeyError, match="registered"):
        build_attention(cfg, 0)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):
        register("mha")(type("Dup", (Attention,), {}))


def test_output_shape(rope_tables):
    x = torch.randn(B, T, CFG.d_model)
    assert MHA(CFG, 0)(x, rope_tables).shape == (B, T, CFG.d_model)


def test_matches_naive_attention(rope_tables):
    attn = MHA(CFG, 0)
    x = torch.randn(B, T, CFG.d_model)
    torch.testing.assert_close(attn(x, rope_tables), naive_attention(attn, x, *rope_tables),
                               atol=1e-5, rtol=1e-4)


def test_causal_future_tokens_do_not_leak(rope_tables):
    attn = MHA(CFG, 0)
    x = torch.randn(B, T, CFG.d_model)
    x2 = x.clone()
    x2[:, 10:] = torch.randn(B, T - 10, CFG.d_model)  # change positions 10..15
    y, y2 = attn(x, rope_tables), attn(x2, rope_tables)
    torch.testing.assert_close(y[:, :10], y2[:, :10])        # past unaffected
    assert not torch.allclose(y[:, 10:], y2[:, 10:])         # future did change


def test_hf_names_and_no_bias():
    names = set(dict(MHA(CFG, 0).named_parameters()))
    assert names == {f"{p}_proj.weight" for p in "qkvo"}


def test_rejects_gqa_config():
    with pytest.raises(ValueError, match="gqa"):
        MHA(get_config("smoke", n_kv_heads=2), 0)


def test_cache_bytes_per_token():
    # main: 2 (K,V) × 8 heads × 64 dims × 2 bytes (bf16) = 2048 bytes per layer
    assert MHA(get_config("main"), 0).cache_bytes_per_token() == 2048


def test_cached_chunks_match_full_pass(rope_tables):
    # 4.3 replaced the old "cache not implemented yet" placeholder test.
    # Feed 16 tokens as chunks 6 + 1 + 5 + 4 (prefill, decode, two multi-token steps).
    from tinyllm.kv_cache import KVCache
    attn = MHA(CFG, 0)
    x = torch.randn(B, T, CFG.d_model)
    cos, sin = rope_tables
    cache = KVCache(n_layers=1, max_len=T)
    outs = []
    for n in (6, 1, 5, 4):
        s = cache.pos
        outs.append(attn(x[:, s:s + n], (cos[s:s + n], sin[s:s + n]), cache.layer(0)))
        cache.advance(n)
    torch.testing.assert_close(torch.cat(outs, dim=1), attn(x, rope_tables), atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_cuda_close_to_fp32(rope_tables):
    attn = MHA(CFG, 0).cuda()
    x = torch.randn(B, T, CFG.d_model, device="cuda")
    cos, sin = (t.cuda() for t in rope_tables)
    ref = attn(x, (cos, sin))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = attn(x, (cos, sin))
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)
