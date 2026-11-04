"""Step 4.6: INT8 weight-only quantization (fp32 on CPU, so every check is deterministic)."""

import copy
import dataclasses

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from test_engine import CFG, make_requests, solo
from tinyllm.engine import run_continuous
from tinyllm.generate import generate, sample
from tinyllm.model import Transformer
from tinyllm.quant import (QMAX, Int8Linear, dequantize, quantize_model, quantize_per_channel,
                           weight_bytes)


def make_model(tie: bool) -> Transformer:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(dataclasses.replace(CFG, tie_embeddings=tie)).eval()
    with torch.no_grad():                         # well-separated logits, as in test_engine
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return m


@pytest.fixture(scope="module", params=[True, False], ids=["tied", "untied"])
def pair(request):
    fp = make_model(request.param)
    q = copy.deepcopy(fp)
    n = quantize_model(q)
    return fp, q, n


def test_roundtrip_error_is_at_most_half_a_step():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(32, 64, generator=g) * torch.logspace(-3, 1, 32)[:, None]  # rows of very different size
    q, s = quantize_per_channel(w)
    assert q.dtype == torch.int8 and s.shape == (32,)
    assert (q.abs().amax(dim=1) == QMAX).all()                  # each row uses the full range
    err = (dequantize(q, s) - w).abs()
    assert (err <= s[:, None] / 2 + 1e-7).all()


def test_zero_row_gives_zeros_not_nan():
    w = torch.randn(4, 8)
    w[2] = 0
    q, s = quantize_per_channel(w)
    assert s[2] == 0 and (q[2] == 0).all()
    assert torch.isfinite(dequantize(q, s)).all()


def test_int8_linear_equals_linear_on_dequantized_weight():
    lin = nn.Linear(64, 48, bias=False)
    ql = Int8Linear.from_linear(lin)
    x = torch.randn(5, 3, 64)
    ref = F.linear(x, dequantize(ql.qweight, ql.scale))         # dequantize first, then matmul
    torch.testing.assert_close(ql(x), ref, atol=1e-5, rtol=1e-5)


def test_bias_is_rejected():
    with pytest.raises(ValueError, match="bias-free"):
        Int8Linear.from_linear(nn.Linear(4, 4, bias=True))


def test_every_linear_is_replaced_and_embedding_kept(pair):
    fp, q, n = pair
    assert n == 7 * CFG.n_layers + 1                            # q k v o, gate up down, lm_head
    assert not any(isinstance(m, nn.Linear) for m in q.modules())
    assert torch.equal(q.embed_tokens.weight, fp.embed_tokens.weight)
    if fp.cfg.tie_embeddings:                                   # lm_head quantized from the shared matrix
        expect, _ = quantize_per_channel(fp.embed_tokens.weight)
        assert torch.equal(q.lm_head.qweight, expect)


def test_matmul_weight_bytes_halve(pair):
    fp, q, _ = pair
    b16, b8 = weight_bytes(fp, float_bytes=2), weight_bytes(q, float_bytes=2)
    rows = sum(m.out_features for m in q.modules() if isinstance(m, Int8Linear))
    assert b8["matmul"] == b16["matmul"] // 2 + 4 * rows        # 1 byte per weight + an fp32 scale per row
    table = CFG.vocab_size * CFG.d_model * 2
    assert b16["embedding"] == (0 if fp.cfg.tie_embeddings else table)  # tied: counted under lm_head
    assert b8["embedding"] == table                             # untied from lm_head's int8 copy


def test_quantized_logits_stay_close(pair):
    fp, q, _ = pair
    idx = torch.randint(0, CFG.vocab_size, (4, 32), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a, b = fp(idx)[0], q(idx)[0]
    rel = ((a - b).abs().max() / a.abs().max()).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    # Measured ~2% here: the worst of 16K logits, on a toy whose rows have only 64-160 weights
    # (little averaging of rounding errors) and whose weights are scaled x4. The real model is in bench_quant.
    assert rel < 0.05, rel
    assert agree > 0.95, agree


def test_quantized_kv_generate_matches_naive(pair):
    _, q, _ = pair
    idx = torch.randint(0, CFG.vocab_size, (3, 7), generator=torch.Generator().manual_seed(3))
    assert torch.equal(generate(q, idx, 20, temperature=0), sample(q, idx, 20, temperature=0))


@pytest.mark.parametrize("kv", ["slots", "paged"])
def test_quantized_model_in_the_engine_matches_solo(pair, kv):
    _, q, _ = pair
    reqs = make_requests(7)
    results, _ = run_continuous(q, reqs, n_slots=3, kv=kv, block_size=4)
    assert [r.tokens for r in results] == [solo(q, r) for r in reqs]
