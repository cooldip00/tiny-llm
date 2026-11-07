"""Step 4.10a: the Triton RMSNorm and residual add + RMSNorm (tinyllm.kernels.rmsnorm), and
RMSNorm's backends.

GPU tests (skipped in CI): the kernels against tinyllm.model.RMSNorm's PyTorch code at the 30M (512)
and TinyLlama (2,048) widths plus odd ones, from 1 row (decode) to 4,096 (prefill), in bf16, fp16,
fp32 and mixed dtypes; the residual h = x + a must be bitwise equal to PyTorch's. Then CUDA-graph
replay, and a small model's greedy tokens on both backends. CPU tests (run in CI): the backend
switch, the fallbacks, and that Block's restructured forward computes exactly what it did before.
"""

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.generate import generate
from tinyllm.model import Block, RMSNorm, Transformer, norm_backend, set_norm_backend

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
gpu = pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA and Triton")

CFG = ModelConfig(vocab_size=128, max_seq_len=96, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                  ffn_hidden=160)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _reference(x, w, eps):
    """tinyllm.model.RMSNorm's PyTorch code, standalone."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * w


def _close(y, ref):
    """Only the fp32 sum's order differs from the reference, so in a half dtype the normalized
    value can round one ulp the other way, and the weight multiply rounds again: allow 3 ulps of
    the output dtype (assert_close's fp16 default, rtol 1e-3, is about one ulp). Almost every
    element must still round the same."""
    if y.dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(y, ref, rtol=3 * torch.finfo(y.dtype).eps, atol=1e-5)
        assert (y == ref).float().mean() > 0.99
    else:
        torch.testing.assert_close(y, ref)


def _xw(rows, d, x_dtype, w_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, d, device="cuda", generator=g) * 3).to(x_dtype)
    w = (1 + 0.1 * torch.randn(d, device="cuda", generator=g)).to(w_dtype)
    return x, w


# ---------- GPU: the kernels ----------

@gpu
@pytest.mark.parametrize("d", [512, 2048, 5632, 100, 64])
@pytest.mark.parametrize("rows", [1, 3, 33, 4096])
@pytest.mark.parametrize("dt", ["bf16", "fp16", "fp32"])
def test_rmsnorm_matches_reference(d, rows, dt):
    from tinyllm.kernels.rmsnorm import rmsnorm
    x, w = _xw(rows, d, DTYPES[dt], DTYPES[dt])
    ref, y = _reference(x, w, 1e-5), rmsnorm(x, w, 1e-5)
    assert y.dtype == ref.dtype and y.shape == ref.shape
    _close(y, ref)


@gpu
@pytest.mark.parametrize("x_dt,w_dt", [("bf16", "fp32"), ("fp32", "bf16")])
def test_rmsnorm_promotes_like_the_reference(x_dt, w_dt):
    from tinyllm.kernels.rmsnorm import rmsnorm
    x, w = _xw(5, 2048, DTYPES[x_dt], DTYPES[w_dt])
    ref, y = _reference(x, w, 1e-6), rmsnorm(x, w, 1e-6)
    assert y.dtype == ref.dtype == torch.float32
    torch.testing.assert_close(y, ref)


@gpu
@pytest.mark.parametrize("x_dt,a_dt,w_dt", [("bf16", "bf16", "bf16"), ("fp32", "bf16", "fp32"),
                                            ("fp16", "fp16", "fp16")])
@pytest.mark.parametrize("rows", [1, 32, 4096])
def test_add_rmsnorm_matches_reference(x_dt, a_dt, w_dt, rows):
    from tinyllm.kernels.rmsnorm import add_rmsnorm
    x, w = _xw(rows, 2048, DTYPES[x_dt], DTYPES[w_dt])
    a, _ = _xw(rows, 2048, DTYPES[a_dt], DTYPES[w_dt], seed=1)
    h_ref = x + a
    y, h = add_rmsnorm(x, a, w, 1e-5)
    assert h.dtype == h_ref.dtype and torch.equal(h, h_ref)        # the residual stream: bitwise
    _close(y, _reference(h_ref, w, 1e-5))


@gpu
def test_strided_and_batched_inputs():
    from tinyllm.kernels.rmsnorm import add_rmsnorm, rmsnorm
    big, w = _xw(6, 4096, torch.bfloat16, torch.bfloat16)
    x = big[:, :2048]                                  # row stride 4096: the kernel's stride path
    torch.testing.assert_close(rmsnorm(x, w[:2048], 1e-5), _reference(x, w[:2048], 1e-5))
    xt = big.t()[:2048]                                # last-dim stride != 1: copied first
    torch.testing.assert_close(rmsnorm(xt, w[:6], 1e-5), _reference(xt, w[:6], 1e-5))
    x3, a3 = big.view(2, 3, 4096), big.flip(0).view(2, 3, 4096)  # (B, T, D) as in the model
    y, h = add_rmsnorm(x3, a3, w, 1e-5)
    assert y.shape == h.shape == (2, 3, 4096) and torch.equal(h, x3 + a3)
    torch.testing.assert_close(y, _reference(x3 + a3, w, 1e-5))


@gpu
def test_kernel_rejects_what_it_does_not_take():
    from tinyllm.kernels.rmsnorm import add_rmsnorm, rmsnorm
    x, w = _xw(2, 64, torch.bfloat16, torch.bfloat16)
    with pytest.raises(ValueError, match="weight of shape"):
        rmsnorm(x, w[:32], 1e-5)
    with pytest.raises(ValueError, match="CUDA"):
        rmsnorm(x.cpu(), w, 1e-5)
    with pytest.raises(ValueError, match="differ"):
        add_rmsnorm(x, x[:1], w, 1e-5)


@gpu
def test_triton_backend_replays_in_a_cuda_graph():
    norm = RMSNorm(2048, backend="triton").cuda().to(torch.bfloat16)
    x, a = _xw(1, 2048, torch.bfloat16, torch.bfloat16)[0], _xw(1, 2048, torch.bfloat16, torch.bfloat16, 1)[0]
    with torch.no_grad():
        side = torch.cuda.Stream()                     # compile before capture
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            norm.add_forward(x, a)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y, h = norm.add_forward(x, a)
        x.copy_(torch.randn_like(x))
        g.replay()
        y2, h2 = norm.add_forward(x, a)
    assert torch.equal(h, h2) and torch.equal(y, y2)


@gpu
def test_triton_backend_steps_aside_for_autograd():
    norm = RMSNorm(256, backend="triton").cuda()
    x = torch.randn(4, 256, device="cuda", requires_grad=True)
    y = norm(x)                                       # grad needed: the PyTorch path
    y.sum().backward()
    assert x.grad is not None and norm.weight.grad is not None
    with torch.no_grad():
        assert norm(x).grad_fn is None
        torch.testing.assert_close(norm(x), y.detach())


@gpu
@torch.no_grad()
def test_model_greedy_same_on_both_backends():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = Transformer(CFG).eval()
    for p in model.parameters():                      # larger weights: well-separated greedy tokens
        if p.dim() == 2:
            p.mul_(4)
    model = model.cuda()
    assert set_norm_backend(model, "torch") == 2 * CFG.n_layers + 1
    assert norm_backend(model) == "torch"
    ids = torch.randint(0, CFG.vocab_size, (2, 10), generator=torch.Generator().manual_seed(1)).cuda()
    new = CFG.max_seq_len - ids.shape[1]
    want = generate(model, ids, new, temperature=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = model(want)[0].float()
    assert set_norm_backend(model, "triton") == 2 * CFG.n_layers + 1
    assert norm_backend(model) == "triton"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = model(want)[0].float()
    assert (got - ref).abs().max() <= 0.02 * ref.abs().max()
    assert torch.equal(generate(model, ids, new, temperature=0), want)
    assert torch.equal(generate(model, ids, new, temperature=0, cuda_graph=True), want)


# ---------- CPU: backends and fallbacks (run in CI) ----------

def test_default_backend_is_triton():
    """D-032: the fused kernel is the default; it runs only where it can (CUDA, no gradient)."""
    assert RMSNorm(8).backend == "triton"
    model = Transformer(CFG)
    assert {m.backend for m in model.modules() if isinstance(m, RMSNorm)} == {"triton"}
    assert norm_backend(model) == "torch"                       # CPU weights: the PyTorch code runs
    assert norm_backend() == ("triton" if CUDA and HAVE_TRITON else "torch")
    model.norm.backend = "torch"
    assert norm_backend(model) == "mixed"


def test_cpu_falls_back_to_the_torch_path():
    torch.manual_seed(0)
    a, b = RMSNorm(64, backend="torch"), RMSNorm(64, backend="triton")
    x, r = torch.randn(3, 64), torch.randn(3, 64)
    assert torch.equal(a(x), b(x))
    ya, ha = a.add_forward(x, r)
    yb, hb = b.add_forward(x, r)
    assert torch.equal(ha, x + r) and torch.equal(hb, ha)
    assert torch.equal(ya, a(x + r)) and torch.equal(yb, ya)
    assert "backend=triton" in repr(b)


def test_block_forward_unchanged_on_torch_backend():
    torch.manual_seed(0)
    blk = Block(CFG, 0).eval()
    x = torch.randn(2, 7, CFG.d_model)
    rope = Transformer(CFG).rope(7)
    with torch.no_grad():
        h = x + blk.self_attn(blk.input_layernorm(x), rope, None)       # the pre-4.10a forward
        want = h + blk.mlp(blk.post_attention_layernorm(h))
        assert torch.equal(blk(x, rope), want)


def test_backend_switch_keeps_state_dict():
    model = Transformer(CFG)
    keys = set(model.state_dict())
    assert set_norm_backend(model, "triton") == 2 * CFG.n_layers + 1
    assert {m.backend for m in model.modules() if isinstance(m, RMSNorm)} == {"triton"}
    assert set(model.state_dict()) == keys
    assert set_norm_backend(model, "torch") == 2 * CFG.n_layers + 1
    with pytest.raises(ValueError, match="unknown RMSNorm backend"):
        set_norm_backend(model, "cuda")
    with pytest.raises(ValueError, match="unknown RMSNorm backend"):
        RMSNorm(8, backend="int8")
