"""Step 4.10b: the Triton INT8 matmul (tinyllm.kernels.int8_gemv) and Int8Linear's backends.

GPU tests (skipped in CI): the kernel against an fp64 dequantize-then-matmul reference on every
TinyLlama Linear shape, at decode-sized M. It must be at least as close as the PyTorch path, which
rounds the matmul to bf16 before scaling. CPU tests (run in CI): the backend switch and fallbacks.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tinyllm.quant import Int8Linear, quantize_model, quantize_per_channel, set_backend

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
gpu = pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA and Triton")

# TinyLlama 1.1B: q/o (2048 x 2048), k/v (256 x 2048), gate/up (5632 x 2048), down (2048 x 5632),
# lm_head (32000 x 2048); plus an odd shape for the masks
SHAPES = [(2048, 2048), (256, 2048), (5632, 2048), (2048, 5632), (32000, 2048), (100, 300)]


def _case(n: int, k: int, m: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    q, s = quantize_per_channel(w)
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    ref = x.double() @ (q.double() * s.double()[:, None]).T
    return x, q, s, ref


@gpu
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 8, 16, 33, 64])
def test_kernel_matches_fp64_reference(n, k, m):
    from tinyllm.kernels.int8_gemv import int8_matmul
    x, q, s, ref = _case(n, k, m)
    y = int8_matmul(x, q, s)
    assert y.shape == (m, n) and y.dtype == torch.float32
    torch_path = F.linear(x, q.to(torch.bfloat16)) * s           # Int8Linear backend "torch"
    err, err_torch = (y.double() - ref).abs().max().item(), (torch_path.double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err / scale < 1e-4, (err, scale)                      # fp32 accumulation of exact products
    assert err <= err_torch, (err, err_torch)                    # no bf16 rounding before the scale


@gpu
def test_kernel_rejects_what_it_does_not_take():
    from tinyllm.kernels.int8_gemv import MAX_M, int8_matmul
    x, q, s, _ = _case(64, 128, 4)
    with pytest.raises(ValueError, match="M <="):
        int8_matmul(torch.zeros(MAX_M + 1, 128, device="cuda", dtype=torch.bfloat16), q, s)
    with pytest.raises(ValueError, match="bf16/fp16"):
        int8_matmul(x.float(), q, s)
    with pytest.raises(ValueError, match="matching K"):
        int8_matmul(x[:, :64], q, s)


@gpu
def test_int8linear_triton_backend_under_autocast():
    from tinyllm.kernels.int8_gemv import MAX_M
    torch.manual_seed(0)
    lin = nn.Linear(512, 384, bias=False).cuda()
    a, b = Int8Linear.from_linear(lin, "torch").cuda(), Int8Linear.from_linear(lin, "triton").cuda()
    x_decode = torch.randn(4, 1, 512, device="cuda")              # fp32 residual stream, 4 rows
    x_prefill = torch.randn(2, MAX_M, 512, device="cuda")         # 128 rows: falls back to torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ya, yb = a(x_decode), b(x_decode)
        assert yb.shape == ya.shape == (4, 1, 384) and yb.dtype == ya.dtype == torch.float32
        torch.testing.assert_close(yb, ya, rtol=1e-2, atol=1e-2 * ya.abs().max().item())
        assert torch.equal(b(x_prefill), a(x_prefill))
    assert torch.equal(b(x_decode), a(x_decode))                 # fp32 without autocast: torch path


@gpu
def test_triton_backend_replays_in_a_cuda_graph():
    torch.manual_seed(0)
    lin = Int8Linear.from_linear(nn.Linear(2048, 2048, bias=False), "triton").cuda()
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()                                   # warm up (autotune) before capture
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        lin(x)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = lin(x)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.testing.assert_close(out, lin(x), rtol=0, atol=0)


def test_cpu_falls_back_to_the_torch_path():
    torch.manual_seed(0)
    lin = nn.Linear(64, 48, bias=False)
    a, b = Int8Linear.from_linear(lin, "torch"), Int8Linear.from_linear(lin, "triton")
    x = torch.randn(3, 64)
    assert torch.equal(a(x), b(x))
    assert "backend=triton" in repr(b)


def test_backend_switch_shares_weights_and_keeps_state_dict():
    model = nn.Sequential(nn.Linear(16, 32, bias=False), nn.ReLU(), nn.Linear(32, 8, bias=False))
    assert quantize_model(model, backend="triton") == 2
    assert {m.backend for m in model.modules() if isinstance(m, Int8Linear)} == {"triton"}
    keys, ptr = set(model.state_dict()), model[0].qweight.data_ptr()
    assert set_backend(model, "torch") == 2
    assert {m.backend for m in model.modules() if isinstance(m, Int8Linear)} == {"torch"}
    assert set(model.state_dict()) == keys and model[0].qweight.data_ptr() == ptr   # same buffers, no copy
    with pytest.raises(ValueError, match="unknown Int8Linear backend"):
        set_backend(model, "cuda")
    with pytest.raises(ValueError, match="unknown Int8Linear backend"):
        Int8Linear.from_linear(nn.Linear(4, 4, bias=False), "int4")
