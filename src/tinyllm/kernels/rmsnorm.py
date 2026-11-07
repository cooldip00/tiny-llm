"""Step 4.10a (D-016): RMSNorm, and residual add + RMSNorm, each as one Triton kernel.

The PyTorch RMSNorm (tinyllm.model.RMSNorm, backend "torch") runs about seven small kernels per
call: cast to fp32, square, mean, add eps + rsqrt, multiply, cast back, multiply by the weight. At
decode a row is one token's 2,048 activations (4 KB in bf16), so each of those kernels is mostly
fixed cost. Here one program per row loads the row once and does it all in registers:

    rmsnorm:      y = (x * rsqrt(mean(x^2) + eps)).to(x.dtype) * w
    add_rmsnorm:  h = x + a  (written out: the new residual stream);  y = rmsnorm(h)

The roundings match the reference: h is rounded to its dtype before the statistic, the normalized
row is rounded to x's dtype before the weight multiply, and the product is rounded to the promoted
dtype of (x, w). Only the order of the fp32 sum differs, so results are allclose, not bitwise.

One compiled config per (row width, dtypes): no autotuning, so nothing has to run before CUDA-graph
capture beyond the first call (GraphDecoder warms up first).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

MAX_D = 65536  # a whole row lives in one program's registers


@triton.jit
def _rmsnorm_kernel(x_ptr, a_ptr, h_ptr, w_ptr, y_ptr, D, eps, stride_x, stride_a, stride_h, stride_y,
                    ADD: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    ok = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=ok, other=0.0)
    if ADD:
        a = tl.load(a_ptr + row * stride_a + offs, mask=ok, other=0.0)
        h = (x.to(tl.float32) + a.to(tl.float32)).to(h_ptr.dtype.element_ty)   # x + a in the promoted dtype
        tl.store(h_ptr + row * stride_h + offs, h, mask=ok)
        x = h
    xf = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / D + eps)
    n = (xf * rstd).to(x.dtype).to(tl.float32)                                 # .to(x.dtype), as the reference
    w = tl.load(w_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, (n * w).to(y_ptr.dtype.element_ty), mask=ok)


def _rows(t: Tensor) -> Tensor:
    t = t.reshape(-1, t.shape[-1])
    return t if t.stride(-1) == 1 else t.contiguous()


def _launch(x: Tensor, a: Tensor | None, weight: Tensor, eps: float) -> tuple[Tensor, Tensor | None]:
    D = x.shape[-1]
    if not (x.is_cuda and weight.is_cuda) or weight.shape != (D,) or D > MAX_D:
        raise ValueError(f"rmsnorm: x {tuple(x.shape)} on {x.device}, weight {tuple(weight.shape)}; "
                         f"needs CUDA tensors, weight of shape ({D},), width <= {MAX_D}")
    x2 = _rows(x)
    h = h2 = a2 = None
    if a is not None:
        if a.shape != x.shape:
            raise ValueError(f"add_rmsnorm: x {tuple(x.shape)} and a {tuple(a.shape)} differ")
        a2 = _rows(a)
        h = torch.empty(x.shape, device=x.device, dtype=torch.result_type(x, a))
        h2 = h.view(-1, D)
    src = x2 if h2 is None else h2
    y = torch.empty(x.shape, device=x.device, dtype=torch.result_type(src, weight))
    y2 = y.view(-1, D)
    rows = x2.shape[0]
    if rows:
        block = triton.next_power_of_2(D)
        warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
        _rmsnorm_kernel[(rows,)](
            x2, x2 if a2 is None else a2, x2 if h2 is None else h2, weight, y2, D, eps,
            x2.stride(0), 0 if a2 is None else a2.stride(0), 0 if h2 is None else h2.stride(0), y2.stride(0),
            ADD=a is not None, BLOCK_D=block, num_warps=warps)
    return y, h


def rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """x (..., D) CUDA -> y (..., D) in result_type(x, weight); same values as tinyllm.model.RMSNorm."""
    return _launch(x, None, weight, eps)[0]


def add_rmsnorm(x: Tensor, a: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """(y, h) with h = x + a and y = rmsnorm(h, weight, eps), from one pass over the row."""
    y, h = _launch(x, a, weight, eps)
    return y, h
