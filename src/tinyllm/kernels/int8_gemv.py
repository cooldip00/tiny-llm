"""Step 4.10b (D-016): INT8 weight-only matmul for decode, reading the int8 weights directly.

The PyTorch path in tinyllm.quant (Int8Linear, backend "torch") runs three kernels per layer:
cast the int8 weight to bf16 (reads 1 byte and writes 2 per weight), matmul (reads those 2 bytes
again), then scale. Here one kernel reads each int8 weight once, converts it in registers (exact:
|q| <= 127), accumulates in fp32 and applies the per-row scale at the end:

    y[m, n] = scale[n] * sum_k x[m, k] * q[n, k]          x: (M, K) bf16/fp16, q: (N, K) int8

At decode, M (tokens in the step = batch) is small, so the step is bound by reading q. The tile
over M is padded to 16 rows for tl.dot (tensor cores); the wasted flops are cheap next to the
bytes. Each program owns BLOCK_N output columns and loops over K. Block sizes are autotuned per
(N, K, M bucket) on first use, which must happen outside CUDA-graph capture (GraphDecoder warms up
first). M above MAX_M (prefill) stays on the PyTorch path; see Int8Linear.forward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

MAX_M = 64  # rows per call this kernel takes; more (prefill) is a GEMM, left to cuBLAS

_CONFIGS = [triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
            for bn in (16, 32, 64) for bk in (128, 256) for w in (4,) for s in (3, 4)]
_CONFIGS += [triton.Config({"BLOCK_N": 32, "BLOCK_K": 512}, num_warps=8, num_stages=3),
             triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=2, num_stages=4)]


@triton.autotune(configs=_CONFIGS, key=["N", "K", "BLOCK_M"])
@triton.jit
def _int8_matmul_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_qn, stride_ym,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m_ok, n_ok = offs_m < M, offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_ok = offs_k < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_ok[:, None] & k_ok[None, :], other=0.0)
        q = tl.load(q_ptr + offs_n[:, None] * stride_qn + offs_k[None, :],
                    mask=n_ok[:, None] & k_ok[None, :], other=0)
        acc = tl.dot(x, tl.trans(q.to(x.dtype)), acc)       # int8 -> bf16 is exact
    s = tl.load(s_ptr + offs_n, mask=n_ok, other=0.0)
    y = acc * s[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], y, mask=m_ok[:, None] & n_ok[None, :])


def int8_matmul(x: Tensor, q: Tensor, scale: Tensor) -> Tensor:
    """x (M, K) bf16/fp16 CUDA, M <= MAX_M; q (N, K) int8; scale (N,) fp32 -> y (M, N) fp32.

    Same values as F.linear(x, q.to(x.dtype)) * scale, except that the matmul result isn't rounded
    to bf16 before scaling (closer to the fp32 dequantize-then-matmul reference, not further)."""
    M, K = x.shape
    N = q.shape[0]
    if M > MAX_M or q.shape[1] != K or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"int8_matmul: x {tuple(x.shape)} {x.dtype}, q {tuple(q.shape)}; needs M <= {MAX_M}, "
                         "matching K, bf16/fp16 x")
    x = x if x.stride(1) == 1 else x.contiguous()
    q = q if q.stride(1) == 1 else q.contiguous()
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    block_m = max(16, triton.next_power_of_2(M))
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    _int8_matmul_kernel[grid](x, q, scale, y, M, N, K, x.stride(0), q.stride(0), y.stride(0), BLOCK_M=block_m)
    return y


def best_configs() -> dict[str, str]:
    """The autotuned config per (N, K, BLOCK_M) key so far, for the benchmark JSON."""
    return {str(k): str(v) for k, v in _int8_matmul_kernel.cache.items()}
