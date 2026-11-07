"""Step 4.6: INT8 weight-only quantization (per-output-channel, symmetric absmax).

Each Linear weight row j gets one scale s_j = max|W_j| / 127 and is stored as
q_j = round(W_j / s_j), clipped to [-127, 127] (int8, 1 byte instead of 2 for bf16).
Activations stay in floating point ("weight-only", W8A16).

Because the scale is per output channel, it factors out of the matmul exactly:
    y_j = sum_i x_i * (q_ji * s_j) = s_j * sum_i x_i * q_ji
so the forward is  F.linear(x, q.to(dtype)) * s : cast int8 -> bf16 (exact: |q| <= 127 < 256),
matmul, scale the output. Values are the same as dequantizing W first. In PyTorch the int8 ->
bf16 cast is its own kernel (backend "torch"). Backend "triton" (4.10b) runs one kernel that reads
the int8 weights directly (tinyllm.kernels.int8_gemv) for decode-sized inputs on CUDA, and falls
back to the torch path otherwise (CPU, fp32, prefill-sized inputs, no Triton).

Embeddings stay in floating point: a lookup reads only the B rows it needs, not the table.
With tied embeddings, lm_head gets its own int8 copy and the embedding keeps the original.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QMAX = 127  # symmetric: -128 is unused so that +/- ranges match
BACKENDS = ("torch", "triton")


def quantize_per_channel(w: Tensor) -> tuple[Tensor, Tensor]:
    """(out, in) float weight -> (int8 q (out, in), fp32 scale (out,)). An all-zero row gets scale 0."""
    w = w.detach().float()
    scale = w.abs().amax(dim=1) / QMAX
    q = torch.round(w / scale.clamp_min(1e-12)[:, None]).clamp_(-QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q: Tensor, scale: Tensor) -> Tensor:
    return q.float() * scale[:, None]


class Int8Linear(nn.Module):
    """Drop-in for a bias-free nn.Linear, with int8 weights and one fp32 scale per output channel."""

    def __init__(self, q: Tensor, scale: Tensor, backend: str = "torch"):
        super().__init__()
        self.in_features, self.out_features = q.shape[1], q.shape[0]
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        self.backend = _check_backend(backend)

    @classmethod
    def from_linear(cls, m: nn.Linear, backend: str = "torch") -> Int8Linear:
        if m.bias is not None:
            raise ValueError("Int8Linear supports bias-free Linear only (the model has no biases)")
        return cls(*quantize_per_channel(m.weight), backend=backend)

    def forward(self, x: Tensor) -> Tensor:
        dev = x.device.type
        dt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        if self.backend == "triton" and x.is_cuda and dt in (torch.bfloat16, torch.float16):
            kernel = _triton_kernel()
            rows = x.numel() // self.in_features
            if kernel is not None and rows <= kernel.MAX_M:
                y = kernel.int8_matmul(x.reshape(rows, self.in_features).to(dt), self.qweight, self.scale)
                return y.view(*x.shape[:-1], self.out_features)
        # Output keeps fp32 under autocast (bf16 matmul result * fp32 scale), so the scale isn't
        # rounded to bf16. Downstream ops cast as they would after an nn.Linear.
        return F.linear(x.to(dt), self.qweight.to(dt)) * self.scale

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, int8 per-channel, "
                f"backend={self.backend}")


def _check_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unknown Int8Linear backend {backend!r}; one of {BACKENDS}")
    return backend


_KERNEL: list = []


def _triton_kernel():
    """tinyllm.kernels.int8_gemv, or None without Triton (imported once, on first CUDA use)."""
    if not _KERNEL:
        try:
            from tinyllm.kernels import int8_gemv
        except ImportError:
            int8_gemv = None
        _KERNEL.append(int8_gemv)
    return _KERNEL[0]


def quantize_model(model: nn.Module, backend: str = "torch") -> int:
    """Replace every nn.Linear in `model` (in place) with an Int8Linear. Returns how many."""
    targets = [(parent, name, child) for parent in model.modules()
               for name, child in parent.named_children() if isinstance(child, nn.Linear)]
    for parent, name, child in targets:
        q = Int8Linear.from_linear(child, backend).to(child.weight.device)
        setattr(parent, name, q)
    return len(targets)


def set_backend(model: nn.Module, backend: str) -> int:
    """Switch every Int8Linear in `model` to `backend` (same weights, no copy). Returns how many."""
    _check_backend(backend)
    mods = [m for m in model.modules() if isinstance(m, Int8Linear)]
    for m in mods:
        m.backend = backend
    return len(mods)


def weight_bytes(model: nn.Module, float_bytes: int | None = None) -> dict[str, int]:
    """Bytes of persistent weights (params + saved buffers; RoPE tables excluded), split into the
    matmul weights read every decode step and the embedding table (only B rows read per step).

    float_bytes: count float tensors at this width (2 = as if stored in bf16); None = as stored.
    Int8Linear tensors are always counted as stored (int8 weights, fp32 scales).
    A tied matrix is counted once, under matmul: lm_head reads all of it every step."""
    def nbytes(t: Tensor, as_stored: bool) -> int:
        if t.is_floating_point() and float_bytes is not None and not as_stored:
            return t.numel() * float_bytes
        return t.numel() * t.element_size()

    is_mm = lambda m: isinstance(m, (nn.Linear, Int8Linear))  # noqa: E731
    mods = sorted(model.modules(), key=lambda m: not is_mm(m))  # matmul modules claim shared tensors first
    seen, out = set(), {"matmul": 0, "embedding": 0, "other": 0}
    for mod in mods:
        tensors = list(mod.named_parameters(recurse=False)) + [
            (n, b) for n, b in mod.named_buffers(recurse=False)
            if n not in mod._non_persistent_buffers_set]
        for _, t in tensors:
            if id(t) in seen:
                continue
            seen.add(id(t))
            kind = "matmul" if is_mm(mod) else "embedding" if isinstance(mod, nn.Embedding) else "other"
            out[kind] += nbytes(t, as_stored=isinstance(mod, Int8Linear))  # its scales stay fp32
    return out
