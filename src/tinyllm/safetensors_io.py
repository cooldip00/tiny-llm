"""Minimal .safetensors reader and writer (step 4.9b, D-026). No dependency beyond torch.

File layout (https://github.com/huggingface/safetensors#format):
  8 bytes   little-endian u64 N = header size
  N bytes   UTF-8 JSON: {name: {"dtype": "BF16", "shape": [...], "data_offsets": [begin, end]}, ...}
            plus an optional "__metadata__": {str: str}. Padded with spaces to a multiple of 8.
  rest      raw little-endian tensor bytes; offsets are relative to the end of the header

Reading memory-maps the file, so a tensor is copied once, straight to its target device.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import torch

DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
NAMES = {v: k for k, v in DTYPES.items()}


def read_header(path: str | Path) -> tuple[dict, int]:
    """(header without __metadata__, byte offset where the data starts)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def load_file(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
              keys: set[str] | None = None) -> dict[str, torch.Tensor]:
    """name -> tensor on `device`. dtype: cast floating tensors to it (integers are kept)."""
    header, start = read_header(path)
    out = {}
    with open(path, "rb") as f:
        # ACCESS_COPY: a writable (copy-on-write) view, so torch.frombuffer doesn't warn.
        # Nothing is written back and pages are only copied if touched for writing.
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            for name, info in header.items():
                if keys is not None and name not in keys:
                    continue
                dt = DTYPES[info["dtype"]]
                begin, end = info["data_offsets"]
                shape = info["shape"]
                numel = 1
                for s in shape:
                    numel *= s
                if (end - begin) != numel * torch.empty((), dtype=dt).element_size():
                    raise ValueError(f"{name}: {end - begin} bytes for shape {shape} {info['dtype']}")
                if numel == 0:
                    t = torch.empty(shape, dtype=dt)
                else:
                    t = torch.frombuffer(mm, dtype=dt, count=numel, offset=start + begin).view(shape)
                target = dtype if (dtype is not None and dt.is_floating_point) else dt
                out[name] = t.to(device=device, dtype=target, copy=True)   # copy: detach from the mmap
                del t                                    # release the view, or mm.close() raises BufferError
        finally:
            mm.close()
    return out


def save_file(tensors: dict[str, torch.Tensor], path: str | Path, metadata: dict[str, str] | None = None) -> None:
    """Write tensors (any device) in safetensors layout. Shared tensors are written once per name."""
    header: dict = {}
    blobs = []
    offset = 0
    for name in sorted(tensors):
        t = tensors[name].detach().contiguous().cpu()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[name] = {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for raw in blobs:
            f.write(raw)
