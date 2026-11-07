"""bench_quant times each variant with only its weights on the GPU (4.10b memory-spill fix)."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import bench_generate  # noqa: E402
from bench_quant import place, shared_copy, tensors, time_alone  # noqa: E402
from test_quant import make_model  # noqa: E402

from tinyllm.quant import quantize_model  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def variants():
    torch.manual_seed(0)
    bf16 = make_model(tie=False)
    int8 = copy.deepcopy(bf16)
    quantize_model(int8)
    return {"bf16": bf16, "int8": int8, "int8-triton": shared_copy(int8, "triton")}


GROUPS = {"bf16": "bf16", "int8": "int8", "int8-triton": "int8"}


def devices(m) -> set:
    return {t.device.type for t in tensors(m)}


def shared(a, b) -> bool:
    return all(x is y and x.data_ptr() == y.data_ptr() for x, y in zip(tensors(a), tensors(b)))


@cuda
def test_place_moves_shared_trees_together_and_keeps_outputs():
    v = variants()
    x = torch.randint(0, 100, (2, 9))
    before = v["int8"](x)[0]
    place([v["int8"], v["int8-triton"]], "cuda")
    assert devices(v["int8"]) == devices(v["int8-triton"]) == {"cuda"}
    assert shared(v["int8"], v["int8-triton"])
    assert devices(v["bf16"]) == {"cpu"}
    place([v["int8"], v["int8-triton"]], "cpu")
    assert shared(v["int8"], v["int8-triton"])
    assert torch.equal(v["int8"](x)[0], before)
    assert torch.equal(v["int8-triton"](x)[0], before)


@cuda
def test_time_alone_keeps_only_the_timed_group_on_the_gpu():
    v = variants()
    place(v.values(), "cuda")
    for name, other in (("bf16", "int8"), ("int8-triton", "bf16"), ("int8", "bf16")):
        bench_generate._DECODERS["stale"] = object()
        time_alone(v, GROUPS, name, "cuda")
        assert devices(v[name]) == {"cuda"} and devices(v[other]) == {"cpu"}
        assert not bench_generate._DECODERS
    assert shared(v["int8"], v["int8-triton"])


@cuda
def test_time_alone_frees_the_other_group():
    v = variants()
    place(v.values(), "cuda")
    torch.cuda.synchronize()
    all_on = torch.cuda.memory_allocated()
    time_alone(v, GROUPS, "int8", "cuda")
    bf16_bytes = sum(t.numel() * t.element_size() for t in tensors(v["bf16"]))
    assert torch.cuda.memory_allocated() <= all_on - bf16_bytes
