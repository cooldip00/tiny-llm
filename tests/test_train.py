import copy
import math

import numpy as np
import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.model import Transformer
from tinyllm.train import TrainConfig, lr_at, make_optimizer, train, train_step

# Tiny model + a periodic corpus (next token = current + 1 mod 64), easy to learn on CPU.
TINY = ModelConfig(vocab_size=64, max_seq_len=32, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64)
TOKENS = np.tile(np.arange(64, dtype=np.uint16), 200)


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(1e-4)            # 1/10 of peak on the first step
    assert lr_at(9, cfg) == pytest.approx(1e-3)            # peak at the end of warmup
    assert lr_at(10, cfg) == pytest.approx(1e-3)           # cosine starts at peak
    assert lr_at(60, cfg) == pytest.approx(5.5e-4)         # halfway: midpoint of peak and floor
    assert lr_at(110, cfg) == pytest.approx(1e-4)          # floor at max_steps
    assert lr_at(500, cfg) == pytest.approx(1e-4)          # and after
    after = [lr_at(s, cfg) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(after, after[1:]))   # never increases after warmup


def test_tokens_per_step_default():
    assert TrainConfig().tokens_per_step == 131_072


def test_optimizer_groups():
    model = Transformer(TINY)
    opt = make_optimizer(model, TrainConfig(device="cpu"))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert len(no_decay["params"]) == 2 * TINY.n_layers + 1           # 2 norms per layer + final
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == model.num_params()                                    # tied weight counted once


def test_grad_accumulation_matches_full_batch():
    torch.manual_seed(0)
    a = Transformer(TINY)
    b = copy.deepcopy(a)
    idx = torch.randint(0, TINY.vocab_size, (4, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    # Plain SGD: the update is lr * grad, so equal params after the step means equal grads.
    la, _ = train_step(a, torch.optim.SGD(a.parameters()), [(x, y)], lr=0.1, grad_clip=1e9)
    lb, _ = train_step(b, torch.optim.SGD(b.parameters()),
                       [(x[:2], y[:2]), (x[2:], y[2:])], lr=0.1, grad_clip=1e9)
    torch.testing.assert_close(la, lb)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)


def test_grad_clip_reports_pre_clip_norm_and_clips():
    torch.manual_seed(0)
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 17))
    opt = torch.optim.SGD(model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    _, norm = train_step(model, opt, [(idx[:, :-1], idx[:, 1:])], lr=1.0, grad_clip=1e-3)
    assert norm.item() > 1e-3                              # reported norm is the raw one
    step = torch.sqrt(sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), before)))
    assert step.item() == pytest.approx(1e-3, rel=1e-3)    # SGD lr=1 step size = clipped norm


def test_train_loop_learns_on_cpu():
    cfg = TrainConfig(seq_len=32, micro_batch=8, grad_accum=2, max_steps=60, lr=3e-3,
                      min_lr=3e-4, warmup_steps=5, device="cpu", log_every=1)
    _, hist = train(cfg, model_cfg=TINY, tokens=TOKENS, log=lambda s: None)
    assert len(hist) == 60
    assert hist[0]["loss"] == pytest.approx(math.log(64), abs=0.3)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    assert all(math.isfinite(h["grad_norm"]) for h in hist)


def test_seq_len_longer_than_model_raises():
    with pytest.raises(ValueError):
        train(TrainConfig(seq_len=64, device="cpu"), model_cfg=TINY, tokens=TOKENS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_train_loop_cuda_fused_bf16():
    cfg = TrainConfig(seq_len=32, micro_batch=8, grad_accum=2, max_steps=30, lr=3e-3,
                      warmup_steps=5, device="cuda", log_every=1)
    model, hist = train(cfg, model_cfg=TINY, tokens=TOKENS, log=lambda s: None)
    assert next(model.parameters()).is_cuda
    assert hist[-1]["loss"] < hist[0]["loss"]
