import copy
import math

import numpy as np
import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.generate import sample
from tinyllm.logger import Logger, make_logger
from tinyllm.model import Transformer
from tinyllm.train import TrainConfig, load_checkpoint, lr_at, make_optimizer, train, train_step

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


def tiny_cfg(tmp_path, **kw) -> TrainConfig:
    base = dict(seq_len=32, micro_batch=8, grad_accum=2, max_steps=60, lr=3e-3, min_lr=3e-4,
                warmup_steps=5, device="cpu", log_every=1, eval_every=10, eval_batches=2,
                sample_tokens=0, out_dir=str(tmp_path), logger="none")
    return TrainConfig(**{**base, **kw})


def run(cfg, logger=None):
    return train(cfg, model_cfg=TINY, tokens=TOKENS, val_tokens=TOKENS, log=lambda s: None,
                 logger=logger)


class RecordingLogger(Logger):
    run_id = "test-run-id"

    def __init__(self):
        self.calls, self.finished = [], False

    def log(self, metrics, step):
        self.calls.append((step, metrics))

    def finish(self):
        self.finished = True


def test_logger_receives_train_and_val_metrics(tmp_path):
    rec = RecordingLogger()
    run(tiny_cfg(tmp_path, max_steps=12, log_every=5, eval_every=10), logger=rec)
    train_steps = [s for s, m in rec.calls if "train/loss" in m]
    val_steps = [s for s, m in rec.calls if "val/loss" in m]
    assert train_steps == [0, 5, 10, 11] and val_steps == [0, 10, 11]
    first = next(m for _, m in rec.calls if "train/loss" in m)
    assert set(first) == {"train/loss", "train/lr", "train/grad_norm", "train/tok_per_s", "train/tokens"}
    assert first["train/tokens"] == 8 * 2 * 32                        # tokens seen after step 0
    assert rec.finished
    assert load_checkpoint(tmp_path / "ckpt.pt", "cpu")["wandb_id"] == "test-run-id"  # resume target


def test_logger_finishes_even_if_training_crashes(tmp_path):
    rec = RecordingLogger()
    with pytest.raises(IndexError):
        # Tokens beyond the 64-token vocab make the embedding lookup fail mid-run.
        train(tiny_cfg(tmp_path), model_cfg=TINY, tokens=TOKENS + 100, val_tokens=TOKENS,
              log=lambda s: None, logger=rec)
    assert rec.finished


def test_wandb_logger_disabled_mode(tmp_path):
    lg = make_logger("wandb", project="tinyllm-test", name="t", config={"a": 1},
                     mode="disabled", dir=str(tmp_path))
    lg.log({"train/loss": 1.0}, step=0)
    lg.log_sample("Once upon a time", step=0)
    lg.finish()


def test_unknown_logger_raises():
    with pytest.raises(ValueError):
        make_logger("tensorboard")


def test_train_loop_learns_on_cpu(tmp_path):
    _, hist = run(tiny_cfg(tmp_path))
    steps = [h for h in hist if "loss" in h]
    assert len(steps) == 60
    assert steps[0]["loss"] == pytest.approx(math.log(64), abs=0.3)
    assert steps[-1]["loss"] < 0.5 * steps[0]["loss"]
    assert all(math.isfinite(h["grad_norm"]) for h in steps)


def test_eval_schedule_and_val_loss_drops(tmp_path):
    _, hist = run(tiny_cfg(tmp_path))
    evals = [h for h in hist if "val_loss" in h]
    assert [e["step"] for e in evals] == [0, 10, 20, 30, 40, 50, 59]   # every 10 + the last step
    assert evals[-1]["val_loss"] < 0.5 * evals[0]["val_loss"]


def test_checkpoint_contents(tmp_path):
    run(tiny_cfg(tmp_path, max_steps=12))
    ck = load_checkpoint(tmp_path / "ckpt.pt", "cpu")
    assert ck["step"] == 11
    assert set(ck) >= {"model", "optimizer", "rng", "best_val", "model_cfg", "train_cfg"}
    assert ModelConfig.from_dict(ck["model_cfg"]) == TINY
    best = load_checkpoint(tmp_path / "best.pt", "cpu")
    assert best["val_loss"] == ck["best_val"] and "optimizer" not in best
    m = Transformer(TINY)
    m.load_state_dict(best["model"])                                  # loads cleanly into a fresh model


def test_resume_is_exact(tmp_path):
    """20 steps in one go == 10 steps, stop, resume, 10 more."""
    full, _ = run(tiny_cfg(tmp_path / "full", max_steps=20))
    run(tiny_cfg(tmp_path / "split", max_steps=20, stop_at=10))
    resumed, hist = run(tiny_cfg(tmp_path / "split", max_steps=20, resume=True))
    assert [h for h in hist if "loss" in h][0]["step"] == 10
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)              # bit-identical on CPU


def test_seq_len_longer_than_model_raises(tmp_path):
    with pytest.raises(ValueError):
        run(tiny_cfg(tmp_path, seq_len=64))


def test_sample_greedy_is_deterministic_and_stops_at_eos():
    torch.manual_seed(0)
    model = Transformer(TINY)
    prompt = torch.tensor([[1, 2, 3]])
    a = sample(model, prompt, 10, temperature=0)
    assert a.shape == (1, 13) and torch.equal(a[:, :3], prompt)
    assert torch.equal(a, sample(model, prompt, 10, temperature=0))
    eos = a[0, 3].item()                                              # first generated token as "EOS"
    assert sample(model, prompt, 10, temperature=0, eos_id=eos).shape == (1, 4)
    assert model.training                                             # restores train mode


def test_sample_top_k_1_equals_greedy():
    torch.manual_seed(0)
    model = Transformer(TINY)
    prompt = torch.tensor([[5, 6]])
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample(model, prompt, 8, temperature=1.0, top_k=1, generator=g),
                       sample(model, prompt, 8, temperature=0))


def test_sample_slides_past_max_seq_len():
    model = Transformer(TINY)
    out = sample(model, torch.zeros(1, 30, dtype=torch.long), 10, temperature=0)
    assert out.shape == (1, 40)                                       # > max_seq_len 32, no error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_train_loop_cuda_fused_bf16(tmp_path):
    model, hist = run(tiny_cfg(tmp_path, max_steps=30, device="cuda"))
    assert next(model.parameters()).is_cuda
    steps = [h for h in hist if "loss" in h]
    assert steps[-1]["loss"] < steps[0]["loss"]
    assert load_checkpoint(tmp_path / "ckpt.pt", "cuda")["step"] == 29
