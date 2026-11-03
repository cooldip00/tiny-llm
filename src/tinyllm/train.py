"""Training loop (Phase 3, hyperparameters in DECISIONS.md D-019).

Step 3.2 covers the core loop: AdamW, warmup + cosine LR, gradient accumulation,
gradient clipping and bf16 autocast. Eval/checkpoints (3.3), the logging backend
(3.4) and MFU (3.5) are added on top.

    uv run python -m tinyllm.train --preset smoke --max_steps 100
"""

import argparse
import math
import time
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor, nn

from tinyllm.config import ModelConfig, get_config
from tinyllm.data import get_batch, load_tokens
from tinyllm.model import Transformer


@dataclass
class TrainConfig:
    preset: str = "smoke"
    seq_len: int = 512
    micro_batch: int = 32            # sequences per forward/backward
    grad_accum: int = 8              # micro-batches per optimizer step: 32 x 512 x 8 = 131,072 tokens/step
    max_steps: int = 2000
    lr: float = 1e-3                 # peak LR, reached at the end of warmup
    min_lr: float = 1e-4             # cosine floor (10% of peak)
    warmup_steps: int = 200
    weight_decay: float = 0.1        # matrices and embeddings only, not norm weights
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    log_every: int = 10

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.seq_len


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr at cfg.max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on 2-D weights only. Norm gains (1-D) are not decayed."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.device(cfg.device).type == "cuda"  # one kernel for all params instead of a loop
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, fused=fused)


def train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batches: list[tuple[Tensor, Tensor]],
    lr: float,
    grad_clip: float,
) -> tuple[Tensor, Tensor]:
    """One optimizer step over len(batches) micro-batches. Returns (mean loss, pre-clip grad norm).

    Each micro-batch loss is divided by the number of micro-batches, so the summed
    gradient equals the gradient of one big batch. Results stay on the device;
    calling .item() forces a GPU sync, so the caller only does it when logging.
    """
    for g in opt.param_groups:
        g["lr"] = lr
    device_type = batches[0][0].device.type
    total = torch.zeros((), device=batches[0][0].device)
    for x, y in batches:
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            _, loss = model(x, targets=y)
        (loss / len(batches)).backward()
        total += loss.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total / len(batches), grad_norm


def train(
    cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    tokens: np.ndarray | None = None,
    log=print,
) -> tuple[Transformer, list[dict]]:
    """Run cfg.max_steps optimizer steps. Returns the model and the logged history."""
    model_cfg = model_cfg or get_config(cfg.preset)
    if cfg.seq_len > model_cfg.max_seq_len:
        raise ValueError(f"seq_len {cfg.seq_len} > model max_seq_len {model_cfg.max_seq_len}")
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # TF32 for any matmul left in fp32
    tokens = load_tokens("train") if tokens is None else tokens
    rng = np.random.default_rng(cfg.seed)

    model = Transformer(model_cfg).to(cfg.device)
    opt = make_optimizer(model, cfg)
    log(f"params {model.num_params():,} | {cfg.tokens_per_step:,} tokens/step "
        f"({cfg.micro_batch}x{cfg.seq_len} x {cfg.grad_accum} accum) | {cfg.max_steps} steps "
        f"= {cfg.tokens_per_step * cfg.max_steps / 1e6:.1f}M tokens")

    history: list[dict] = []
    t_last, step_last = time.perf_counter(), -1
    for step in range(cfg.max_steps):
        lr = lr_at(step, cfg)
        batches = [get_batch(tokens, cfg.micro_batch, cfg.seq_len, rng, cfg.device)
                   for _ in range(cfg.grad_accum)]
        loss, grad_norm = train_step(model, opt, batches, lr, cfg.grad_clip)
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            rec = {"step": step, "loss": loss.item(), "lr": lr, "grad_norm": grad_norm.item()}
            now = time.perf_counter()  # .item() above synced the GPU, so this timing is real
            rec["tok_per_s"] = (step - step_last) * cfg.tokens_per_step / (now - t_last)
            t_last, step_last = now, step
            history.append(rec)
            log(f"step {step:5d} | loss {rec['loss']:.4f} | lr {lr:.2e} | "
                f"grad {rec['grad_norm']:.3f} | {rec['tok_per_s'] / 1e3:.1f}K tok/s")
    return model, history


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tinyllm model")
    for f in fields(TrainConfig):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    train(TrainConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
