"""Training loop (Phase 3, hyperparameters in DECISIONS.md D-019).

- Step 3.2: AdamW, warmup + cosine LR, gradient accumulation, clipping, bf16 autocast.
- Step 3.3: val loss every eval_every steps, checkpoint save/resume, sample text.

The logging backend (3.4) and MFU (3.5) are added on top.

    uv run python -m tinyllm.train --preset smoke --max_steps 2000
    uv run python -m tinyllm.train --preset main --stop_at 3000      # train in chunks...
    uv run python -m tinyllm.train --preset main --resume            # ...and continue later
"""

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from tinyllm.config import ModelConfig, get_config
from tinyllm.data import DATA_DIR, get_batch, load_tokens
from tinyllm.generate import sample
from tinyllm.model import Transformer
from tinyllm.tokenizer import EOS, load_tokenizer

REPO = DATA_DIR.parent
EVAL_SEED = 1234                       # same val windows at every eval, so losses are comparable
SAMPLE_PROMPT = "Once upon a time"


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
    eval_every: int = 200            # val loss + checkpoint (+ sample) every N steps and at the end
    eval_batches: int = 20           # 20 x 32 x 512 = 327,680 val tokens per eval
    sample_tokens: int = 200         # tokens generated at each eval; 0 = no samples
    out_dir: str = ""                # default: checkpoints/<preset>
    resume: bool = False             # continue from <out_dir>/ckpt.pt
    stop_at: int = 0                 # stop after this many steps (0 = max_steps); the LR schedule still uses max_steps

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.seq_len

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) if self.out_dir else REPO / "checkpoints" / self.preset


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


@torch.no_grad()
def evaluate(model: nn.Module, tokens: np.ndarray, cfg: TrainConfig) -> float:
    """Mean loss over cfg.eval_batches fixed val batches (same windows every call)."""
    model.eval()
    rng = np.random.default_rng(EVAL_SEED)
    device_type = torch.device(cfg.device).type
    total = torch.zeros((), device=cfg.device)
    for _ in range(cfg.eval_batches):
        x, y = get_batch(tokens, cfg.micro_batch, cfg.seq_len, rng, cfg.device)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            _, loss = model(x, targets=y)
        total += loss
    model.train()
    return (total / cfg.eval_batches).item()


def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temp file, then rename, so a crash mid-save never corrupts the last checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: str) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def train(
    cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    tokens: np.ndarray | None = None,
    val_tokens: np.ndarray | None = None,
    log=print,
) -> tuple[Transformer, list[dict]]:
    """Run optimizer steps up to cfg.stop_at (or cfg.max_steps). Returns the model and the history.

    History rows are {"step", "loss", "lr", "grad_norm", "tok_per_s"} at log steps and
    {"step", "val_loss"} at eval steps.
    """
    run_dir = cfg.run_dir
    ckpt = load_checkpoint(run_dir / "ckpt.pt", cfg.device) if cfg.resume else None
    if ckpt is not None:
        model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])  # the checkpoint defines the model
    model_cfg = model_cfg or get_config(cfg.preset)
    if cfg.seq_len > model_cfg.max_seq_len:
        raise ValueError(f"seq_len {cfg.seq_len} > model max_seq_len {model_cfg.max_seq_len}")
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # TF32 for any matmul left in fp32
    tokens = load_tokens("train") if tokens is None else tokens
    val_tokens = load_tokens("validation") if val_tokens is None else val_tokens
    tok = load_tokenizer() if cfg.sample_tokens else None
    rng = np.random.default_rng(cfg.seed)

    model = Transformer(model_cfg).to(cfg.device)
    opt = make_optimizer(model, cfg)
    start, best_val = 0, math.inf
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        rng.bit_generator.state = ckpt["rng"]
        start, best_val = ckpt["step"] + 1, ckpt["best_val"]
        log(f"resumed from {run_dir / 'ckpt.pt'} at step {start} (best val {best_val:.4f})")
    end = min(cfg.stop_at or cfg.max_steps, cfg.max_steps)
    log(f"params {model.num_params():,} | {cfg.tokens_per_step:,} tokens/step "
        f"({cfg.micro_batch}x{cfg.seq_len} x {cfg.grad_accum} accum) | steps {start}..{end - 1} "
        f"of {cfg.max_steps} ({cfg.tokens_per_step * cfg.max_steps / 1e6:.1f}M tokens total) | {run_dir}")

    history: list[dict] = []
    t_last, step_last = time.perf_counter(), start - 1
    for step in range(start, end):
        lr = lr_at(step, cfg)
        batches = [get_batch(tokens, cfg.micro_batch, cfg.seq_len, rng, cfg.device)
                   for _ in range(cfg.grad_accum)]
        loss, grad_norm = train_step(model, opt, batches, lr, cfg.grad_clip)

        if step % cfg.log_every == 0 or step == end - 1:
            rec = {"step": step, "loss": loss.item(), "lr": lr, "grad_norm": grad_norm.item()}
            now = time.perf_counter()  # .item() above synced the GPU, so this timing is real
            rec["tok_per_s"] = (step - step_last) * cfg.tokens_per_step / (now - t_last)
            t_last, step_last = now, step
            history.append(rec)
            log(f"step {step:5d} | loss {rec['loss']:.4f} | lr {lr:.2e} | "
                f"grad {rec['grad_norm']:.3f} | {rec['tok_per_s'] / 1e3:.1f}K tok/s")

        if step % cfg.eval_every == 0 or step == end - 1:
            val = evaluate(model, val_tokens, cfg)
            history.append({"step": step, "val_loss": val})
            saved = "ckpt.pt"
            if val < best_val:
                best_val = val
                # Weights only, for sampling / export later.
                save_checkpoint(run_dir / "best.pt", {"model": model.state_dict(),
                                "model_cfg": model_cfg.to_dict(), "step": step, "val_loss": val})
                saved += " + best.pt"
            save_checkpoint(run_dir / "ckpt.pt", {
                "model": model.state_dict(), "optimizer": opt.state_dict(), "rng": rng.bit_generator.state,
                "step": step, "best_val": best_val, "model_cfg": model_cfg.to_dict(), "train_cfg": asdict(cfg),
            })
            log(f"eval  {step:5d} | val {val:.4f} | best {best_val:.4f} | saved {saved}")
            if tok is not None:
                gen = torch.Generator(cfg.device).manual_seed(cfg.seed)  # same sampling noise each eval
                prompt = torch.tensor([tok.encode(SAMPLE_PROMPT).ids], device=cfg.device)
                out = sample(model, prompt, cfg.sample_tokens, temperature=0.8, top_k=50,
                             eos_id=tok.token_to_id(EOS), generator=gen)
                log("sample | " + tok.decode(out[0].tolist()).replace("\n", " "))
            t_last, step_last = time.perf_counter(), step  # eval time doesn't count toward tok/s
    return model, history


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tinyllm model")
    for f in fields(TrainConfig):
        if isinstance(f.default, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    train(TrainConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
