"""Step 4.1: benchmark naive generation (no KV cache). Baseline for every later Phase 4 step.

Naive = each new token runs a full forward over the whole context (sliding window of
max_seq_len), so step t costs O(t^2) attention and O(t) matmul work. Greedy only.

Protocol (docs/INFERENCE.md): prompts 32/128/512 tokens, generate 256, batch 1/8/32,
3 warmup runs, torch.cuda.synchronize around timers, median of 5 timed runs.
  prefill tok/s = B * prompt_len / TTFT     (TTFT = first forward + first argmax)
  decode  tok/s = B * (new_tokens - 1) / time after the first token
  peak VRAM     = torch.cuda.max_memory_allocated (weights + activations)

Prompts are real validation text (fixed offsets), so the outputs can serve as the
greedy-identical reference for 4.2+. The timed loop is checked against
tinyllm.generate.sample(temperature=0) before any timing.

    uv run python scripts/bench_generate.py [--ckpt checkpoints/main/best.pt]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from tinyllm.config import ModelConfig
from tinyllm.data import load_tokens
from tinyllm.generate import sample
from tinyllm.model import Transformer
from tinyllm.train import load_checkpoint


def load_model(path: Path, device: str) -> Transformer:
    ckpt = load_checkpoint(path, device)
    model = Transformer(ModelConfig.from_dict(ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def make_prompts(batch: int, prompt_len: int, device: str) -> torch.Tensor:
    """`batch` windows of validation tokens at fixed, evenly spaced offsets."""
    val = load_tokens("validation")
    starts = np.linspace(0, len(val) - prompt_len - 1, batch, dtype=np.int64)
    rows = np.stack([val[s:s + prompt_len].astype(np.int64) for s in starts])
    return torch.from_numpy(rows).to(device)


@torch.no_grad()
def naive_greedy(model: Transformer, prompt: torch.Tensor, new_tokens: int):
    """Same computation as sample(temperature=0), with a timer after the first token."""
    max_len = model.cfg.max_seq_len
    idx = prompt
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_first = t0
    for i in range(new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx[:, -max_len:])
        nxt = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if i == 0:
            torch.cuda.synchronize()
            t_first = time.perf_counter()
    torch.cuda.synchronize()
    return idx, t_first - t0, time.perf_counter() - t_first


def bench(model, batch, prompt_len, new_tokens, warmup, repeats, device) -> dict:
    prompt = make_prompts(batch, prompt_len, device)
    for _ in range(warmup):
        naive_greedy(model, prompt, new_tokens)
    torch.cuda.reset_peak_memory_stats()
    ttfts, decodes = [], []
    for _ in range(repeats):
        _, ttft, dec = naive_greedy(model, prompt, new_tokens)
        ttfts.append(ttft)
        decodes.append(dec)
    ttft, dec = statistics.median(ttfts), statistics.median(decodes)
    return {
        "batch": batch, "prompt_len": prompt_len, "new_tokens": new_tokens,
        "ttft_ms": ttft * 1e3,
        "prefill_tok_s": batch * prompt_len / ttft,
        "decode_tok_s": batch * (new_tokens - 1) / dec,
        "ms_per_step": dec / (new_tokens - 1) * 1e3,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "window_slides": prompt_len + new_tokens > model.cfg.max_seq_len,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/main/best.pt")
    p.add_argument("--prompt-lens", default="32,128,512")
    p.add_argument("--batch-sizes", default="1,8,32")
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default="runs/bench/4.1_naive.json")
    args = p.parse_args()
    device = "cuda"

    model = load_model(Path(args.ckpt), device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | "
          f"{n_params / 1e6:.1f}M params | max_seq_len {model.cfg.max_seq_len}")

    # Correctness gate: the timed loop must reproduce the project's sampler exactly.
    check = make_prompts(2, 32, device)
    ours, _, _ = naive_greedy(model, check, 64)
    ref = sample(model, check, 64, temperature=0)
    assert torch.equal(ours, ref), "timed loop diverges from sample(temperature=0)"
    print("check: timed loop == sample(temperature=0) on 2x32 prompt, 64 tokens: OK")

    rows = []
    for b in map(int, args.batch_sizes.split(",")):
        for plen in map(int, args.prompt_lens.split(",")):
            r = bench(model, b, plen, args.new_tokens, args.warmup, args.repeats, device)
            rows.append(r)
            print(f"B={b:>2} P={plen:>3} | TTFT {r['ttft_ms']:7.2f} ms | "
                  f"prefill {r['prefill_tok_s']:9.0f} tok/s | decode {r['decode_tok_s']:7.0f} tok/s "
                  f"({r['ms_per_step']:.2f} ms/step) | peak {r['peak_vram_mib']:6.0f} MiB"
                  + (" | window slides" if r["window_slides"] else ""), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "gpu": torch.cuda.get_device_name(),
                               "torch": torch.__version__, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
