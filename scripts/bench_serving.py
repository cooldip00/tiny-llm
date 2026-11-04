"""Step 4.4 benchmark: static vs continuous batching on a mixed offline workload.

Workload (fixed seed): 64 requests, all queued at t=0. Prompts are validation windows of
32-256 tokens, max_new_tokens 16-256 (uniform), greedy, no EOS. That length spread is what
static batching handles badly: each batch decodes until its longest request finishes.

Reports per engine: useful tok/s (generated tokens that belong to a request / wall time),
mean and p95 latency and TTFT, decode steps, row utilisation (live rows / computed rows),
and KV cache reserved MiB with used fraction (mean, peak).

Correctness (bf16 on CUDA): each engine's tokens are compared with solo generate() per
request. bf16 near-ties may flip a token, after which the rows diverge (as in 4.3), so the
report gives identical requests and the first divergence instead of asserting equality.

    uv run python scripts/bench_serving.py [--ckpt checkpoints/main/best.pt] [--requests 64]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_generate import load_model  # noqa: E402

from tinyllm.data import load_tokens
from tinyllm.engine import Request, run_continuous, run_static
from tinyllm.generate import generate


def make_workload(n: int, seed: int) -> list[Request]:
    rng = np.random.default_rng(seed)
    val = load_tokens("validation")
    reqs = []
    for _ in range(n):
        t = int(rng.integers(32, 257))
        s = int(rng.integers(0, len(val) - t))
        reqs.append(Request(val[s:s + t].astype(np.int64).tolist(), int(rng.integers(16, 257))))
    return reqs


def solo_tokens(model, reqs, device) -> list[list[int]]:
    out = []
    for r in reqs:
        idx = torch.tensor([r.prompt], device=device)
        out.append(generate(model, idx, r.max_new_tokens, temperature=0.0)[0, len(r.prompt):].tolist())
    return out


def agreement(got: list[list[int]], ref: list[list[int]]) -> dict:
    first_div = []
    for g, r in zip(got, ref):
        d = next((i for i, (a, b) in enumerate(zip(g, r)) if a != b), None)
        if d is not None:
            first_div.append(d)
    return {"identical": len(got) - len(first_div), "of": len(got), "first_divergence": sorted(first_div)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/main/best.pt"))
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="8,32", help="static batch sizes = continuous slot counts")
    ap.add_argument("--out", type=Path, default=Path("runs/bench/serving.json"))
    args = ap.parse_args()
    device = "cuda"
    model = load_model(args.ckpt, device)
    reqs = make_workload(args.requests, args.seed)
    print(f"{len(reqs)} requests: prompt {min(len(r.prompt) for r in reqs)}-"
          f"{max(len(r.prompt) for r in reqs)}, new {min(r.max_new_tokens for r in reqs)}-"
          f"{max(r.max_new_tokens for r in reqs)}, {sum(r.max_new_tokens for r in reqs)} tokens to generate")

    warm = make_workload(8, args.seed + 1)                    # kernels, allocator, autotune
    run_static(model, warm, 8)
    run_continuous(model, warm, 8)

    print("solo reference (generate, batch 1) ...", flush=True)
    ref = solo_tokens(model, reqs, device)

    rows = []
    for size in (int(s) for s in args.sizes.split(",")):
        for name, fn in (("static", run_static), ("continuous", run_continuous)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            results, stats = fn(model, reqs, size)
            stats.update(engine=name, size=size,
                         peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                         correctness=agreement([r.tokens for r in results], ref))
            rows.append(stats)
            c = stats["correctness"]
            print(f"{name:>10} {size:>3} | {stats['tok_s']:7.0f} tok/s | wall {stats['wall_s']:6.2f}s | "
                  f"lat mean {stats['latency_mean_s']:5.2f} p95 {stats['latency_p95_s']:5.2f}s | "
                  f"ttft mean {stats['ttft_mean_s']:5.2f} p95 {stats['ttft_p95_s']:5.2f}s | "
                  f"steps {stats['decode_steps']:5d} | rows {stats['row_util']:.0%} | "
                  f"KV {stats['kv_reserved_mib']:5.1f} MiB used {stats['kv_used_mean']:.0%}"
                  f"/{stats['kv_used_peak']:.0%} | VRAM {stats['peak_vram_mib']:.0f} MiB | "
                  f"same as solo {c['identical']}/{c['of']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"workload": {"requests": len(reqs), "seed": args.seed},
                                    "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
