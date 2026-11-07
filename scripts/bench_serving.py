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

Step 4.9d: --model tinyllama runs the same workload shape on TinyLlama (bf16), with the prompts
cut from the validation text re-tokenized by its tokenizer (bench_generate.tinyllama_val_tokens).

Step 4.11b: --cuda-graph adds "cont+graph" rows: continuous batching with the decode step
replayed as CUDA graphs. One decoder per slot count, every bucket captured before the timed
runs (capture_s is reported apart, not in wall time) and freed after its row.

Step 6.8: the default output name carries the row-adding flags (serving[_graph][_paged].json),
and an existing output is refused unless --overwrite. Before, --paged / --cuda-graph runs all
wrote serving.json, and that is how the 11-04 TinyLlama serving JSON was lost.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_generate import use_model, val_tokens  # noqa: E402

from tinyllm.cuda_graph import SlotGraphDecoder
from tinyllm.engine import Request, run_continuous, run_static
from tinyllm.generate import generate
from tinyllm.kv_cache import SlotKVCache
from tinyllm.model import norm_backend
from tinyllm.provenance import GpuMonitor, Provenance, claim_output


def make_workload(n: int, seed: int) -> list[Request]:
    rng = np.random.default_rng(seed)
    val = val_tokens()
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


def default_out(tiny: bool, cuda_graph: bool, paged: bool) -> Path:
    name = "serving" + ("_graph" if cuda_graph else "") + ("_paged" if paged else "")
    return Path(f"runs/bench/{'tinyllama/' if tiny else ''}{name}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/main/best.pt"))
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="8,32", help="static batch sizes = continuous slot counts")
    ap.add_argument("--repeats", type=int, default=3, help="runs per config; the median-wall run is reported")
    ap.add_argument("--paged", action="store_true",
                    help="4.5: add paged-KV rows with the 8-slot cache's memory budget")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="4.11b: add continuous rows with the decode step as CUDA graphs")
    ap.add_argument("--model", default=None, help="'tinyllama' instead of --ckpt (4.9d)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs/bench[/tinyllama]/serving[_graph][_paged].json")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = ap.parse_args()
    tiny = args.model == "tinyllama"
    if args.model not in (None, "tinyllama"):
        ap.error("--model takes 'tinyllama'; use --ckpt for a checkpoint")
    args.out = claim_output(args.out or default_out(tiny, args.cuda_graph, args.paged), args.overwrite)
    device = "cuda"
    model = use_model("tinyllama" if tiny else str(args.ckpt), device)
    reqs = make_workload(args.requests, args.seed)
    print(f"{len(reqs)} requests: prompt {min(len(r.prompt) for r in reqs)}-"
          f"{max(len(r.prompt) for r in reqs)}, new {min(r.max_new_tokens for r in reqs)}-"
          f"{max(r.max_new_tokens for r in reqs)}, {sum(r.max_new_tokens for r in reqs)} tokens to generate")

    warm = make_workload(8, args.seed + 1)                    # kernels, allocator, autotune
    run_static(model, warm, 8)
    run_continuous(model, warm, 8)
    if args.paged:
        run_continuous(model, warm, 8, kv="paged")

    print("solo reference (generate, batch 1) ...", flush=True)
    ref = solo_tokens(model, reqs, device)

    runs = []                                                 # (label, size, fn, kwargs)
    for size in (int(s) for s in args.sizes.split(",")):
        runs += [("static", size, run_static, {}), ("continuous", size, run_continuous, {})]
        if args.cuda_graph:
            runs.append(("cont+graph", size, run_continuous, {"cuda_graph": "build"}))
    if args.paged:
        # Budget = the 8-slot cache's memory: 8 × 512 positions = 4096 tokens (64 MiB in bf16).
        budget = 8 * model.cfg.max_seq_len
        runs += [(f"paged/{bs}", s, run_continuous, dict(kv="paged", block_size=bs, kv_blocks=budget // bs))
                 for s, bs in ((8, 16), (32, 16), (32, 8), (32, 32), (64, 16))]

    rows = []
    prov = Provenance(weights=f"{'bf16' if tiny else 'fp32'} + bf16 autocast", script="bench_serving",
                      repeats=args.repeats, model="tinyllama" if tiny else str(args.ckpt),
                      norm_backend=norm_backend()).start()
    for name, size, fn, kw in runs:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        capture_s = None
        if kw.get("cuda_graph") == "build":                   # one decoder per row, freed after it
            cfg = model.cfg
            dec = SlotGraphDecoder(model, SlotKVCache(cfg.n_layers, size, cfg.n_kv_heads, cfg.head_dim,
                                                      cfg.max_seq_len, torch.bfloat16, device))
            capture_s = dec.capture()
            run_continuous(model, warm, size, cuda_graph=dec)  # warm the eager prefill shapes too
            kw = {"cuda_graph": dec}
        mon = GpuMonitor().start()  # per row, so an outlier row can be matched to clocks / throttling
        trials = [fn(model, reqs, size, **kw) for _ in range(args.repeats)]
        gpu_during = mon.stop()
        kw = dec = None
        torch.cuda.empty_cache()
        walls = [s["wall_s"] for _, s in trials]
        results, stats = sorted(trials, key=lambda t: t[1]["wall_s"])[len(trials) // 2]  # median run
        stats.update(engine=name, size=size, wall_all_s=walls,
                     peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                     correctness=agreement([r.tokens for r in results], ref), gpu_during=gpu_during,
                     capture_s=capture_s)
        rows.append(stats)
        c = stats["correctness"]
        alloc = stats["kv_alloc_used_mean"]
        print(f"{name:>10} {size:>3} | {stats['tok_s']:7.0f} tok/s | wall {stats['wall_s']:6.2f}s | "
              f"lat mean {stats['latency_mean_s']:5.2f} p95 {stats['latency_p95_s']:5.2f}s | "
              f"ttft mean {stats['ttft_mean_s']:5.2f} p95 {stats['ttft_p95_s']:5.2f}s | "
              f"steps {stats['decode_steps']:5d} | rows {stats['row_util']:.0%} | "
              f"KV {stats['kv_reserved_mib']:5.1f} MiB used {stats['kv_used_mean']:.0%}"
              f"/{stats['kv_used_peak']:.0%} | of claimed {'-' if alloc is None else f'{alloc:.0%}'} | "
              f"active {stats['peak_active']} | preempt {stats['preemptions']} | "
              f"VRAM {stats['peak_vram_mib']:.0f} MiB | "
              f"same as solo {c['identical']}/{c['of']}", flush=True)
    args.out.write_text(json.dumps({"workload": {"requests": len(reqs), "seed": args.seed},
                                    "provenance": prov.finish(), "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
