"""Step 4.11: profile one KV-cached decode step, eager vs CUDA graph.

For each mode: ms/step (wall, synced), host issue time per step (Python + launches for one
step, started with an empty GPU queue), GPU busy time per step (sum of GPU activity from the
profiler), GPU activities per step, the GPU idle fraction 1 - busy / span (both from the
profiled steps, which run slower than unprofiled ones), and the median SM clock and power
over ~2 s of decoding (nvidia-smi every 100 ms). Writes a
Chrome trace per mode (open in chrome://tracing or ui.perfetto.dev) and a JSON summary.

Weights are bf16 (D-023). Prompt tokens are random: the cost of a step doesn't depend on
which tokens they are. Decode tok/s is reported as a % of the bandwidth ceiling
(weight bytes / 423 GB/s measured in Phase 0.5) for B = 1.

    uv run python scripts/profile_decode.py [--model checkpoints/main/best.pt | tinyllama]
        [--batch 1] [--ctx 128]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from tinyllm.config import ModelConfig
from tinyllm.cuda_graph import GraphDecoder
from tinyllm.kv_cache import KVCache
from tinyllm.model import RotaryEmbedding, Transformer
from tinyllm.provenance import GpuMonitor, Provenance, claim_output
from tinyllm.train import load_checkpoint

BANDWIDTH = 423e9           # bytes/s, Phase 0.5 measurement (docs/HARDWARE.md)


def load(name: str) -> Transformer:
    if name == "tinyllama":
        from tinyllm import hf_llama as hl
        return hl.load_from_dir(hl.download(local_only=True))
    ckpt = load_checkpoint(Path(name), "cuda")
    model = Transformer(ModelConfig.from_dict(ckpt["model_cfg"])).cuda()
    model.load_state_dict(ckpt["model"])
    model.to(torch.bfloat16)
    cfg = model.cfg                                  # RoPE tables stay fp32 (as hf_llama does)
    model.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta).cuda()
    return model.eval()


class Eager:
    """The decode loop of generate(): one model() call per step under autocast."""

    def __init__(self, model: Transformer):
        self.model = model

    def prefill(self, idx):
        self.cache = KVCache(self.model.cfg.n_layers, self.model.cfg.max_seq_len)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(idx, cache=self.cache)[0]

    def step(self, tok):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(tok, cache=self.cache)[0]


def run_steps(dec, logits, n):
    for _ in range(n):
        logits = dec.step(logits[:, -1].float().argmax(-1, keepdim=True))
    return logits


@torch.no_grad()
def measure(dec, prompt, steps: int, trace: Path) -> dict:
    # Clocks while decoding: ~2 s of steps (re-prefilling as the context fills) under
    # nvidia-smi. This also captures the graphs and lets the clocks settle before timing.
    mon = GpuMonitor(interval_ms=100).start()
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        run_steps(dec, dec.prefill(prompt), steps)
        torch.cuda.synchronize()
    smi = mon.stop()
    pick = lambda k: (smi.get(k) or {}).get("median")                 # noqa: E731

    logits = run_steps(dec, dec.prefill(prompt), 10)
    # Host cost of issuing one step with an empty GPU queue (Python + launches, no waiting).
    issue = []
    for _ in range(20):
        torch.cuda.synchronize()
        t = time.perf_counter()
        logits = run_steps(dec, logits, 1)
        issue.append(time.perf_counter() - t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = run_steps(dec, logits, steps)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n_prof = 20
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run_steps(dec, logits, n_prof)
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    busy_us = sum(e.time_range.elapsed_us() for e in gpu)
    starts = [e.time_range.start for e in gpu]
    ends = [e.time_range.end for e in gpu]
    span_us = max(ends) - min(starts)                             # first to last GPU activity
    trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))
    return {
        "sm_clock_mhz": pick("clocks.sm"),
        "power_w": pick("power.draw"),
        "throttle_reasons": smi.get("throttle_reasons_seen"),
        "ms_per_step": wall / steps * 1e3,
        "host_issue_ms_per_step": statistics.median(issue) * 1e3,
        "gpu_busy_ms_per_step": busy_us / n_prof / 1e3,
        "gpu_ops_per_step": len(gpu) / n_prof,
        "gpu_idle_frac": 1 - busy_us / span_us,
        "trace": str(trace),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/main/best.pt", help="a checkpoint path, or 'tinyllama'")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ctx", type=int, default=128, help="prompt length before the profiled steps")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out-dir", default="runs/profile")
    p.add_argument("--overwrite", action="store_true", help="replace an existing summary (6.8)")
    args = p.parse_args()
    name = "tinyllama" if args.model == "tinyllama" else Path(args.model).parent.name
    out = claim_output(Path(args.out_dir) / f"decode_{name}_B{args.batch}.json", args.overwrite)

    model = load(args.model)
    weight_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
    ceiling = BANDWIDTH / weight_bytes                            # tok/s per row
    assert args.ctx + 10 + 20 + args.steps + 20 <= model.cfg.max_seq_len, "ctx + steps exceed max_seq_len"
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, model.cfg.vocab_size, (args.batch, args.ctx), generator=g).cuda()
    print(f"{name}: {weight_bytes / 2**20:.0f} MiB bf16 weights, ceiling {ceiling:.0f} tok/s per row "
          f"| B={args.batch} ctx={args.ctx}")

    prov = Provenance(weights="bf16 + bf16 autocast", script="profile_decode", model=name).start()
    rows = {}
    for mode, dec in [("eager", Eager(model)), ("graph", GraphDecoder(model, args.batch))]:
        r = measure(dec, prompt, args.steps, Path(args.out_dir) / f"{name}_B{args.batch}_{mode}.json")
        r["decode_tok_s"] = args.batch * 1e3 / r["ms_per_step"]
        r["pct_of_ceiling"] = 100 * r["decode_tok_s"] / (ceiling * args.batch)
        rows[mode] = r
        print(f"{mode:>5}: {r['ms_per_step']:6.2f} ms/step | host issue {r['host_issue_ms_per_step']:6.3f} ms | "
              f"GPU busy {r['gpu_busy_ms_per_step']:6.2f} ms | {r['gpu_ops_per_step']:5.0f} GPU ops | "
              f"GPU idle {r['gpu_idle_frac']:5.1%} | SM {r['sm_clock_mhz']} MHz {r['power_w']} W | "
              f"{r['decode_tok_s']:7.0f} tok/s "
              f"({r['pct_of_ceiling']:.1f}% of ceiling at this B)", flush=True)
    speedup = rows["eager"]["ms_per_step"] / rows["graph"]["ms_per_step"]
    print(f"graph speedup: {speedup:.2f}x")
    out.write_text(json.dumps({"model": name, "batch": args.batch, "ctx": args.ctx,
                               "weight_bytes": weight_bytes, "ceiling_tok_s_per_row": ceiling,
                               "speedup": speedup, "modes": rows, "provenance": prov.finish()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
