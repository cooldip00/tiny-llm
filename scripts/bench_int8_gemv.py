"""Step 4.10b: the Triton INT8 matmul alone, per TinyLlama Linear shape and decode batch M.

Three ways to compute y = x @ W.T for x (M, K) bf16 and W (N, K):
  bf16    F.linear(x, w)                       W stored bf16 (the published weights)
  torch   F.linear(x, q.to(bf16)) * scale      Int8Linear backend "torch": cast, matmul, scale
  triton  int8_matmul(x, q, scale)             Int8Linear backend "triton": one kernel

Each is checked against the fp64 reference, then timed as --calls back-to-back calls captured in
one CUDA graph (no Python launch overhead, as in the kv-graph engine), replayed --iters times
between CUDA events; the median over --repeats is reported. The calls cycle through enough weight
copies (>= 64 MiB) that the 4 MiB L2 cache can't serve the weights. Bandwidth is the bytes the
call has to move at minimum (weights as stored + x + y) over its time, as % of the measured
423 GB/s (bench_generate.BANDWIDTH). "step" sums one decode step's matmuls: 22 layers x 7 Linear
+ lm_head. The autotuned Triton configs are saved with the results.

    uv run python scripts/bench_int8_gemv.py [--m 1,8,16,32,64] [--out runs/bench/tinyllama/int8_gemv.json]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bench_generate import BANDWIDTH  # noqa: E402

from tinyllm.kernels.int8_gemv import MAX_M, best_configs, int8_matmul
from tinyllm.provenance import Provenance, claim_output
from tinyllm.quant import quantize_per_channel

# TinyLlama 1.1B Linear shapes (N, K) and how many of each one decode step runs
SHAPES = {"q/o": ((2048, 2048), 44), "k/v": ((256, 2048), 44), "gate/up": ((5632, 2048), 44),
          "down": ((2048, 5632), 22), "lm_head": ((32000, 2048), 1)}
L2_FLUSH = 64 * 2**20
VARIANTS = ("bf16", "torch", "triton")


def min_bytes(variant: str, n: int, k: int, m: int) -> int:
    w = n * k * 2 if variant == "bf16" else n * k + n * 4      # int8 weights + fp32 scales
    y = m * n * (2 if variant == "bf16" else 4)                # Int8Linear outputs fp32
    return w + m * k * 2 + y


def make_calls(variant: str, n: int, k: int, m: int, gen) -> tuple[list, torch.Tensor]:
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    q, s = quantize_per_channel(w)
    x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
    ref = x.double() @ (q.double() * s.double()[:, None]).T
    copies = max(2, -(-L2_FLUSH // (n * k)))                   # int8 bytes per copy; bf16 gets 2x the margin
    if variant == "bf16":
        wb = (q.to(torch.bfloat16) * s.to(torch.bfloat16)[:, None])
        ws = [wb] + [wb.clone() for _ in range(copies - 1)]
        calls = [lambda w=w_: F.linear(x, w) for w_ in ws]
    elif variant == "torch":
        qs = [q] + [q.clone() for _ in range(copies - 1)]
        calls = [lambda q=q_: F.linear(x, q.to(torch.bfloat16)) * s for q_ in qs]
    else:
        qs = [q] + [q.clone() for _ in range(copies - 1)]
        calls = [lambda q=q_: int8_matmul(x, q, s) for q_ in qs]
    return calls, ref


def check(calls, ref, variant: str) -> float:
    rel = ((calls[0]().double() - ref).abs().max() / ref.abs().max()).item()
    limit = 2e-2 if variant == "bf16" else 1e-2              # bf16 weights round W; int8 paths are exact up to bf16 out
    assert rel < limit, f"{variant}: max error {rel:.2e} of max |y| (limit {limit})"
    return rel


def time_calls(calls, n_calls: int, iters: int, repeats: int) -> float:
    """Median seconds per call, from CUDA-graph replays of n_calls calls."""
    seq = [calls[i % len(calls)] for i in range(n_calls)]
    side = torch.cuda.Stream()                                  # warm up (cuBLAS, Triton autotune) first
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in seq:
            c()
    for _ in range(3):
        g.replay()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / (iters * n_calls))
    del g
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m", default="1,8,16,32,64", help=f"rows per call (decode batch), each <= {MAX_M}")
    p.add_argument("--calls", type=int, default=20, help="calls per captured graph")
    p.add_argument("--iters", type=int, default=20, help="graph replays per timing")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--out", default="runs/bench/tinyllama/int8_gemv.json")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out (6.8)")
    args = p.parse_args()
    ms = list(map(int, args.m.split(",")))
    if max(ms) > MAX_M:
        p.error(f"--m values must be <= {MAX_M}")
    out = claim_output(args.out, args.overwrite)
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | bandwidth {BANDWIDTH / 1e9:.0f} GB/s",
          flush=True)

    gen = torch.Generator(device="cuda").manual_seed(0)
    prov = Provenance(script="bench_int8_gemv", m=ms, calls=args.calls, iters=args.iters,
                      repeats=args.repeats).start()
    rows = []
    for name, ((n, k), count) in SHAPES.items():
        for m in ms:
            r = {"shape": name, "n": n, "k": k, "m": m, "per_step": count}
            for v in VARIANTS:
                calls, ref = make_calls(v, n, k, m, gen)
                r[f"{v}_rel_err"] = check(calls, ref, v)
                sec = time_calls(calls, args.calls, args.iters, args.repeats)
                gbs = min_bytes(v, n, k, m) / sec / 1e9
                r[f"{v}_us"], r[f"{v}_gb_s"], r[f"{v}_pct_bw"] = sec * 1e6, gbs, 100 * gbs * 1e9 / BANDWIDTH
                del calls
                torch.cuda.empty_cache()
            r["triton_vs_torch"] = r["torch_us"] / r["triton_us"]
            r["triton_vs_bf16"] = r["bf16_us"] / r["triton_us"]
            rows.append(r)
            print(f"{name:>7} {n:>5}x{k:<5} M={m:>2} | " + " | ".join(
                f"{v} {r[f'{v}_us']:7.1f} us {r[f'{v}_pct_bw']:5.1f}%" for v in VARIANTS)
                + f" | triton {r['triton_vs_torch']:.2f}x torch, {r['triton_vs_bf16']:.2f}x bf16", flush=True)

    step = {}
    for m in ms:
        step[m] = {v: sum(r[f"{v}_us"] * r["per_step"] for r in rows if r["m"] == m) for v in VARIANTS}
        s = step[m]
        print(f"decode step matmuls, M={m:>2}: " + " | ".join(f"{v} {s[v] / 1e3:6.2f} ms" for v in VARIANTS)
              + f" | triton {s['torch'] / s['triton']:.2f}x torch, {s['bf16'] / s['triton']:.2f}x bf16")
    provenance = prov.finish()
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                               "bandwidth": BANDWIDTH, "provenance": provenance, "rows": rows,
                               "step_us": {str(m): s for m, s in step.items()},
                               "triton_configs": best_configs()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
