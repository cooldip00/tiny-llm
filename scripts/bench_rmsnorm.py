"""Step 4.10a: the Triton RMSNorm and residual add + RMSNorm, alone and in TinyLlama decode.

1. Kernels alone (TinyLlama width 2,048, bf16), at decode rows (1, 8, 32) and a prefill-sized
   4,096 rows (32 x 128). Two operations, each two ways:
     norm      torch: tinyllm.model.RMSNorm's PyTorch code     triton: rmsnorm()
     add+norm  torch: h = x + a, then the norm                 triton: add_rmsnorm()
   Each is checked against the PyTorch result, then timed as --calls calls captured in one CUDA
   graph (no launch overhead, as in the kv-graph engine), with bench_int8_gemv.time_calls. Bandwidth
   is the bytes the operation must move (norm: read x and w, write y; add+norm: read x, a and w,
   write h and y) over its time, as % of 423 GB/s. At decode sizes a row is 4 KB and sits in L2,
   so those times are fixed costs and their GB/s is not a bandwidth figure; only the 4,096-row
   numbers are.
2. Correctness in the model: on backend "triton", the timed kv loop reproduces
   generate(temperature=0); then greedy rows identical to backend "torch", and the teacher-forced
   logit gap and argmax agreement on the torch-backend greedy sequences (8 x 256 tokens).
3. One graph decode step at B=1 under the profiler (profile_decode.measure), per backend: GPU ops
   per step, GPU busy time per step and idle fraction.
4. End to end: kv-graph decode (P=128, 256 new tokens, 3 warmups, median of 5) at each batch size,
   backends interleaved, the sweep repeated --rounds times; the median round is reported. The
   graphs are recaptured after every backend switch (a captured graph keeps the kernels it saw).

    uv run python scripts/bench_rmsnorm.py [--rounds 3] [--out runs/bench/tinyllama/rmsnorm_graph.json]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import bench_generate  # noqa: E402
from bench_generate import BANDWIDTH, bench, kv_greedy, make_prompts, use_model  # noqa: E402
from bench_int8_gemv import time_calls  # noqa: E402
from profile_decode import measure  # noqa: E402

from tinyllm.cuda_graph import GraphDecoder
from tinyllm.generate import generate
from tinyllm.kernels.rmsnorm import add_rmsnorm, rmsnorm
from tinyllm.model import set_norm_backend
from tinyllm.provenance import Provenance, claim_output

D, EPS = 2048, 1e-5
BACKENDS = ("torch", "triton")


def torch_norm(x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return xf.to(x.dtype) * w


def kernels_alone(rows_list, n_calls, iters, repeats) -> list[dict]:
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for rows in rows_list:
        x = (torch.randn(rows, D, device="cuda", generator=g) * 3).to(torch.bfloat16)
        a = torch.randn(rows, D, device="cuda", generator=g).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
        ops = {
            "norm": ({"torch": lambda: torch_norm(x, w), "triton": lambda: rmsnorm(x, w, EPS)},
                     2 * rows * D * 2 + D * 2),
            "add+norm": ({"torch": lambda: (lambda h: (torch_norm(h, w), h))(x + a),
                          "triton": lambda: add_rmsnorm(x, a, w, EPS)},
                         4 * rows * D * 2 + D * 2),
        }
        for op, (fns, nbytes) in ops.items():
            ref, got = fns["torch"](), fns["triton"]()
            if op == "add+norm":
                assert torch.equal(got[1], ref[1]), "add+norm: h differs from x + a"
                ref, got = ref[0], got[0]
            torch.testing.assert_close(got, ref)
            r = {"op": op, "rows": rows, "bytes": nbytes,
                 "exact_frac": (got == ref).float().mean().item()}
            for k, fn in fns.items():
                sec = time_calls([fn], n_calls, iters, repeats)
                gbs = nbytes / sec / 1e9
                r[f"{k}_us"], r[f"{k}_gb_s"], r[f"{k}_pct_bw"] = sec * 1e6, gbs, 100 * gbs * 1e9 / BANDWIDTH
            r["speedup"] = r["torch_us"] / r["triton_us"]
            out.append(r)
            print(f"{op:>8} rows={rows:>4} | torch {r['torch_us']:7.2f} us {r['torch_pct_bw']:5.1f}% | "
                  f"triton {r['triton_us']:7.2f} us {r['triton_pct_bw']:5.1f}% | {r['speedup']:.2f}x | "
                  f"bitwise equal {r['exact_frac']:.2%}", flush=True)
    return out


@torch.no_grad()
def logit_gap(model, seqs) -> dict:
    """Teacher-forced on seqs: max |logit| gap and argmax agreement, triton vs torch backend."""
    gaps, agree, total, scale = [], 0, 0, 0.0
    for i in range(0, seqs.shape[0], 2):
        s = seqs[i:i + 2]
        out = {}
        for k in BACKENDS:
            set_norm_backend(model, k)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[k] = model(s)[0].float()
        gaps.append((out["triton"] - out["torch"]).abs().max().item())
        scale = max(scale, out["torch"].abs().max().item())
        agree += (out["triton"].argmax(-1) == out["torch"].argmax(-1)).sum().item()
        total += out["torch"].shape[0] * out["torch"].shape[1]
        del out
    return {"max_abs_gap": max(gaps), "logit_scale": scale, "argmax_agree": agree / total}


def use_backend(model, k: str) -> None:
    set_norm_backend(model, k)
    bench_generate._DECODERS.clear()      # captured graphs keep the old backend's kernels
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", default="1,8,32")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rows", default="1,8,32,4096", help="rows per kernel call (part 1)")
    p.add_argument("--calls", type=int, default=20, help="calls per captured graph (part 1)")
    p.add_argument("--iters", type=int, default=20, help="graph replays per timing (part 1)")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--out", default="runs/bench/tinyllama/rmsnorm_graph.json")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out (6.8)")
    args = p.parse_args()
    out = claim_output(args.out, args.overwrite)
    dev = "cuda"
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | bandwidth {BANDWIDTH / 1e9:.0f} GB/s",
          flush=True)
    batches = list(map(int, args.batch_sizes.split(",")))
    prov = Provenance(weights="bf16 (published)", script="bench_rmsnorm", model="tinyllama",
                      engine="kv-graph", rounds=args.rounds).start()

    print("1. kernels alone (width 2048, bf16, CUDA-graph timed):")
    alone = kernels_alone(list(map(int, args.rows.split(","))), args.calls, args.iters, args.repeats)

    model = use_model("tinyllama", dev)
    n_norms = set_norm_backend(model, "triton")
    with torch.no_grad():   # the backend falls back silently without the kernel: make sure it's there
        assert model.norm._kernel(torch.zeros(1, D, device=dev, dtype=torch.bfloat16)) is not None
    check = make_prompts(2, 32, dev)
    assert torch.equal(kv_greedy(model, check, 64)[0], generate(model, check, 64, temperature=0))
    print(f"2. {n_norms} RMSNorms on triton | check: timed kv loop == generate(temperature=0): OK")
    prompts = make_prompts(8, args.prompt_len, dev)
    greedy = {}
    for k in BACKENDS:
        set_norm_backend(model, k)
        greedy[k] = kv_greedy(model, prompts, args.new_tokens)[0]
    same = int((greedy["triton"] == greedy["torch"]).all(1).sum())
    tf = logit_gap(model, greedy["torch"])
    print(f"   greedy rows identical to torch backend {same}/{prompts.shape[0]} | teacher-forced: "
          f"argmax agree {tf['argmax_agree']:.2%}, max logit gap {tf['max_abs_gap']:.3f} "
          f"(max |logit| {tf['logit_scale']:.1f})", flush=True)

    print("3. one graph decode step, B=1, under the profiler:")
    prof = {}
    prompt1 = make_prompts(1, args.prompt_len, dev)
    for k in BACKENDS:
        use_backend(model, k)
        r = measure(GraphDecoder(model, 1), prompt1, 100, out.parent / "rmsnorm_profile" / f"B1_graph_{k}.json")
        prof[k] = r
        print(f"   {k:>6}: {r['ms_per_step']:6.2f} ms/step | GPU busy {r['gpu_busy_ms_per_step']:6.2f} ms | "
              f"{r['gpu_ops_per_step']:5.0f} GPU ops | idle {r['gpu_idle_frac']:5.1%} | "
              f"SM {r['sm_clock_mhz']} MHz", flush=True)
    bench_generate._DECODERS.clear()
    torch.cuda.empty_cache()

    print("4. end to end, kv-graph:")
    runs = {k: {b: [] for b in batches} for k in BACKENDS}
    for rnd in range(args.rounds):
        for b in batches:
            for k in BACKENDS:
                use_backend(model, k)
                runs[k][b].append(bench(model, "kv-graph", b, args.prompt_len, args.new_tokens, 3, 5, dev))
        print(f"   round {rnd + 1}/{args.rounds}: " + " | ".join(
            f"B={b} " + "/".join(f"{runs[k][b][-1]['decode_tok_s']:.0f}" for k in BACKENDS) for b in batches)
            + "  (torch/triton decode tok/s)", flush=True)
    set_norm_backend(model, "torch")
    provenance = prov.finish()

    rows = []
    for b in batches:
        for k in BACKENDS:
            rs = sorted(runs[k][b], key=lambda x: x["decode_tok_s"])
            med = rs[len(rs) // 2]
            rows.append({"variant": k, **med, "decode_tok_s_rounds": [x["decode_tok_s"] for x in runs[k][b]]})
        t, n = rows[-2], rows[-1]
        print(f"   B={b:>2} | torch {t['decode_tok_s']:6.0f} tok/s ({t['ms_per_step']:.2f} ms/step) | "
              f"triton {n['decode_tok_s']:6.0f} tok/s ({n['ms_per_step']:.2f} ms/step) | "
              f"{n['decode_tok_s'] / t['decode_tok_s']:.3f}x | TTFT {t['ttft_ms']:.1f} / {n['ttft_ms']:.1f} ms | "
              f"rounds {[round(x) for x in t['decode_tok_s_rounds']]} / {[round(x) for x in n['decode_tok_s_rounds']]}")

    out.write_text(json.dumps({"model": "tinyllama", "engine": "kv-graph", "gpu": torch.cuda.get_device_name(),
                               "torch": torch.__version__, "bandwidth": BANDWIDTH, "provenance": provenance,
                               "n_norms": n_norms, "kernels_alone": alone, "greedy_rows_identical": same,
                               "n_greedy_rows": prompts.shape[0], "teacher_forced": tf, "profile_b1": prof,
                               "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
