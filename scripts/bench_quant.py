"""Step 4.6: INT8 weight-only quantization, quality and speed vs floating point.

Three variants of checkpoints/main/best.pt, all run under bf16 autocast:
  fp32      weights stored fp32, autocast casts them to bf16 at every matmul (the engine default so far)
  bf16      weights stored bf16 (model.to(bfloat16)): no per-step cast
  int8      tinyllm.quant: matmul weights int8 + per-channel fp32 scales; embedding/norms stay fp32

Quality:
  val loss       train.evaluate on the same fixed val windows (EVAL_SEED) for every variant
  vs exact fp32  teacher-forced on 32 sequences of 128 val tokens + 256 greedy tokens: next-token argmax
                 agreement and mean KL(fp32 || variant) against the model in plain fp32 (no autocast)
  greedy         free-running greedy from the same prompts: rows identical to the fp32-autocast run
Speed: bench_generate.bench with the KV engine (P=128, 256 new tokens, 3 warmups, median of 5),
interleaved over variants, the whole sweep repeated --rounds times; the median round is reported.

    uv run python scripts/bench_quant.py [--rounds 3] [--eval-batches 20]

Step 4.9d: --model tinyllama. Variants bf16 (the published weights, so the baseline) and int8; an
fp32 copy (4.4 GB) beside them does not fit in 8 GB. The "exact" reference is the bf16 model
without autocast. Val loss is on the validation text re-tokenized with TinyLlama's tokenizer, 8
sequences x 512 tokens per batch (4x the batches, same token count), teacher-forced on 8 sequences.

Step 4.10b: --triton adds variant int8-triton, the same int8 tensors (shared, not copied) through
the Triton kernel (tinyllm.kernels.int8_gemv) at decode. Its prefill and the val-loss /
teacher-forced passes are prefill-sized, so they take the torch path and match int8 by
construction; the decode-path check is its greedy rows. --engine kv-graph times every variant with
CUDA graphs (the D-016 judge). Default output names carry both: int8_graph_triton.json etc.
Without the flags, the run and its output name are as before.

While one variant is timed, the others wait in host memory (`place`), as in real use where one
model is loaded. Before 2026-11-07 the bf16 weights (2.1 GB) stayed on the GPU while int8 was
timed, and int8 at batch 32 reserved ~8.7 GB of the 8 GB card: WSL spilled to host memory.
Quality checks still run with every variant on the GPU (they are not timed).
"""

import argparse
import copy
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import bench_generate  # noqa: E402
from bench_generate import bench, kv_greedy, make_prompts, use_model, val_tokens  # noqa: E402

from tinyllm.generate import generate
from tinyllm.provenance import Provenance, claim_output
from tinyllm.quant import quantize_model, set_backend, weight_bytes
from tinyllm.train import TrainConfig, evaluate

MiB = 2**20


@torch.no_grad()
def teacher_forced(variants, ref, seq, prompt_len):
    """Next-token argmax agreement and mean KL(ref || variant) on positions after the prompt."""
    ref_logp = F.log_softmax(ref(seq)[0][:, prompt_len - 1:-1].float(), -1)
    top2 = ref_logp.topk(2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]                                    # log-prob margin of the fp32 argmax
    out = {}
    for name, m in variants.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp = F.log_softmax(m(seq)[0][:, prompt_len - 1:-1].float(), -1)
        diff = logp.argmax(-1) != ref_logp.argmax(-1)
        out[name] = {
            "argmax_agree": 1 - diff.float().mean().item(),
            "kl_mean": (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean().item(),
            "max_gap_at_flip": gap[diff].max().item() if diff.any() else None,
        }
    out["_median_gap"] = gap.median().item()
    return out


def default_out(tiny: bool, engine: str, triton: bool) -> str:
    """runs/bench[/tinyllama]/int8[_graph][_triton].json: each mode its own file (6.8)."""
    tag = ("_graph" if engine == "kv-graph" else "") + ("_triton" if triton else "")
    return f"runs/bench/{'tinyllama/' if tiny else ''}int8{tag}.json"


def shared_copy(model, backend: str):
    """A second module tree over the same tensors (no weight copy), with its Int8Linear backend
    switched. A separate object, so bench_generate's per-model GraphDecoder cache can't mix them."""
    memo = {id(t): t for t in itertools.chain(model.parameters(), model.buffers())}
    twin = copy.deepcopy(model, memo)
    set_backend(twin, backend)
    assert all(a is b for a, b in zip(model.buffers(), twin.buffers()))
    assert all(a is b for a, b in zip(model.parameters(), twin.parameters()))
    return twin


def tensors(m) -> list:
    return list(itertools.chain(m.parameters(), m.buffers()))


def place(models, device: str) -> None:
    """Move every parameter and buffer of `models` to `device` in place (t.data), so module trees
    that share tensors (int8 / int8-triton) move together and stay shared."""
    seen = set()
    for m in models:
        for t in tensors(m):
            if id(t) not in seen:
                seen.add(id(t))
                t.data = t.data.to(device)


def time_alone(variants: dict, groups: dict, name: str, device: str) -> None:
    """Only `name`'s weight group on `device`; every other group in host memory."""
    place([m for k, m in variants.items() if groups[k] != groups[name]], "cpu")
    torch.cuda.empty_cache()
    place([m for k, m in variants.items() if groups[k] == groups[name]], device)
    bench_generate._DECODERS.clear()    # graphs captured before a move point at freed memory


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/main/best.pt")
    p.add_argument("--batch-sizes", default="1,8,32")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--model", default=None, help="'tinyllama' instead of --ckpt (4.9d)")
    p.add_argument("--engine", default="kv", choices=("kv", "kv-graph"), help="timed engine (4.10b)")
    p.add_argument("--triton", action="store_true", help="add variant int8-triton (4.10b)")
    p.add_argument("--out", default=None, help="default: runs/bench[/tinyllama]/int8[_graph][_triton].json")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out (6.8)")
    args = p.parse_args()
    tiny = args.model == "tinyllama"
    if args.model not in (None, "tinyllama"):
        p.error("--model takes 'tinyllama'; use --ckpt for a checkpoint")
    if args.triton:
        try:
            from tinyllm.kernels import int8_gemv
        except ImportError as e:
            p.error(f"--triton needs Triton: {e}")
    out = claim_output(args.out or default_out(tiny, args.engine, args.triton), args.overwrite)
    dev = "cuda"

    if tiny:
        bf16 = use_model("tinyllama", dev)
        int8 = copy.deepcopy(bf16)
        n_q = quantize_model(int8)
        variants, base = {"bf16": bf16, "int8": int8}, "bf16"
    else:
        fp32 = use_model(args.ckpt, dev)
        bf16 = copy.deepcopy(fp32).to(torch.bfloat16)
        int8 = copy.deepcopy(fp32)
        n_q = quantize_model(int8)
        variants, base = {"fp32": fp32, "bf16": bf16, "int8": int8}, "fp32"
    if args.triton:
        variants["int8-triton"] = shared_copy(int8, "triton")
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | {n_q} Linear layers -> Int8Linear | "
          f"engine {args.engine} | variants {', '.join(variants)}")

    sizes = {k: {kind: b / MiB for kind, b in weight_bytes(m).items()} for k, m in variants.items()}
    for k, s in sizes.items():
        print(f"{k:>11} stored weights: matmul {s['matmul']:6.1f} MiB | embedding {s['embedding']:5.1f} MiB | "
              f"other {s['other']:.2f} MiB")

    # Correctness gate: the timed loop on each int8 model reproduces generate(temperature=0).
    check = make_prompts(2, 32, dev)
    for k in (k for k in variants if k.startswith("int8")):
        assert torch.equal(kv_greedy(variants[k], check, 64)[0], generate(variants[k], check, 64, temperature=0))
        print(f"check: {k} timed kv loop == generate(temperature=0): OK")

    val = val_tokens()
    cfg = (TrainConfig(eval_batches=4 * args.eval_batches, micro_batch=8) if tiny    # same tokens, fits 8 GB
           else TrainConfig(eval_batches=args.eval_batches))
    losses = {k: evaluate(m, val, cfg) for k, m in variants.items()}
    print(f"val loss ({cfg.eval_batches} x {cfg.micro_batch} x {cfg.seq_len} tokens): "
          + " | ".join(f"{k} {v:.4f}" for k, v in losses.items()) + " | "
          + ", ".join(f"{k} - {base} = {losses[k] - losses[base]:+.4f}" for k in variants if k != base))

    n_seq = 8 if tiny else 32
    prompts = make_prompts(n_seq, args.prompt_len, dev)
    greedy = {k: kv_greedy(m, prompts, args.new_tokens)[0] for k, m in variants.items()}
    same = {k: int((g == greedy[base]).all(1).sum()) for k, g in greedy.items()}
    tf = teacher_forced(variants, variants[base], greedy[base], args.prompt_len)
    print(f"vs plain {base} (teacher-forced, {n_seq} x {args.new_tokens} tokens, median top-1 margin "
          f"{tf['_median_gap']:.2f} nats):")
    for k in variants:
        t = tf[k]
        gap = "-" if t["max_gap_at_flip"] is None else f"{t['max_gap_at_flip']:.3f}"
        print(f"  {k:>11}: argmax agree {t['argmax_agree']:.2%} | KL {t['kl_mean']:.2e} nats | "
              f"largest margin at a flip {gap} | greedy rows identical to {base}-autocast {same[k]}/{n_seq}")

    batches = list(map(int, args.batch_sizes.split(",")))
    runs = {k: {b: [] for b in batches} for k in variants}
    groups = {k: "int8" if k == "int8-triton" else k for k in variants}      # variants sharing weights
    for g in set(groups.values()):                                           # groups must not share
        mine = {id(t) for k in variants if groups[k] == g for t in tensors(variants[k])}
        rest = {id(t) for k in variants if groups[k] != g for t in tensors(variants[k])}
        assert not mine & rest, f"weight group {g} shares tensors with another group"
    prov = Provenance(weights=("bf16 / int8 (per row), bf16 autocast" if tiny else
                               "fp32 + bf16 autocast / bf16 / int8 (per row)"), script="bench_quant",
                      rounds=args.rounds, model="tinyllama" if tiny else args.ckpt,
                      engine=args.engine, triton=args.triton).start()
    for r in range(args.rounds):
        for b in batches:
            for k, m in variants.items():
                time_alone(variants, groups, k, dev)
                runs[k][b].append(bench(m, args.engine, b, args.prompt_len, args.new_tokens, 3, 5, dev))
        print(f"round {r + 1}/{args.rounds}: " + " | ".join(
            f"B={b} " + "/".join(f"{runs[k][b][-1]['decode_tok_s']:.0f}" for k in variants) for b in batches)
            + f"  ({'/'.join(variants)} decode tok/s)", flush=True)

    provenance = prov.finish()
    rows = []
    for b in batches:
        for k in variants:
            rs = sorted(runs[k][b], key=lambda x: x["decode_tok_s"])
            med = rs[len(rs) // 2]
            rows.append({"variant": k, **med, "decode_tok_s_rounds": [x["decode_tok_s"] for x in runs[k][b]]})
            ref = next(x for x in rows if x["variant"] == base and x["batch"] == b)
            print(f"B={b:>2} {k:>11} | decode {med['decode_tok_s']:7.0f} tok/s ({med['ms_per_step']:.2f} ms/step, "
                  f"{med['decode_tok_s'] / ref['decode_tok_s']:.2f}x {base}, "
                  f"{med['pct_of_ceiling']:.1f}% of its ceiling) | TTFT {med['ttft_ms']:.2f} ms | "
                  f"rounds {[round(x) for x in rows[-1]['decode_tok_s_rounds']]}")

    extra = {"triton_configs": int8_gemv.best_configs()} if args.triton else {}
    out.write_text(json.dumps({"model": "tinyllama" if tiny else args.ckpt, "baseline": base, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                               "engine": args.engine, "timed_alone": True, "provenance": provenance, "weights_mib": sizes, "val_loss": losses,
                               "eval_batches": args.eval_batches, "teacher_forced": tf, "greedy_rows_identical": same,
                               "rows": rows, **extra}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
