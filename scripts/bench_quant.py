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
"""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bench_generate import bench, kv_greedy, load_model, make_prompts  # noqa: E402

from tinyllm.data import load_tokens
from tinyllm.generate import generate
from tinyllm.provenance import Provenance
from tinyllm.quant import quantize_model, weight_bytes
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/main/best.pt")
    p.add_argument("--batch-sizes", default="1,8,32")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--out", default="runs/bench/int8.json")
    args = p.parse_args()
    dev = "cuda"

    fp32 = load_model(Path(args.ckpt), dev)
    bf16 = copy.deepcopy(fp32).to(torch.bfloat16)
    int8 = copy.deepcopy(fp32)
    n_q = quantize_model(int8)
    variants = {"fp32": fp32, "bf16": bf16, "int8": int8}
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | {n_q} Linear layers -> Int8Linear")

    sizes = {k: {kind: b / MiB for kind, b in weight_bytes(m).items()} for k, m in variants.items()}
    for k, s in sizes.items():
        print(f"{k:>4} stored weights: matmul {s['matmul']:6.1f} MiB | embedding {s['embedding']:5.1f} MiB | "
              f"other {s['other']:.2f} MiB")

    # Correctness gate: the timed loop on the int8 model reproduces generate(temperature=0).
    check = make_prompts(2, 32, dev)
    assert torch.equal(kv_greedy(int8, check, 64)[0], generate(int8, check, 64, temperature=0))
    print("check: int8 timed kv loop == generate(temperature=0): OK")

    val = load_tokens("validation")
    cfg = TrainConfig(eval_batches=args.eval_batches)
    losses = {k: evaluate(m, val, cfg) for k, m in variants.items()}
    print("val loss (" + f"{args.eval_batches} x {cfg.micro_batch} x {cfg.seq_len} tokens): "
          + " | ".join(f"{k} {v:.4f}" for k, v in losses.items())
          + f" | int8 - fp32 = {losses['int8'] - losses['fp32']:+.4f}, bf16 - fp32 = {losses['bf16'] - losses['fp32']:+.4f}")

    prompts = make_prompts(32, args.prompt_len, dev)
    greedy = {k: kv_greedy(m, prompts, args.new_tokens)[0] for k, m in variants.items()}
    same = {k: int((g == greedy["fp32"]).all(1).sum()) for k, g in greedy.items()}
    tf = teacher_forced(variants, fp32, greedy["fp32"], args.prompt_len)
    print(f"vs plain fp32 (teacher-forced, 32 x {args.new_tokens} tokens, median top-1 margin "
          f"{tf['_median_gap']:.2f} nats):")
    for k in variants:
        t = tf[k]
        gap = "-" if t["max_gap_at_flip"] is None else f"{t['max_gap_at_flip']:.3f}"
        print(f"  {k:>4}: argmax agree {t['argmax_agree']:.2%} | KL {t['kl_mean']:.2e} nats | "
              f"largest margin at a flip {gap} | greedy rows identical to fp32-autocast {same[k]}/32")

    batches = list(map(int, args.batch_sizes.split(",")))
    runs = {k: {b: [] for b in batches} for k in variants}
    prov = Provenance(weights="fp32 + bf16 autocast / bf16 / int8 (per row)", script="bench_quant",
                      rounds=args.rounds).start()
    for r in range(args.rounds):
        for b in batches:
            for k, m in variants.items():
                runs[k][b].append(bench(m, "kv", b, args.prompt_len, args.new_tokens, 3, 5, dev))
        print(f"round {r + 1}/{args.rounds}: " + " | ".join(
            f"B={b} " + "/".join(f"{runs[k][b][-1]['decode_tok_s']:.0f}" for k in variants) for b in batches)
            + "  (fp32/bf16/int8 decode tok/s)", flush=True)

    provenance = prov.finish()
    rows = []
    for b in batches:
        for k in variants:
            rs = sorted(runs[k][b], key=lambda x: x["decode_tok_s"])
            med = rs[len(rs) // 2]
            rows.append({"variant": k, **med, "decode_tok_s_rounds": [x["decode_tok_s"] for x in runs[k][b]]})
            base = next(x for x in rows if x["variant"] == "fp32" and x["batch"] == b)
            print(f"B={b:>2} {k:>4} | decode {med['decode_tok_s']:7.0f} tok/s ({med['ms_per_step']:.2f} ms/step, "
                  f"{med['decode_tok_s'] / base['decode_tok_s']:.2f}x fp32) | TTFT {med['ttft_ms']:.2f} ms | "
                  f"rounds {[round(x) for x in rows[-1]['decode_tok_s_rounds']]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                               "provenance": provenance, "weights_mib": sizes, "val_loss": losses, "eval_batches": args.eval_batches,
                               "teacher_forced": tf, "greedy_rows_identical": same, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
