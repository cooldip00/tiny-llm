"""Step 4.9c: TinyLlama logits, ours vs HF Transformers (reference only, D-015).

Ground truth is HF LlamaForCausalLM in fp32 with eager attention. Each prompt is extended by
fp32 greedy decoding, and that fixed sequence is fed to every model (teacher-forced):
  - ours, bf16 weights + autocast, prefill then one token per step through our KV cache
  - HF, bf16 weights + SDPA, the same way through HF's cache
so the "decode" rows measure the cached path, not a full forward. Free-running greedy text is
also compared (first token where ours and HF bf16 diverge).

    uv run python scripts/tinyllama_vs_hf.py [--new-tokens 64] [--out runs/bench/tinyllama_vs_hf.json]
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from tinyllm.generate import generate
from tinyllm.hf_llama import TINYLLAMA, TINYLLAMA_REV, HFTokenizer, download, load_from_dir
from tinyllm.logit_compare import first_divergence, hf_stepwise, logit_stats, ours_stepwise
from tinyllm.provenance import Provenance

CHAT = [
    "What is the capital of France? Answer in one sentence.",
    "Write a short poem about the sea.",
    "Explain what a GPU is to a ten-year-old.",
    "List three tips for writing clean Python code.",
]
PLAIN = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
         "combined movable metal type, oil-based ink and a wooden screw press. Within fifty years")


def hf_model(path, dtype, attn):
    return LlamaForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=attn).to("cuda").eval()


def free(m):
    del m
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--out", default="runs/bench/tinyllama_vs_hf.json")
    args = p.parse_args()
    N = args.new_tokens
    path = download()
    tok = HFTokenizer(path)
    prompts = [tok.encode(tok.chat_prompt(q)) for q in CHAT] + [tok.encode(PLAIN)]
    prompts = [torch.tensor([x], device="cuda") for x in prompts]
    gen = dict(max_new_tokens=N, do_sample=False, temperature=None, top_p=None, top_k=None)
    prov = Provenance(weights="ref fp32 / HF bf16 / ours bf16", script="tinyllama_vs_hf",
                      model=f"{TINYLLAMA}@{TINYLLAMA_REV[:7]}").start()

    # 1. fp32 reference: greedy continuation (the shared sequence) and its full-forward logits.
    hf = hf_model(path, torch.float32, "eager")
    seqs, truth, greedy = [], [], {"fp32": []}
    for x in prompts:
        s = hf.generate(x, **gen)
        seqs.append(s)
        greedy["fp32"].append(s[0, x.shape[1]:].tolist())
        with torch.no_grad():
            truth.append(hf(s).logits.float().cpu())
    free(hf)

    # 2. HF bf16 and 3. ours bf16, both stepwise through their caches on the same sequences.
    logits = {}
    hf = hf_model(path, torch.bfloat16, "sdpa")
    logits["hf_bf16"] = [hf_stepwise(hf, s, x.shape[1]) for s, x in zip(seqs, prompts)]
    greedy["hf_bf16"] = [hf.generate(x, **gen)[0, x.shape[1]:].tolist() for x in prompts]
    free(hf)
    ours = load_from_dir(path)
    logits["ours_bf16"] = [ours_stepwise(ours, s, x.shape[1]) for s, x in zip(seqs, prompts)]
    greedy["ours_bf16"] = []
    for x in prompts:                                     # stop at EOS like HF does, then trim the same way
        g = generate(ours, x, N, temperature=0, eos_id=tok.eos_id)[0, x.shape[1]:].tolist()
        greedy["ours_bf16"].append(g[:g.index(tok.eos_id) + 1] if tok.eos_id in g else g)
    free(ours)

    def pooled(rows, phase):
        cut = [x.shape[1] for x in prompts]
        sl = [slice(0, c) if phase == "prefill" else slice(c, None) for c in cut]
        return torch.cat([r[0, s] for r, s in zip(rows, sl)])

    stats = {}
    for phase in ("prefill", "decode"):
        ref = pooled(truth, phase)
        stats[phase] = {name: logit_stats(ref, pooled(v, phase)) for name, v in logits.items()}
        stats[phase]["ours_vs_hf_bf16"] = logit_stats(pooled(logits["hf_bf16"], phase),
                                                      pooled(logits["ours_bf16"], phase))
        stats[phase]["bit_identical_to_hf_bf16"] = torch.equal(pooled(logits["hf_bf16"], phase),
                                                               pooled(logits["ours_bf16"], phase))
        # Every top-1 flip against fp32: how close was fp32's own top-2 (a near-tie)?
        top2 = ref.topk(2, -1).values
        margin = top2[:, 0] - top2[:, 1]
        stats[phase]["flips"] = {
            name: [round(m, 4) for m in margin[pooled(v, phase).argmax(-1) != ref.argmax(-1)].tolist()]
            for name, v in logits.items()}
        stats[phase]["median_margin"] = margin.median().item()

    print(f"{TINYLLAMA}@{TINYLLAMA_REV[:7]}: {len(prompts)} prompts, {N} teacher-forced decode tokens each; "
          "reference = HF fp32 eager")
    print(f"{'phase':8} {'model':16} {'pos':>5} {'max|Δ|':>8} {'mean|Δ|':>8} {'KL mean':>9} {'KL max':>9} {'top-1':>7}  "
          "flips (fp32 top1-top2 margin)")
    for phase, d in stats.items():
        for name in ("ours_bf16", "hf_bf16", "ours_vs_hf_bf16"):
            s = d[name]
            fl = d["flips"].get(name, "")
            print(f"{phase:8} {name:16} {s['positions']:5d} {s['max_abs']:8.4f} {s['mean_abs']:8.4f} "
                  f"{s['kl_mean']:9.2e} {s['kl_max']:9.2e} {s['top1']:7.2%}  {fl}")
        print(f"{'':8} median fp32 top1-top2 margin: {d['median_margin']:.3f}; "
              f"ours bit-identical to HF bf16: {d['bit_identical_to_hf_bf16']}")
    print("\ngreedy, first divergence (token index; None = identical):")
    div = []
    for i, x in enumerate(prompts):
        a, b, c = greedy["ours_bf16"][i], greedy["hf_bf16"][i], greedy["fp32"][i]
        row = {"prompt": i, "ours_vs_hf_bf16": first_divergence(a, b), "ours_vs_fp32": first_divergence(a, c),
               "hf_bf16_vs_fp32": first_divergence(b, c), "len": [len(a), len(b), len(c)]}
        div.append(row)
        print(f"  prompt {i}: ours vs HF bf16 {row['ours_vs_hf_bf16']}, ours vs fp32 {row['ours_vs_fp32']}, "
              f"HF bf16 vs fp32 {row['hf_bf16_vs_fp32']}  (lengths {row['len']})")
    print("\nours:", repr(tok.decode(greedy["ours_bf16"][0])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": TINYLLAMA, "revision": TINYLLAMA_REV, "new_tokens": N,
                               "stats": stats, "greedy_divergence": div, "provenance": prov.finish()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
