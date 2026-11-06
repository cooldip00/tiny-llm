"""Step 4.8b: export our trained 30M model and tokenizer as an HF Llama folder, then verify it.

Run:  uv run python scripts/export_hf.py [--ckpt checkpoints/main/best.pt] [--out runs/export/tinystories-30m]

Writes to --out (git-ignored: weights never go into the repo):
  config.json, generation_config.json, model.safetensors (fp32, as trained),
  tokenizer.json, tokenizer_config.json
Anyone can then load it with `AutoModelForCausalLM` / `AutoTokenizer`, without our code.

Then checks it on CPU in fp32, with HF Transformers as the reference only (D-015):
  1. logits: HF LlamaForCausalLM (eager attention) vs ours on validation stories, both as one
     full forward and as prefill + one cached decode step per token
  2. greedy generation: HF generate() == our generate() token for token, stopping at EOS (id 0)
  3. tokenizer: AutoTokenizer gives our ids on the whole validation split, and decode gives the text back
  4. round trip: our load_from_dir() reads the folder back with identical weights
Report: <out>-verify.json, next to the folder (so the folder itself stays upload-ready).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tinyllm import hf_llama as hl
from tinyllm.config import ModelConfig
from tinyllm.generate import generate
from tinyllm.logit_compare import first_divergence, hf_stepwise, logit_stats, ours_stepwise
from tinyllm.model import Transformer
from tinyllm.provenance import git_info
from tinyllm.tokenizer import EOS, load_tokenizer

MAX_ABS = 1e-3      # fp32 vs fp32 eager: op-order rounding only. A mapping bug is orders of magnitude larger.


def load_ours(path: Path) -> Transformer:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig.from_dict(ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def check_logits(ours, hf, stories: list[list[int]], n_prefill: int) -> dict:
    full_o, full_h, step_o, step_h = [], [], [], []
    with torch.no_grad():
        for ids in stories:
            x = torch.tensor([ids])
            full_o.append(ours(x)[0].float())
            full_h.append(hf(x).logits.float())
            step_o.append(ours_stepwise(ours, x, n_prefill))
            step_h.append(hf_stepwise(hf, x, n_prefill))
    cat = lambda xs: torch.cat([t.flatten(0, 1) for t in xs])      # noqa: E731  (stories differ in length)
    return {"full_forward": logit_stats(cat(full_h), cat(full_o)),
            "prefill_then_decode": logit_stats(cat(step_h), cat(step_o)),
            "stories": len(stories), "tokens_per_story": sorted({len(s) for s in stories}), "n_prefill": n_prefill}


def check_generation(ours, hf, prompts: list[list[int]], max_new: int, eos: int) -> dict:
    rows = []
    for p in prompts:
        x = torch.tensor([p])
        a = generate(ours, x, max_new, temperature=0, eos_id=eos)[0, len(p):].tolist()
        b = hf.generate(x, max_new_tokens=max_new, do_sample=False)[0, len(p):].tolist()
        rows.append({"prompt_tokens": len(p), "ours": len(a), "hf": len(b), "equal": a == b,
                     "first_divergence": first_divergence(a, b), "stopped_at_eos": bool(a) and a[-1] == eos})
    return {"prompts": len(rows), "all_equal": all(r["equal"] for r in rows),
            "stopped_at_eos": sum(r["stopped_at_eos"] for r in rows), "max_new_tokens": max_new, "rows": rows}


def check_tokenizer(out: Path, texts: list[str]) -> dict:
    from transformers import AutoTokenizer
    ref, tok = AutoTokenizer.from_pretrained(out), load_tokenizer()
    t0 = time.perf_counter()
    ours = [e.ids for e in tok.encode_batch(texts)]
    theirs = ref(texts)["input_ids"]
    id_mismatch = [i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    text_mismatch = [i for i, (ids, s) in enumerate(zip(theirs, texts)) if ref.decode(ids) != s]
    return {"texts": len(texts), "tokens": sum(map(len, ours)), "id_mismatches": len(id_mismatch),
            "decode_mismatches": len(text_mismatch), "first_bad": (id_mismatch + text_mismatch)[:5],
            "eos_token_id": ref.eos_token_id, "bos_token_id": ref.bos_token_id,
            "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="checkpoints/main/best.pt")
    p.add_argument("--out", default="runs/export/tinystories-30m")
    p.add_argument("--stories", type=int, default=16, help="validation stories for the logit check")
    p.add_argument("--tokens", type=int, default=256, help="max tokens per story in the logit check")
    p.add_argument("--prompts", type=int, default=8, help="prompts for the greedy-generation check")
    p.add_argument("--max-new", type=int, default=200)
    args = p.parse_args()
    torch.manual_seed(0)
    out = Path(args.out)

    from transformers import LlamaForCausalLM, __version__ as tf_version   # reference only (D-015)

    from tinyllm.data import load_clean

    ours = load_ours(Path(args.ckpt))
    tok = load_tokenizer()
    eos = tok.token_to_id(EOS)
    hl.save_as_hf(ours, out, eos_id=eos)
    hl.save_tokenizer_as_hf(out, model_max_length=ours.cfg.max_seq_len)
    files = {f.name: f.stat().st_size for f in sorted(out.iterdir())}
    print(f"wrote {out}: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in files.items()))

    hf = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32, attn_implementation="eager").eval()
    back = hl.load_from_dir(out, device="cpu", dtype=torch.float32)
    same = all(torch.equal(v, back.state_dict()[k]) for k, v in ours.state_dict().items())

    val = list(load_clean("validation")["text"])          # a datasets Column; transformers wants a list
    enc = [e.ids for e in tok.encode_batch(val[:max(args.stories, args.prompts) * 4])]
    long = [ids for ids in enc if len(ids) >= 32]
    stories = [ids[:args.tokens] for ids in long[:args.stories]]
    prompts = [ids[:12] for ids in long[args.stories:args.stories + args.prompts]]   # unseen by the logit check

    t0 = time.perf_counter()
    logits = check_logits(ours, hf, stories, n_prefill=8)
    gen = check_generation(ours, hf, prompts, args.max_new, eos)
    toks = check_tokenizer(out, val)
    elapsed = round(time.perf_counter() - t0, 1)

    ok = {
        "logits_full": logits["full_forward"]["max_abs"] <= MAX_ABS,
        "logits_decode": logits["prefill_then_decode"]["max_abs"] <= MAX_ABS,
        "generation": gen["all_equal"],
        "tokenizer": toks["id_mismatches"] == 0 and toks["decode_mismatches"] == 0,
        "round_trip": same,
        "hf_config_ids": hf.config.eos_token_id == eos and hf.generation_config.eos_token_id == eos,
    }
    report = {"ckpt": args.ckpt, "out": str(out), "files_bytes": files, "model_cfg": ours.cfg.to_dict(),
              "params": sum(t.numel() for t in ours.parameters()), "max_abs_threshold": MAX_ABS,
              "checks": ok, "passed": all(ok.values()), "logits": logits, "generation": gen, "tokenizer": toks,
              "seconds": elapsed, "provenance": {"git": git_info(), "torch": torch.__version__,
                                                  "transformers": tf_version, "device": "cpu", "dtype": "float32"}}
    Path(f"{out}-verify.json").write_text(json.dumps(report, indent=2))

    f, d = logits["full_forward"], logits["prefill_then_decode"]
    print(f"logits ({logits['stories']} stories, {f['positions']} positions): full max|Δ| {f['max_abs']:.2e} "
          f"top1 {f['top1']:.4f} KL {f['kl_mean']:.1e} | prefill+decode max|Δ| {d['max_abs']:.2e} top1 {d['top1']:.4f}")
    print(f"greedy: {sum(r['equal'] for r in gen['rows'])}/{gen['prompts']} identical, "
          f"{gen['stopped_at_eos']} stopped at EOS, lengths {[r['ours'] for r in gen['rows']]}")
    print(f"tokenizer: {toks['texts']:,} stories, {toks['tokens']:,} tokens, id mismatches {toks['id_mismatches']}, "
          f"decode mismatches {toks['decode_mismatches']}")
    print(f"round trip identical: {same} | HF eos id {hf.config.eos_token_id}, bos {hf.config.bos_token_id}")
    print(("PASS" if report["passed"] else "FAIL " + str([k for k, v in ok.items() if not v]))
          + f"  ({elapsed}s)  -> {out}-verify.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
