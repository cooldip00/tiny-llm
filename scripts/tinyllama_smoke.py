"""Step 4.9b: load TinyLlama-1.1B-Chat v1.0 into our Transformer and chat with it (greedy, KV cache).

Downloads the pinned revision into the HF cache on first run (~2.1 GB), never into the repo.
Prints parameter count, load time, VRAM, and one greedy answer per prompt. Logits matching
the HF reference is 4.9c; this is the "it loads and talks" check.

    uv run python scripts/tinyllama_smoke.py [--new-tokens 96]
"""

import argparse
import time

import torch

from tinyllm.generate import generate
from tinyllm.hf_llama import TINYLLAMA, TINYLLAMA_REV, HFTokenizer, download, load_from_dir

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a short poem about the sea.",
    "Explain what a GPU is to a ten-year-old.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-tokens", type=int, default=96)
    args = p.parse_args()

    t = time.perf_counter()
    path = download()
    print(f"{TINYLLAMA}@{TINYLLAMA_REV[:7]} -> {path} ({time.perf_counter() - t:.1f}s incl. cache check)")

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    model = load_from_dir(path, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    n = model.num_params()
    print(f"loaded in {load_s:.1f}s | {n:,} params (estimate_params {model.cfg.estimate_params():,}) | "
          f"dtypes {sorted({str(p.dtype) for p in model.parameters()})} | "
          f"VRAM allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"layers {model.cfg.layer_types[0]} x {model.cfg.n_layers}, heads {model.cfg.n_heads}/{model.cfg.n_kv_heads}, "
          f"max_seq_len {model.cfg.max_seq_len}")

    tok = HFTokenizer(path)
    for q in PROMPTS:
        ids = torch.tensor([tok.encode(tok.chat_prompt(q))], device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = generate(model, ids, args.new_tokens, temperature=0.0, eos_id=tok.eos_id)
        torch.cuda.synchronize()
        new = out[0, ids.shape[1]:].tolist()
        dt = time.perf_counter() - t
        print(f"\n### {q}\n{tok.decode(new).strip()}\n[{len(new)} tokens in {dt:.2f}s = {len(new) / dt:.0f} tok/s, "
              f"eos {'yes' if tok.eos_id in new else 'no'}; unoptimised eager, see 4.11 / 4.9d]")


if __name__ == "__main__":
    main()
