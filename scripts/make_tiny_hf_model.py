"""A tiny random model in the 4.8b export format, for checking the server and the Docker image (6.5b).

Run:  python scripts/make_tiny_hf_model.py --out /tmp/tiny-model

Same files as scripts/export_hf.py (our TinyStories tokenizer, EOS id 0), but ~0.3M random
weights, so CI can start the server without downloading anything. Its text is gibberish.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinyllm.config import ModelConfig
from tinyllm.hf_llama import save_as_hf, save_tokenizer_as_hf
from tinyllm.model import Transformer
from tinyllm.tokenizer import EOS, load_tokenizer

TINY = dict(vocab_size=8192, max_seq_len=128, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64)


def make(out: Path, seed: int = 0) -> Path:
    eos_id = load_tokenizer().token_to_id(EOS)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Transformer(ModelConfig(**TINY)).eval()
    save_as_hf(model, out, eos_id=eos_id)
    save_tokenizer_as_hf(out, model_max_length=TINY["max_seq_len"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(f"wrote {make(args.out, args.seed)}")


if __name__ == "__main__":
    main()
