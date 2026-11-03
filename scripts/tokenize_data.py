"""Steps 1.3 + 1.4: tokenize cleaned TinyStories into data/train.bin and data/val.bin,
then sanity-check the files.

Run:  uv run python scripts/tokenize_data.py
"""

import time

import numpy as np

from tinyllm.data import bin_path, load_clean, load_tokens, write_tokens
from tinyllm.tokenizer import EOS, load_tokenizer


def main() -> None:
    tok = load_tokenizer()
    eos = tok.token_to_id(EOS)

    for split in ("validation", "train"):
        t0 = time.perf_counter()
        n_stories, n_tokens = write_tokens(split, tok, eos)
        dt = time.perf_counter() - t0
        path = bin_path(split)

        # --- 1.4 sanity checks, on the file as written to disk ---
        arr = load_tokens(split)
        assert arr.dtype == np.uint16 and len(arr) == n_tokens
        eos_pos = np.flatnonzero(arr == eos)
        assert len(eos_pos) == n_stories, "one EOS per story"
        assert eos_pos[-1] == len(arr) - 1, "file ends with EOS"
        assert int(arr.max()) < tok.get_vocab_size()

        # Decode the first, a middle, and the last story straight from the memmap.
        clean = load_clean(split)
        starts = np.concatenate([[0], eos_pos[:-1] + 1])
        for k in (0, n_stories // 2, n_stories - 1):
            ids = arr[starts[k]: eos_pos[k]].tolist()
            assert tok.decode(ids) == clean[k]["text"], f"roundtrip failed at story {k}"

        lens = np.diff(np.concatenate([[-1], eos_pos])) - 1  # tokens per story, EOS excluded
        print(f"== {split} -> {path.name} ==")
        print(f"stories         {n_stories:,}")
        print(f"tokens          {n_tokens:,}  (incl. {n_stories:,} EOS)")
        print(f"file size       {path.stat().st_size / 1e6:,.1f} MB")
        print(f"tokens/story    avg {lens.mean():.0f}   median {np.median(lens):.0f}   "
              f"p99 {np.percentile(lens, 99):.0f}   max {lens.max()}")
        print(f"over 512 ctx    {(lens + 1 > 512).mean():.2%} of stories")
        print(f"time            {dt:.0f}s   roundtrip checks OK\n")


if __name__ == "__main__":
    main()
