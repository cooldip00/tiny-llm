"""Step 1.1: download TinyStories and print basic statistics.

The data lands in the Hugging Face cache (~/.cache/huggingface), not in the repo.
Later steps (tokenizer training, tokenization) load it from there.

Run:  uv run python scripts/download_data.py
"""

import pyarrow.compute as pc
from datasets import load_dataset

DATASET = "roneneldan/TinyStories"


def main() -> None:
    ds = load_dataset(DATASET)
    print(ds)

    for split, d in ds.items():
        text = d.data.column("text")  # pyarrow column: stats without loading into Python
        lengths = pc.utf8_length(text)
        empty = pc.sum(pc.equal(pc.utf8_length(pc.utf8_trim_whitespace(text)), 0)).as_py()
        non_ascii = pc.sum(pc.match_substring_regex(text, r"[^\x00-\x7F]")).as_py()
        total = pc.sum(lengths).as_py()
        print(f"\n== {split} ==")
        print(f"stories        {len(d):,}")
        print(f"characters     {total:,}  (~{total / 1e9:.2f} GB of text)")
        print(f"avg chars      {total / len(d):,.0f}   min {pc.min(lengths).as_py()}   max {pc.max(lengths).as_py():,}")
        print(f"empty          {empty:,}")
        print(f"non-ASCII      {non_ascii:,} stories ({non_ascii / len(d):.2%})")

    print("\n== sample (train[0]) ==")
    print(ds["train"][0]["text"][:500])
    print(f"\ncache files: {ds['train'].cache_files[0]['filename']}")


if __name__ == "__main__":
    main()
