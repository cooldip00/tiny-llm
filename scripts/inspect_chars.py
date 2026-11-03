"""Step 1.2 prep: which non-ASCII characters appear in TinyStories, and how often?

Decides whether we need text cleanup before training the tokenizer.
Run:  uv run python scripts/inspect_chars.py
"""

import unicodedata
from collections import Counter

import pyarrow.compute as pc
from datasets import load_dataset


def main() -> None:
    train = load_dataset("roneneldan/TinyStories", split="train")
    text = train.data.column("text")
    mask = pc.match_substring_regex(text, r"[^\x00-\x7F]")
    stories = pc.filter(text, mask).to_pylist()

    counts: Counter[str] = Counter()
    for s in stories:
        counts.update(c for c in s if ord(c) > 127)

    total = sum(counts.values())
    print(f"{len(stories):,} stories, {total:,} non-ASCII chars, {len(counts):,} distinct\n")
    print(f"{'char':<6}{'U+':<8}{'count':>10}  {'share':>6}  name")
    for c, n in counts.most_common(40):
        name = unicodedata.name(c, "?")
        print(f"{c!r:<6}{ord(c):<8X}{n:>10,}  {n / total:6.1%}  {name}")

    # Mojibake check: UTF-8 bytes decoded as cp1252 look like 'â€™' (for ’).
    moji = sum(1 for s in stories if "â€" in s or "Ã" in s)
    print(f"\nstories with mojibake patterns ('â€' or 'Ã'): {moji:,}")
    for s in stories:
        if "â€" in s:
            i = s.index("â€")
            print("example:", repr(s[max(0, i - 60): i + 40]))
            break


if __name__ == "__main__":
    main()
