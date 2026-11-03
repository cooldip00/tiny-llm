"""Step 1.2 prep: apply the D-018 cleanup to both splits and report what changed.

Run:  uv run python scripts/check_clean.py
"""

import pyarrow.compute as pc
from datasets import load_dataset

from tinyllm.data import DATASET, load_clean


def main() -> None:
    for split in ("train", "validation"):
        raw = load_dataset(DATASET, split=split)
        clean = load_clean(split)
        text = clean.data.column("text")
        non_ascii = pc.sum(pc.match_substring_regex(text, r"[^\x00-\x7F]")).as_py()
        chars = pc.sum(pc.utf8_length(text)).as_py()
        dropped = len(raw) - len(clean)
        print(f"== {split} ==")
        print(f"raw stories     {len(raw):,}")
        print(f"kept            {len(clean):,}")
        print(f"dropped         {dropped:,} ({dropped / len(raw):.3%})")
        print(f"characters      {chars:,}")
        print(f"non-ASCII left  {non_ascii}")
        assert non_ascii == 0

    clean = load_clean("train")
    hits = pc.filter(clean.data.column("text"),
                     pc.match_substring_regex(clean.data.column("text"), "Be careful not to touch it"))
    if len(hits):
        s = hits[0].as_py()
        i = s.index("Be careful")
        print("\nexample after cleanup:", repr(s[max(0, i - 40): i + 60]))


if __name__ == "__main__":
    main()
