"""Extract the main run's loss curve from its log into a small committed JSON (6.2, D-030).

checkpoints/ is git-ignored, so without this file the README's training plot could not be redrawn from the
repo. bench_all.py draws docs/img/train_loss.svg from it.

    uv run python scripts/extract_train_curve.py      # checkpoints/main/train.log -> docs/data/train_curve.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEADER = re.compile(r"^params ([\d,]+) \| ([\d,]+) tokens/step")
STEP = re.compile(r"^step\s+(\d+) \| loss ([\d.]+)")
EVAL = re.compile(r"^eval\s+(\d+) \| val ([\d.]+)")


def parse(text: str) -> dict:
    params = tokens_per_step = None
    train, val = {}, {}
    for line in text.splitlines():
        if m := HEADER.match(line):
            params, tokens_per_step = (int(g.replace(",", "")) for g in m.groups())
        elif m := STEP.match(line):
            train[int(m[1])] = float(m[2])          # a resumed run logs a step again: keep the last one
        elif m := EVAL.match(line):
            val[int(m[1])] = float(m[2])
    if not (params and tokens_per_step and train and val):
        raise SystemExit("no header, step or eval lines found: is this a tinyllm.train log?")
    return {"params": params, "tokens_per_step": tokens_per_step,
            "train": [[s, train[s]] for s in sorted(train)], "val": [[s, val[s]] for s in sorted(val)]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", type=Path, default=REPO / "checkpoints/main/train.log")
    p.add_argument("--out", type=Path, default=REPO / "docs/data/train_curve.json")
    args = p.parse_args()
    raw = args.log.read_bytes()
    curve = parse(raw.decode("utf-8", errors="replace"))
    out = {"source": "checkpoints/main/train.log", "source_sha256": hashlib.sha256(raw).hexdigest(), **curve}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {args.out.relative_to(REPO)}: {len(curve['train'])} train / {len(curve['val'])} val points, "
          f"final val {curve['val'][-1][1]} at step {curve['val'][-1][0]}")


if __name__ == "__main__":
    main()
