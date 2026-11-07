"""Extract the load sweeps (ours vs vLLM vs llama.cpp, 8 sequences each) into a small committed JSON (D-030).

runs/ is git-ignored, so without this file the README's load plot could not be redrawn from the repo.
bench_all.py draws docs/img/load_sweep.svg from it. Per-request timings stay in runs/; this keeps one
summary row per request rate. The default --dir is the latest sweep (4.10c, rerun with D-032's Triton
norm); the 4.8a sweep is kept in runs/bench/load/4.8a.

    uv run python scripts/extract_load_sweep.py [--dir runs/bench/load/4.10c]   # *-sweep.json -> docs/data/load_sweep.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# file stem -> label; the equal-seat comparison of docs/SERVING.md (4.8a, rerun in 4.10c)
ENGINES = {"ours-8slots": "Ours", "vllm-8seqs": "vLLM", "llamacpp-bf16": "llama.cpp"}


def summarize(d: dict) -> dict:
    pts = []
    for r in d["rows"]:
        s = r["summary"]
        pts.append([s["rate"], round(s["achieved_rps"], 3), round(s["tok_s"], 1), round(s["ttft_s"]["p95"], 4),
                    round(s["shed_rate"], 3)])
    p = d.get("provenance") or {}
    return {"rated_load_rps": d["rated_load_rps"], "points": pts,
            "commit": ((p.get("git") or {}).get("commit") or "?")[:7], "time_utc": p.get("time_utc")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, default=REPO / "runs/bench/load/4.10c")
    p.add_argument("--out", type=Path, default=REPO / "docs/data/load_sweep.json")
    args = p.parse_args()
    src = args.dir.resolve()
    src = src.relative_to(REPO).as_posix() if src.is_relative_to(REPO) else src.name   # never a personal path
    out = {"source": src, "columns": ["rate_rps", "achieved_rps", "tok_s", "ttft_p95_s", "shed_rate"],
           "engines": {}}
    for stem, label in ENGINES.items():
        raw = (args.dir / f"{stem}-sweep.json").read_bytes()
        d = json.loads(raw)
        out.setdefault("slo_ttft_p95_s", d["slo"]["ttft_p95_s"])
        out.setdefault("workload", {k: d["workload"][k] for k in ("duration", "max_tokens")})
        out["engines"][label] = {"file": f"{stem}-sweep.json", "sha256": hashlib.sha256(raw).hexdigest(), **summarize(d)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)  # then one line per innermost list: one row per request rate
    text = re.sub(r"\[\s+([^\[\]{}]+?)\s+\]", lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    args.out.write_text(text + "\n")
    print(f"wrote {args.out}: " + ", ".join(
        f"{k} {len(v['points'])} rates, rated {v['rated_load_rps']} req/s" for k, v in out["engines"].items()))


if __name__ == "__main__":
    main()
