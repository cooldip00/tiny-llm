"""6.5b: publish our 30M model (the 4.8b export) and its model card to the Hugging Face Hub.

Run:  uv run python scripts/upload_model.py                 # dry run: checks + what would be uploaded
      uv run python scripts/upload_model.py --yes           # creates the public repo and uploads

Before anything goes up it checks:
  - the folder holds exactly the export files (config, generation config, weights, tokenizer)
  - <folder>-verify.json from scripts/export_hf.py says PASS, and its file sizes match the folder
  - the card exists and names the repo it is going to
One commit uploads the five files plus the card as README.md. Login first, inside WSL:
`hf auth login` (the token stays in ~/.cache/huggingface, never in the repo). The commit id is
printed and saved to runs/export/upload_<time>.json; pin it as the Docker image's REVISION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tinyllm.server import HF_MODEL_FILES  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preflight(folder: Path, card: Path, repo_id: str) -> dict:
    files = sorted(p.name for p in folder.iterdir())
    if files != sorted(HF_MODEL_FILES):
        raise SystemExit(f"{folder}: expected exactly {sorted(HF_MODEL_FILES)}, found {files}")
    verify = folder.parent / f"{folder.name}-verify.json"
    if not verify.exists():
        raise SystemExit(f"{verify} missing: run scripts/export_hf.py first")
    report = json.loads(verify.read_text())
    if not report.get("passed"):
        raise SystemExit(f"{verify}: export check did not pass: {report.get('checks')}")
    sizes = {f: (folder / f).stat().st_size for f in HF_MODEL_FILES}
    if sizes != report["files_bytes"]:
        raise SystemExit(f"{folder}: files changed since the export check: {sizes} vs {report['files_bytes']}")
    if not card.exists() or repo_id not in card.read_text():
        raise SystemExit(f"{card}: missing, or does not mention {repo_id}")
    return {f: {"bytes": sizes[f], "sha256": sha256(folder / f)} for f in HF_MODEL_FILES}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="CoolDip00/tiny-llm-tinystories-30m")
    p.add_argument("--folder", type=Path, default=REPO / "runs/export/tinystories-30m")
    p.add_argument("--card", type=Path, default=REPO / "hf/MODEL_CARD.md")
    p.add_argument("--yes", action="store_true", help="really create the public repo and upload")
    args = p.parse_args()

    files = preflight(args.folder, args.card, args.repo)
    print(f"repo   https://huggingface.co/{args.repo} (public model repo)")
    for f, info in files.items():
        print(f"  {f:24s} {info['bytes']:>12,d} B  sha256 {info['sha256'][:16]}")
    print(f"  {'README.md':24s} {args.card.stat().st_size:>12,d} B  (from {args.card.relative_to(REPO)})")
    print("checks ok: exact file set, export check PASS, sizes match, card names the repo")
    if not args.yes:
        print("dry run: nothing uploaded (add --yes to upload)")
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    user = api.whoami()["name"]
    if args.repo.split("/")[0] != user:
        raise SystemExit(f"logged in as {user!r}, not the owner of {args.repo}")
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    ops = [CommitOperationAdd(f, str(args.folder / f)) for f in HF_MODEL_FILES]
    ops.append(CommitOperationAdd("README.md", str(args.card)))
    info = api.create_commit(args.repo, ops, commit_message="tiny-llm 30M TinyStories model (4.8b export, checked)")
    out = {"repo": args.repo, "commit": info.oid, "url": info.commit_url,
           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files}
    log = REPO / "runs/export" / f"upload_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    log.write_text(json.dumps(out, indent=2))
    print(f"uploaded: {info.commit_url}\ncommit {info.oid} (pin as REVISION) -> {log.relative_to(REPO)}")


if __name__ == "__main__":
    main()
