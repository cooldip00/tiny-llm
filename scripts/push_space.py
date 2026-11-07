"""6.5b step 5: create the Hugging Face Space and push the server image's files to it.

Run:  uv run python scripts/push_space.py                    # dry run: checks + file list, no network
      uv run python scripts/push_space.py --create --yes     # create the public Space + LOG_DATASET, no code
      uv run python scripts/push_space.py --yes              # upload HEAD's files; the Space rebuilds

Only an allow-list goes up, read from git at HEAD (never the working tree): the Dockerfile and
.dockerignore, src/tinyllm/, the tokenizer, LICENSE, and space/README.md as README.md (the Space's
settings header and page). Never data, checkpoints, runs, wandb, hf/, notes or secrets.

An upload is refused when tracked files have uncommitted changes or HEAD is not on a remote branch,
so the commit the Space reports on /stats is one anyone can look up on GitHub. That commit goes into
the uploaded Dockerfile as the `ARG GIT_COMMIT` default, so the code and its label arrive in one Space
commit (a GIT_COMMIT Space variable would override it and restart the Space on its own: it is removed).
One commit uploads the files and deletes Space files no longer in the list. Each upload is recorded in
runs/space/push_<time>.json.

Secrets (HF_TOKEN for the log dataset, TINYLLM_LOG_SALT) are added by hand in the Space settings, never
by this script. Login first, inside WSL: `hf auth login` (the token stays in ~/.cache/huggingface).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOW = ["Dockerfile", ".dockerignore", "LICENSE", "src/tinyllm", "tokenizer/tinystories-bpe8192.json"]
README = "space/README.md"
KEEP_ON_SPACE = {".gitattributes"}             # created by HF with the repo; not ours to delete
ARG_LINE = "ARG GIT_COMMIT=unknown\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def plan_files(repo: Path = REPO) -> dict[str, str]:
    """Path on the Space -> path in the repo, for every allow-listed file git tracks at HEAD."""
    tracked = [p for p in git(repo, "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", *ALLOW, README).split("\0")
               if p and "__pycache__" not in p and not p.endswith(".pyc")]
    for a in [*ALLOW, README]:
        if not any(p == a or p.startswith(a + "/") for p in tracked):
            raise SystemExit(f"{a}: not tracked at HEAD")
    return {("README.md" if p == README else p): p for p in sorted(tracked)}


def tree_problems(repo: Path = REPO) -> list[str]:
    problems = []
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        problems.append("tracked files have uncommitted changes: commit first")
    if not git(repo, "branch", "-r", "--contains", "HEAD").strip():
        problems.append("HEAD is not on any remote branch: push it first")
    return problems


def blob(repo: Path, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{path}"], check=True, capture_output=True).stdout


def dockerfile_for(text: str, commit: str) -> str:
    if text.count(ARG_LINE) != 1:
        raise SystemExit(f"Dockerfile: expected one line {ARG_LINE.strip()!r}")
    return text.replace(ARG_LINE, f"ARG GIT_COMMIT={commit}\n")


def contents(repo: Path, files: dict[str, str], commit: str) -> dict[str, bytes]:
    out = {dst: blob(repo, src) for dst, src in files.items()}
    out["Dockerfile"] = dockerfile_for(out["Dockerfile"].decode(), commit).encode()
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--space", default="CoolDip00/tiny-llm")
    p.add_argument("--log-dataset", default="CoolDip00/tiny-llm-space-logs",
                   help="private dataset for the request logs (Space variable LOG_DATASET)")
    p.add_argument("--create", action="store_true", help="create the public Space and its variables; uploads no code")
    p.add_argument("--yes", action="store_true", help="really do it (default: dry run, no network)")
    args = p.parse_args(argv)

    if args.create:
        print(f"space  https://huggingface.co/spaces/{args.space} (public, Docker, cpu-basic)")
        print(f"  variable LOG_DATASET={args.log_dataset}")
        print("  then add the secrets HF_TOKEN and TINYLLM_LOG_SALT in its settings, then push the code")
        if not args.yes:
            print("dry run: nothing created (add --yes)")
            return
        from huggingface_hub import HfApi
        api = HfApi()
        owner(api, args.space)
        url = api.create_repo(args.space, repo_type="space", space_sdk="docker", space_hardware="cpu-basic",
                              private=False, exist_ok=False,
                              space_variables=[{"key": "LOG_DATASET", "value": args.log_dataset}])
        print(f"created: {url}")
        return

    commit = git(REPO, "rev-parse", "--short", "HEAD").strip()
    subject = git(REPO, "log", "-1", "--format=%s").strip()
    files = plan_files(REPO)
    data = contents(REPO, files, commit)
    print(f"space  https://huggingface.co/spaces/{args.space}  <- HEAD {commit} {subject!r}")
    for dst, b in data.items():
        src = files[dst]
        note = f"  (from {src})" if src != dst else ""
        note += f"  (ARG GIT_COMMIT={commit})" if dst == "Dockerfile" else ""
        print(f"  {dst:52s} {len(b):>9,d} B{note}")
    print(f"  {len(data)} files, {sum(map(len, data.values())):,} B; variable LOG_DATASET={args.log_dataset}")
    problems = tree_problems(REPO)
    for msg in problems:
        print(f"refused: {msg}")
    if problems:
        raise SystemExit(1)
    print("checks ok: tracked files clean, HEAD on a remote branch")
    if not args.yes:
        print("dry run: nothing uploaded (add --yes)")
        return

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = REPO / "runs/space" / f"push_{stamp}.json"
    if log.exists():
        raise SystemExit(f"{log} exists: never overwritten")
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi()
    owner(api, args.space)
    variables = api.get_space_variables(args.space)
    if "GIT_COMMIT" in variables:
        api.delete_space_variable(args.space, "GIT_COMMIT")
        print("removed the GIT_COMMIT variable (it would override the Dockerfile's commit)")
    if getattr(variables.get("LOG_DATASET"), "value", None) != args.log_dataset:
        api.add_space_variable(args.space, "LOG_DATASET", args.log_dataset)
        print(f"set LOG_DATASET={args.log_dataset}")
    stale = sorted(set(api.list_repo_files(args.space, repo_type="space")) - set(data) - KEEP_ON_SPACE)
    ops = [CommitOperationAdd(dst, b) for dst, b in data.items()] + [CommitOperationDelete(f) for f in stale]
    info = api.create_commit(args.space, ops, repo_type="space", commit_message=f"tiny-llm {commit}: {subject}")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({
        "space": args.space, "git_commit": commit, "space_commit": info.oid, "url": info.commit_url,
        "time_utc": stamp, "deleted": stale, "log_dataset": args.log_dataset,
        "files": {dst: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for dst, b in data.items()},
    }, indent=2))
    print(f"uploaded: {info.commit_url}" + (f"\ndeleted: {stale}" if stale else "") + f"\n-> {log.relative_to(REPO)}")
    print(f"the Space rebuilds now: https://huggingface.co/spaces/{args.space}?logs=build")


def owner(api, space: str) -> None:
    user = api.whoami()["name"]
    if space.split("/")[0] != user:
        raise SystemExit(f"logged in as {user!r}, not the owner of {space}")


if __name__ == "__main__":
    main()
