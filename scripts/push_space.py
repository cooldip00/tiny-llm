"""6.5b step 5, D-031: create a Hugging Face Space and push one role's image files to it.

Run:  uv run python scripts/push_space.py --role R --space S                 # dry run: checks + file list, no network
      uv run python scripts/push_space.py --role R --space S --create --yes  # create the Space + its variables, no code
      uv run python scripts/push_space.py --role R --space S --yes           # upload HEAD's files; the Space rebuilds

Roles (both flags are required, so a forgotten flag can't turn the router back into a single server):
  server   the model behind one public Space (Dockerfile, space/README.md)
  router   the router, no model (Dockerfile.router, space/router/README.md); created private, for testing
  replica  the model behind the router (Dockerfile, space/replica/README.md); must stay private

Only an allow-list goes up, read from git at HEAD (never the working tree): the role's Dockerfile and
.dockerignore, src/tinyllm/, the tokenizer (not for the router), LICENSE, and the role's README as README.md
(the Space's settings header and page). Never data, checkpoints, runs, wandb, hf/, notes or secrets.

An upload is refused when tracked files have uncommitted changes or HEAD is not on a remote branch,
so the commit the Space reports on /stats is one anyone can look up on GitHub. That commit goes into
the uploaded Dockerfile as the `ARG GIT_COMMIT` default, so the code and its label arrive in one Space
commit (a GIT_COMMIT Space variable would override it and restart the Space on its own: it is removed).
One commit uploads the files and deletes Space files no longer in the list. Each upload is recorded in
runs/space/push_<time>.json.

Secrets (see SECRETS below) are added by hand in the Space settings, never by this script. Login first, inside WSL: `hf auth login` (the token stays in ~/.cache/huggingface).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = (".dockerignore", "LICENSE", "src/tinyllm")
TOKENIZER = "tokenizer/tinystories-bpe8192.json"
DEFAULT_LOG_DATASET = "CoolDip00/tiny-llm-space-logs"


@dataclass(frozen=True)
class Role:
    dockerfile: str
    readme: str
    sources: tuple[str, ...]
    create_private: bool
    ships_logs: bool
    forbidden: tuple[str, ...]              # Space variables that must not exist in this role
    variables: dict[str, str] = field(default_factory=dict)
    must_be_private: bool = False


ROLES = {
    "server": Role("Dockerfile", "space/README.md", (*CODE, TOKENIZER), create_private=False, ships_logs=True,
                   forbidden=("TRUST_ROUTER",)),
    "router": Role("Dockerfile.router", "space/router/README.md", CODE, create_private=True, ships_logs=True,
                   forbidden=("TRUST_ROUTER",)),
    # Private because it believes the router's client header: on a public Space a visitor could send it too.
    "replica": Role("Dockerfile", "space/replica/README.md", (*CODE, TOKENIZER), create_private=True, ships_logs=False,
                    forbidden=("LOG_DATASET",), variables={"TRUST_ROUTER": "1", "RATE_LIMIT": "0"}, must_be_private=True),
}
SECRETS = {"server": "HF_TOKEN, TINYLLM_LOG_SALT", "replica": "none",
           "router": "TINYLLM_REPLICA_TOKEN, REPLICAS, HF_TOKEN, TINYLLM_LOG_SALT"}
KEEP_ON_SPACE = {".gitattributes"}             # created by HF with the repo; not ours to delete
ARG_LINE = "ARG GIT_COMMIT=unknown\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def plan_files(repo: Path, role: str) -> dict[str, str]:
    """Path on the Space -> path in the repo, for every allow-listed file git tracks at HEAD."""
    r = ROLES[role]
    renamed = {r.dockerfile: "Dockerfile", r.readme: "README.md"}
    wanted = [*r.sources, *renamed]
    tracked = [p for p in git(repo, "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", *wanted).split("\0")
               if p and "__pycache__" not in p and not p.endswith(".pyc")]
    for a in wanted:
        if not any(p == a or p.startswith(a + "/") for p in tracked):
            raise SystemExit(f"{a}: not tracked at HEAD")
    return {renamed.get(p, p): p for p in sorted(tracked)}


def wanted_variables(role: str, log_dataset: str | None) -> dict[str, str]:
    r = ROLES[role]
    return {**r.variables, **({"LOG_DATASET": log_dataset} if r.ships_logs and log_dataset else {})}


def variable_changes(role: str, wanted: dict[str, str], current: dict[str, str | None]) -> tuple[dict[str, str], list[str]]:
    """(to set, to delete). GIT_COMMIT would override the Dockerfile's commit and restart the Space on its own."""
    return ({k: v for k, v in wanted.items() if current.get(k) != v},
            [k for k in ("GIT_COMMIT", *ROLES[role].forbidden) if k in current])


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
    p.add_argument("--role", required=True, choices=sorted(ROLES))
    p.add_argument("--space", required=True, metavar="OWNER/NAME")
    p.add_argument("--log-dataset", default=None,
                   help=f"private dataset for the request logs (Space variable LOG_DATASET); default {DEFAULT_LOG_DATASET}; not for a replica")
    p.add_argument("--create", action="store_true", help="create the Space and its variables; uploads no code")
    p.add_argument("--yes", action="store_true", help="really do it (default: dry run, no network)")
    args = p.parse_args(argv)
    role = ROLES[args.role]
    if args.log_dataset and not role.ships_logs:
        p.error("a replica ships no logs: only the router does")
    wanted = wanted_variables(args.role, args.log_dataset or DEFAULT_LOG_DATASET)

    if args.create:
        print(f"space  https://huggingface.co/spaces/{args.space} ({'private' if role.create_private else 'public'}, Docker, cpu-basic, role {args.role})")
        print(f"  variables {wanted}")
        print(f"  then add the secrets in its settings ({SECRETS[args.role]}), then push the code")
        if not args.yes:
            print("dry run: nothing created (add --yes)")
            return
        from huggingface_hub import HfApi
        api = HfApi()
        owner(api, args.space)
        url = api.create_repo(args.space, repo_type="space", space_sdk="docker", space_hardware="cpu-basic",
                              private=role.create_private, exist_ok=False,
                              space_variables=[{"key": k, "value": v} for k, v in wanted.items()])
        print(f"created: {url}")
        return

    commit = git(REPO, "rev-parse", "--short", "HEAD").strip()
    subject = git(REPO, "log", "-1", "--format=%s").strip()
    files = plan_files(REPO, args.role)
    data = contents(REPO, files, commit)
    print(f"space  https://huggingface.co/spaces/{args.space}  <- HEAD {commit} {subject!r}  (role {args.role})")
    for dst, b in data.items():
        src = files[dst]
        note = f"  (from {src})" if src != dst else ""
        note += f"  (ARG GIT_COMMIT={commit})" if dst == "Dockerfile" else ""
        print(f"  {dst:52s} {len(b):>9,d} B{note}")
    print(f"  {len(data)} files, {sum(map(len, data.values())):,} B; variables {wanted}")
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
    if role.must_be_private and not api.space_info(args.space).private:
        raise SystemExit(f"{args.space} is public: a {args.role} must be private")
    current = {k: getattr(v, "value", None) for k, v in api.get_space_variables(args.space).items()}
    to_set, to_delete = variable_changes(args.role, wanted, current)
    for k in to_delete:
        api.delete_space_variable(args.space, k)
        print(f"removed the {k} variable (must not exist for role {args.role})")
    for k, v in to_set.items():
        api.add_space_variable(args.space, k, v)
        print(f"set {k}={v}")
    stale = sorted(set(api.list_repo_files(args.space, repo_type="space")) - set(data) - KEEP_ON_SPACE)
    ops = [CommitOperationAdd(dst, b) for dst, b in data.items()] + [CommitOperationDelete(f) for f in stale]
    info = api.create_commit(args.space, ops, repo_type="space", commit_message=f"tiny-llm {commit}: {subject}")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({
        "space": args.space, "git_commit": commit, "space_commit": info.oid, "url": info.commit_url,
        "time_utc": stamp, "deleted": stale, "role": args.role, "variables": wanted,
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
