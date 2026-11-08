"""6.5b step 5: scripts/push_space.py picks the Space's files from git (no network in these tests)."""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import push_space  # noqa: E402
from push_space import contents, dockerfile_for, plan_files, tree_problems, variable_changes, wanted_variables  # noqa: E402

REAL = Path(__file__).parent.parent


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A repo shaped like ours, with things that must never reach the Space committed on purpose."""
    r = tmp_path / "repo"
    files = {
        "Dockerfile": "FROM python\nARG GIT_COMMIT=unknown\nENV C=$GIT_COMMIT\n", ".dockerignore": "*\n",
        "LICENSE": "MIT", "src/tinyllm/server.py": "x = 1\n", "src/tinyllm/static/demo.html": "<p>",
        "src/tinyllm/__pycache__/server.cpython-312.pyc": "pyc", "tokenizer/tinystories-bpe8192.json": "{}",
        "tokenizer/other.json": "{}", "space/README.md": "---\nsdk: docker\n---\n",
        "Dockerfile.router": "FROM slim\nARG GIT_COMMIT=unknown\n", "space/router/README.md": "router card",
        "space/replica/README.md": "replica card",
        "README.md": "the GitHub readme", "NOTES.local.md": "private", "checkpoints/main/model.pt": "w",
        "data/x.bin": "d", "runs/bench/a.json": "{}", "wandb/run/log": "l", "hf/MODEL_CARD.md": "card",
        "scripts/push_space.py": "", "tests/test_x.py": "", ".env": "HF_TOKEN=secret",
    }
    for name, text in files.items():
        (r / name).parent.mkdir(parents=True, exist_ok=True)
        (r / name).write_text(text)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "add", "-f", ".")
    git(r, "commit", "-q", "-m", "first")
    return r


def test_only_the_allow_list_goes_up(repo):
    files = plan_files(repo, "server")
    assert files == {"Dockerfile": "Dockerfile", ".dockerignore": ".dockerignore", "LICENSE": "LICENSE",
                     "README.md": "space/README.md", "src/tinyllm/server.py": "src/tinyllm/server.py",
                     "src/tinyllm/static/demo.html": "src/tinyllm/static/demo.html",
                     "tokenizer/tinystories-bpe8192.json": "tokenizer/tinystories-bpe8192.json"}
    banned = ("checkpoints/", "data/", "runs/", "wandb/", "hf/", "scripts/", "tests/", "space/")
    assert not [f for f in files if f.startswith(banned) or f in ("NOTES.local.md", ".env") or f.endswith(".pyc")]


def test_contents_come_from_head_with_the_commit_in_the_dockerfile(repo):
    (repo / "src/tinyllm/server.py").write_text("uncommitted\n")          # the working tree is ignored
    data = contents(repo, plan_files(repo, "server"), "abc1234")
    assert data["src/tinyllm/server.py"] == b"x = 1\n" and data["README.md"].startswith(b"---\nsdk: docker")
    assert b"ARG GIT_COMMIT=abc1234\n" in data["Dockerfile"] and b"unknown" not in data["Dockerfile"]


def test_the_real_dockerfile_takes_the_commit():
    text = (REAL / "Dockerfile").read_text()
    out = dockerfile_for(text, "abc1234")
    assert out.count("ARG GIT_COMMIT=abc1234\n") == 1 and len(out) == len(text) - len("unknown") + len("abc1234")
    with pytest.raises(SystemExit):
        dockerfile_for(text.replace("ARG GIT_COMMIT=unknown\n", ""), "abc1234")


def test_a_missing_allow_listed_file_is_an_error(repo):
    git(repo, "rm", "-q", "LICENSE")
    git(repo, "commit", "-q", "-m", "no license")
    with pytest.raises(SystemExit, match="LICENSE"):
        plan_files(repo, "server")


def test_refuses_unpushed_or_uncommitted_work(repo, tmp_path):
    assert tree_problems(repo) == ["HEAD is not on any remote branch: push it first"]
    git(tmp_path, "init", "-q", "--bare", str(tmp_path / "origin.git"))
    git(repo, "remote", "add", "origin", str(tmp_path / "origin.git"))
    git(repo, "push", "-q", "origin", "main")
    assert tree_problems(repo) == []
    (repo / "untracked.txt").write_text("new")                             # untracked files don't count
    assert tree_problems(repo) == []
    (repo / "LICENSE").write_text("changed")
    assert tree_problems(repo) == ["tracked files have uncommitted changes: commit first"]


def test_dry_run_lists_files_and_refuses_unpushed(repo, monkeypatch, capsys):
    monkeypatch.setattr(push_space, "REPO", repo)
    with pytest.raises(SystemExit) as e:
        push_space.main(["--role", "server", "--space", "me/x"])
    out = capsys.readouterr().out
    assert e.value.code == 1 and "refused: HEAD is not on any remote branch" in out
    assert "README.md" in out and "(from space/README.md)" in out and "NOTES.local.md" not in out
    assert "7 files" in out and "ARG GIT_COMMIT=" in out


# -- D-031: roles ------------------------------------------------------------------------

def test_router_has_its_own_dockerfile_and_card_and_no_tokenizer(repo):
    files = plan_files(repo, "router")
    assert files["Dockerfile"] == "Dockerfile.router" and files["README.md"] == "space/router/README.md"
    assert not [f for f in files if f.startswith("tokenizer/")] and "Dockerfile.router" not in files
    data = contents(repo, files, "abc1234")
    assert data["README.md"] == b"router card" and b"ARG GIT_COMMIT=abc1234\n" in data["Dockerfile"]


def test_replica_runs_the_server_image_with_its_own_card(repo):
    files = plan_files(repo, "replica")
    assert files["Dockerfile"] == "Dockerfile" and files["README.md"] == "space/replica/README.md"
    assert "tokenizer/tinystories-bpe8192.json" in files


@pytest.mark.parametrize("role, dataset, expected", [
    ("server", "me/logs", {"LOG_DATASET": "me/logs"}),
    ("router", "me/logs", {"LOG_DATASET": "me/logs"}),
    ("replica", "me/logs", {"TRUST_ROUTER": "1", "RATE_LIMIT": "0"}),      # never ships logs
])
def test_each_role_sets_only_its_variables(role, dataset, expected):
    assert wanted_variables(role, dataset) == expected


@pytest.mark.parametrize("role, current, to_set, to_delete", [
    ("replica", {"LOG_DATASET": "me/logs"}, {"TRUST_ROUTER": "1", "RATE_LIMIT": "0"}, ["LOG_DATASET"]),
    ("replica", {"TRUST_ROUTER": "1", "RATE_LIMIT": "0"}, {}, []),
    ("server", {"TRUST_ROUTER": "1"}, {"LOG_DATASET": "me/logs"}, ["TRUST_ROUTER"]),   # never on a public Space
    ("router", {"GIT_COMMIT": "x", "LOG_DATASET": "me/logs"}, {}, ["GIT_COMMIT"]),
])
def test_variable_changes(role, current, to_set, to_delete):
    assert variable_changes(role, wanted_variables(role, "me/logs"), current) == (to_set, to_delete)


@pytest.mark.parametrize("argv", [
    ["--space", "me/x"], ["--role", "server"], ["--role", "nope", "--space", "me/x"],
    ["--role", "replica", "--space", "me/x", "--log-dataset", "me/logs"],
])
def test_refused_arguments(argv):
    with pytest.raises(SystemExit) as e:
        push_space.main(argv)
    assert e.value.code == 2


def test_the_real_router_image_is_pinned_to_the_lock_file_and_has_no_torch():
    text = (REAL / "Dockerfile.router").read_text()
    locked = {p["name"]: p["version"] for p in tomllib.loads((REAL / "uv.lock").read_text())["package"]}
    pins = re.findall(r"([a-z][a-z0-9-]*)==([0-9.]+)", text)
    assert len(pins) == 6 and [(n, v) for n, v in pins if locked.get(n) != v] == []
    assert "torch==" not in text and dockerfile_for(text, "abc1234").count("ARG GIT_COMMIT=abc1234\n") == 1


class FakeApi:
    def __init__(self, private=True, variables=None, files=()):
        self.private, self.variables, self.files = private, dict(variables or {}), files
        self.created, self.commits, self.changed = [], [], []

    def whoami(self):
        return {"name": "me"}

    def space_info(self, space):
        return SimpleNamespace(private=self.private)

    def get_space_variables(self, space):
        return {k: SimpleNamespace(value=v) for k, v in self.variables.items()}

    def add_space_variable(self, space, key, value):
        self.changed.append(("set", key, value))

    def delete_space_variable(self, space, key):
        self.changed.append(("delete", key))

    def list_repo_files(self, space, repo_type):
        return list(self.files)

    def create_repo(self, space, **kw):
        self.created.append((space, kw))
        return "url"

    def create_commit(self, space, ops, repo_type, commit_message):
        self.commits.append(ops)
        return SimpleNamespace(oid="oid", commit_url="url")


@pytest.fixture
def pushed(repo, tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "--bare", str(tmp_path / "origin.git"))
    git(repo, "remote", "add", "origin", str(tmp_path / "origin.git"))
    git(repo, "push", "-q", "origin", "main")
    monkeypatch.setattr(push_space, "REPO", repo)
    def use(api):
        monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
        return api
    return use


@pytest.mark.parametrize("role, private, variables", [
    ("server", False, [{"key": "LOG_DATASET", "value": push_space.DEFAULT_LOG_DATASET}]),
    ("router", True, [{"key": "LOG_DATASET", "value": push_space.DEFAULT_LOG_DATASET}]),
    ("replica", True, [{"key": "TRUST_ROUTER", "value": "1"}, {"key": "RATE_LIMIT", "value": "0"}]),
])
def test_create(pushed, role, private, variables):
    api = pushed(FakeApi())
    push_space.main(["--role", role, "--space", "me/x", "--create", "--yes"])
    (space, kw), = api.created
    assert space == "me/x" and kw["private"] is private and kw["space_variables"] == variables
    assert kw["space_hardware"] == "cpu-basic" and kw["exist_ok"] is False and api.commits == []


def test_without_yes_nothing_is_touched(pushed):
    api = pushed(FakeApi())
    push_space.main(["--role", "replica", "--space", "me/x", "--create"])
    push_space.main(["--role", "replica", "--space", "me/x"])
    assert (api.created, api.commits, api.changed) == ([], [], [])


def test_a_public_space_can_never_become_a_replica(pushed):
    api = pushed(FakeApi(private=False))
    with pytest.raises(SystemExit, match="must be private"):
        push_space.main(["--role", "replica", "--space", "me/x", "--yes"])
    assert (api.commits, api.changed) == ([], [])


def test_upload_as_replica(pushed, repo):
    api = pushed(FakeApi(variables={"LOG_DATASET": "me/logs"}))
    push_space.main(["--role", "replica", "--space", "me/x", "--yes"])
    assert api.changed == [("delete", "LOG_DATASET"), ("set", "TRUST_ROUTER", "1"), ("set", "RATE_LIMIT", "0")]
    ops = {o.path_in_repo: o for o in api.commits[0]}
    assert ops["README.md"].path_or_fileobj == b"replica card" and "tokenizer/tinystories-bpe8192.json" in ops
    log = json.loads(next((repo / "runs/space").glob("push_*.json")).read_text())
    assert log["role"] == "replica" and log["space"] == "me/x" and "log_dataset" not in log


def test_taking_over_a_public_space_as_router_removes_what_it_no_longer_needs(pushed):
    old = ["tokenizer/tinystories-bpe8192.json", "README.md", ".gitattributes"]
    api = pushed(FakeApi(private=False, variables={"LOG_DATASET": push_space.DEFAULT_LOG_DATASET}, files=old))
    push_space.main(["--role", "router", "--space", "me/x", "--yes"])
    deleted = [o.path_in_repo for o in api.commits[0] if type(o).__name__ == "CommitOperationDelete"]
    assert deleted == ["tokenizer/tinystories-bpe8192.json"] and api.changed == []
