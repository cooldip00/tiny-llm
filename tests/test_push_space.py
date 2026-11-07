"""6.5b step 5: scripts/push_space.py picks the Space's files from git (no network in these tests)."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import push_space  # noqa: E402
from push_space import contents, dockerfile_for, plan_files, tree_problems  # noqa: E402

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
    files = plan_files(repo)
    assert files == {"Dockerfile": "Dockerfile", ".dockerignore": ".dockerignore", "LICENSE": "LICENSE",
                     "README.md": "space/README.md", "src/tinyllm/server.py": "src/tinyllm/server.py",
                     "src/tinyllm/static/demo.html": "src/tinyllm/static/demo.html",
                     "tokenizer/tinystories-bpe8192.json": "tokenizer/tinystories-bpe8192.json"}
    banned = ("checkpoints/", "data/", "runs/", "wandb/", "hf/", "scripts/", "tests/", "space/")
    assert not [f for f in files if f.startswith(banned) or f in ("NOTES.local.md", ".env") or f.endswith(".pyc")]


def test_contents_come_from_head_with_the_commit_in_the_dockerfile(repo):
    (repo / "src/tinyllm/server.py").write_text("uncommitted\n")          # the working tree is ignored
    data = contents(repo, plan_files(repo), "abc1234")
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
        plan_files(repo)


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
        push_space.main([])
    out = capsys.readouterr().out
    assert e.value.code == 1 and "refused: HEAD is not on any remote branch" in out
    assert "README.md" in out and "(from space/README.md)" in out and "NOTES.local.md" not in out
    assert "7 files" in out and "ARG GIT_COMMIT=" in out
