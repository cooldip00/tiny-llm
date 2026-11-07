"""Log shipping to a private HF dataset (6.5b step 4, logship.py). CPU only, no network: the HF side
is a fake `HfApi` / scheduler. The privacy audit sends a marker string through every response path
and checks it never reaches the shipped files."""

import calendar
import datetime as dt
import json
import logging
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import httpx
import pytest

from test_server import greedy, model, serve  # noqa: F401  (model is a fixture)
from tinyllm.logship import LogShipper, ShipHandler, _scheduler_class, apply_retention, log_path, start_id
from tinyllm.server import Faults

T0 = calendar.timegm((2026, 11, 7, 13, 15, 0))          # 2026-11-07 13:15:00 UTC
DAY = 86400


def record(msg, t=T0, level=logging.INFO, exc=False, **fields):
    exc_info = None
    if exc:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys
            exc_info = sys.exc_info()
    r = logging.LogRecord("tinyllm.server", level, __file__, 1, msg, None, exc_info)
    r.created, r.fields = t, fields
    return r


def lines(root):
    return [json.loads(x) for p in sorted(root.rglob("*.jsonl")) for x in p.read_text().splitlines()]


def test_hourly_files_per_start_never_shared(tmp_path):
    h = ShipHandler(tmp_path, start_id(T0), "local")
    assert start_id(T0) == "20261107T131500Z"
    for t in (T0, T0 + 600, T0 + 3600, T0 + 11 * 3600):                    # 13:15, 13:25, 14:15, 00:15 next day
        h.emit(record("request", t, status=200))
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.jsonl"))
    assert files == ["2026-11-07/20261107T131500Z-local-13.jsonl", "2026-11-07/20261107T131500Z-local-14.jsonl",
                     "2026-11-08/20261107T131500Z-local-00.jsonl"]
    assert len((tmp_path / files[0]).read_text().splitlines()) == 2
    # a restart in the same hour writes its own file
    assert log_path(tmp_path, start_id(T0 + 60), "local", T0 + 120) != tmp_path / files[0]


def test_only_allowed_lines_are_shipped(tmp_path):
    h = ShipHandler(tmp_path, "s", "r")
    for r in [record("model loaded", version="v1"), record("server ready"), record("request", status=200),
              record("slow request", level=logging.WARNING, timeline={}), record("shutting down: draining"),
              record("stopped"), record("engine step failed: not ready", level=logging.ERROR, exc=True),
              record("step", level=logging.DEBUG, step=1), record("log upload off: logs stay local",
                                                                      level=logging.WARNING),
              record("something else")]:
        h.emit(r)
    got = lines(tmp_path)
    assert [x["msg"] for x in got] == ["model loaded", "server ready", "request", "slow request",
                                       "shutting down: draining", "stopped", "engine step failed: not ready"]
    assert "exc" not in got[-1] and "boom" not in json.dumps(got)                # no tracebacks


def test_snapshots_are_thinned_per_variant(tmp_path):
    h = ShipHandler(tmp_path, "s", "r", snapshot_every_s=30)
    for i in range(14):                                                    # every 5 s for 65 s, two variants
        for v in ("a", "b"):
            h.emit(record("engine", T0 + 5 * i + 0.01 * (i % 3), variant=v, queue_depth=i))
    got = [(x["variant"], x["queue_depth"]) for x in lines(tmp_path)]
    assert got == [("a", 0), ("b", 0), ("a", 6), ("b", 6), ("a", 12), ("b", 12)]


class FakeScheduler:
    def __init__(self, root, **kw):
        self.root, self.kw, self.lock = root, kw, threading.Lock()
        self.uploads, self.stopped = [], False

    def trigger(self):
        self.uploads.append([x["msg"] for x in lines(self.root)])          # what an upload would see now
        f = Future()
        f.set_result(None)
        return f

    def stop(self):
        self.stopped = True


def test_no_dataset_no_scheduler(tmp_path):
    made = []
    s = LogShipper(tmp_path, None, scheduler_factory=lambda **kw: made.append(kw))
    s.start_upload()
    s.close()
    assert made == [] and s.scheduler is None


def test_unreachable_dataset_keeps_logs_local(tmp_path, caplog):
    def fail(**kw):
        raise OSError("no token")
    s = LogShipper(tmp_path, "me/logs", replica="r1", scheduler_factory=fail)
    s.start_upload()
    s._thread.join(5)
    s.handler.emit(record("request", status=200))
    s.close()
    assert [x["msg"] for x in lines(tmp_path)] == ["request"]
    assert any(r.getMessage() == "log upload off: logs stay local" for r in caplog.records)


class FakeApi:
    """The HfApi calls CommitScheduler and apply_retention make, recorded."""

    def __init__(self, files=(), commit_days=()):
        self.files, self.commits, self.squashed = set(files), [], 0
        self.commit_days = list(commit_days)                               # newest first, as the Hub lists them

    def create_repo(self, repo_id, **kw):
        self.repo_kw = kw
        return SimpleNamespace(repo_id=repo_id)

    def run_as_future(self, fn):
        f = Future()
        try:
            f.set_result(fn())
        except Exception as e:
            f.set_exception(e)
        return f

    def list_repo_files(self, repo_id, repo_type=None):
        return sorted(self.files)

    def list_repo_commits(self, repo_id, repo_type=None):
        return [SimpleNamespace(created_at=dt.datetime.fromtimestamp(d, dt.timezone.utc)) for d in self.commit_days]

    def create_commit(self, repo_id, operations, commit_message, repo_type=None, revision=None):
        done = {}
        for op in operations:
            if hasattr(op, "path_or_fileobj"):
                done[op.path_in_repo] = op.path_or_fileobj.read().decode()
                self.files.add(op.path_in_repo)
            else:
                done[op.path_in_repo] = None
                self.files.discard(op.path_in_repo)
        self.commits.append((commit_message, done))

    def super_squash_history(self, repo_id, repo_type=None, commit_message=None, **kw):
        self.squashed += 1


def test_retention_deletes_old_files_and_squashes(tmp_path):
    files = ["README.md", "logs/2026-10-01/a-r-00.jsonl", "logs/2026-10-07/a-r-23.jsonl",
             "logs/2026-10-08/b-r-00.jsonl", "logs/2026-11-07/c-r-13.jsonl", "logs/notes/x.jsonl"]
    (tmp_path / "2026-09-30").mkdir()
    (tmp_path / "2026-11-07").mkdir()
    api = FakeApi(files, commit_days=[T0, T0 - 31 * DAY])
    done = apply_retention(api, "me/logs", tmp_path, T0)
    assert done == {"deleted": 2, "squashed": True, "local_dirs_deleted": 1}
    assert api.commits[0][1] == {"logs/2026-10-01/a-r-00.jsonl": None, "logs/2026-10-07/a-r-23.jsonl": None}
    assert sorted(api.files) == ["README.md", "logs/2026-10-08/b-r-00.jsonl", "logs/2026-11-07/c-r-13.jsonl",
                                 "logs/notes/x.jsonl"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-11-07"]
    # nothing old, recent history (or a single squashed commit): no commit, no squash
    for days in ([T0, T0 - 5 * DAY], [T0 - 40 * DAY]):
        api = FakeApi(["logs/2026-11-01/a.jsonl"], commit_days=days)
        assert apply_retention(api, "me/logs", None, T0) == {"deleted": 0, "squashed": False,
                                                              "local_dirs_deleted": 0}
        assert api.commits == [] and api.squashed == 0


def test_real_scheduler_uploads_under_logs_and_runs_retention_daily(tmp_path):
    api = FakeApi(["logs/2026-09-01/old.jsonl"], commit_days=[T0])
    now = [T0]
    factory = lambda **kw: _scheduler_class()(hf_api=api, clock=lambda: now[0], **kw)  # noqa: E731
    s = LogShipper(tmp_path, "me/logs", replica="r1", start=T0, scheduler_factory=factory)
    s.handler.emit(record("server ready", T0))
    s.start_upload()
    s._thread.join(5)
    assert s.scheduler is not None and api.repo_kw["private"] is True and api.repo_kw["repo_type"] == "dataset"
    end = threading.Event()
    for _ in range(500):                                                  # the scheduler's first push is async
        if len(api.commits) >= 2:
            break
        end.wait(0.01)
    assert api.commits[0][1] == {"logs/2026-09-01/old.jsonl": None}      # retention first
    assert list(api.commits[1][1]) == ["logs/2026-11-07/20261107T131500Z-r1-13.jsonl"]
    s.handler.emit(record("request", T0 + 60, status=200))
    now[0] = T0 + 3600                                                    # same day: no second retention
    s.close()
    assert len(api.commits) == 3 and json.loads(api.commits[2][1][
        "logs/2026-11-07/20261107T131500Z-r1-13.jsonl"].splitlines()[-1])["msg"] == "request"
    assert s.scheduler.last_retention == T0


@pytest.fixture
def info_logs():
    lg = logging.getLogger("tinyllm")
    old = lg.level
    lg.setLevel(logging.INFO)
    yield
    lg.setLevel(old)


MARKER = "ZQXJ"


def test_privacy_audit_marker_never_reaches_the_shipped_logs(model, tmp_path, info_logs):  # noqa: F811
    fake = {}
    s = LogShipper(tmp_path, "me/logs", replica="t", scheduler_factory=lambda **kw: fake.setdefault(
        "s", FakeScheduler(tmp_path, **kw))).attach()
    s.start_upload()
    texts = []
    with serve(model, n_slots=1, max_queue=1, timeout_s=1.0, faults=Faults(step_delay_s=0.03),
               shipper=s) as live:
        r = live.post(**greedy(prompt=MARKER + " once", max_tokens=10))                  # 200
        assert r.status_code == 200
        texts.append(r.json()["choices"][0]["text"])
        with live.client.stream("POST", "/v1/completions", json=greedy(prompt=MARKER, max_tokens=8,
                                                                        stream=True)) as rs:  # 200 stream
            assert rs.status_code == 200 and rs.read()
        assert live.post(prompt=MARKER, temperature=-1).status_code == 400              # validation
        assert live.post(prompt=MARKER, max_tokens=MARKER).status_code == 400
        assert live.post(prompt=MARKER * 20, max_tokens=5).status_code == 400            # too long
        res = []
        ts = [threading.Thread(target=lambda: res.append(live.post(**greedy(prompt=MARKER, max_tokens=8))))
              for _ in range(2)]
        for t in ts:
            t.start()
        assert live.wait(lambda: live.worker.outstanding() == 2)
        assert live.post(**greedy(prompt=MARKER)).status_code == 429                      # busy
        for t in ts:
            t.join()
        assert live.post(**greedy(prompt=MARKER, max_tokens=40)).status_code == 504       # timeout
        assert live.wait(lambda: live.worker.outstanding() == 0)
        with httpx.Client(base_url=live.url, timeout=10) as c:                           # disconnect
            with c.stream("POST", "/v1/completions", json=greedy(prompt=MARKER, max_tokens=40,
                                                                  stream=True)) as rs:
                assert next(rs.iter_lines()).startswith("data: ")
        assert live.wait(lambda: 'status="499"} 1' in live.client.get("/metrics").text)
    logging.getLogger("tinyllm").removeHandler(s.handler)                 # in case close() did not run
    raw ="".join(p.read_text() for p in tmp_path.rglob("*.jsonl"))
    got = [json.loads(x) for x in raw.splitlines()]
    statuses = {x["status"] for x in got if x["msg"] in ("request", "slow request")}
    assert {200, 400, 429, 504, 499} <= statuses
    assert MARKER not in raw and "127.0.0.1" not in raw
    assert all(t not in raw for t in texts if len(t) >= 4)
    assert all(len(x["client"]) == 16 for x in got if x["msg"] == "request")
    # lifecycle, and the last upload runs after "stopped" is written
    msgs = [x["msg"] for x in got]
    assert "server ready" in msgs and msgs[-1] == "stopped" and "engine" in msgs
    sched = fake["s"]
    assert sched.uploads and sched.uploads[-1][-1] == "stopped" and sched.stopped
    assert sched.kw["repo_type"] == "dataset" and sched.kw["private"] is True and sched.kw["path_in_repo"] == "logs"
