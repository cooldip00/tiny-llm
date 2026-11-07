"""Ship the server's logs to a private Hugging Face dataset (6.5b step 4, docs/SERVING.md "Privacy").

The free Space wipes its disk on restart, so the request logs go to a private dataset:

- `ShipHandler` writes a filtered copy of the `tinyllm` log as JSON lines, one file per server start
  per UTC hour: `<dir>/<YYYY-MM-DD>/<start>-<replica>-<HH>.jsonl`. Files are never shared between
  restarts, so an upload only ever touches the files of this run (in practice the current hour's).
- Shipped: `request` and `slow request` lines, engine snapshots thinned to one per 30 s (the status
  page keeps the 5 s ones in memory), the lifecycle lines (`model loaded`, `server ready`,
  `shutting down: draining`, `stopped`) and errors without their traceback. Never debug lines.
  No prompt or completion text and no raw IP reach the log at all (tests/test_logs.py checks it).
- `LogShipper` uploads the folder with `huggingface_hub.CommitScheduler` every 10 minutes from a
  background thread, plus once more during the graceful drain. Upload failures are logged to stderr
  and never touch serving. A hard kill loses at most the last 10 minutes.
- Retention: once a day (first at start) the uploader deletes dataset files older than 30 days, and
  squashes the dataset history when its oldest commit is older than 30 days, so the deleted files
  stop using storage.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shutil
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from tinyllm.obs import JsonFormatter

log = logging.getLogger("tinyllm.logship")

SHIPPED = frozenset({"model loaded", "server ready", "shutting down: draining", "stopped",
                     "request", "slow request"})
SNAPSHOT = "engine"
KEEP_DAYS = 30
UPLOAD_EVERY_MIN = 10.0
SNAPSHOT_EVERY_S = 30.0
PATH_IN_REPO = "logs"
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def start_id(t: float) -> str:
    """The server start in a file name, UTC: 20261107T131500Z."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t))


def default_replica() -> str:
    """The Space name on HF (`SPACE_ID` is owner/name), else "local"; safe for a file name."""
    name = os.environ.get("SPACE_ID", "local").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name) or "local"


def log_path(root: Path, start: str, replica: str, t: float) -> Path:
    g = time.gmtime(t)
    return root / time.strftime("%Y-%m-%d", g) / f"{start}-{replica}-{g.tm_hour:02d}.jsonl"


class ShipHandler(logging.Handler):
    """Appends the shipped lines to the hourly file. `lock` guards each write against the uploader
    reading the file at the same moment (CommitScheduler caps the upload at the size it saw)."""

    def __init__(self, root: Path, start: str, replica: str, snapshot_every_s: float = SNAPSHOT_EVERY_S,
                 lock: Callable[[], object] | None = None):
        super().__init__(level=logging.INFO)
        self.root, self.start, self.replica = Path(root), start, replica
        self.snapshot_every_s = snapshot_every_s
        self._io_lock = lock or nullcontext
        self._last_snapshot: dict[str, float] = {}
        self.setFormatter(JsonFormatter(with_exc=False))

    def keep(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg == SNAPSHOT and record.levelno < logging.WARNING:
            variant = str((getattr(record, "fields", None) or {}).get("variant"))
            last = self._last_snapshot.get(variant)
            if last is not None and record.created - last < 0.95 * self.snapshot_every_s:
                return False
            self._last_snapshot[variant] = record.created
            return True
        return msg in SHIPPED or record.levelno >= logging.ERROR

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self.keep(record):
                return
            line = self.format(record) + "\n"
            path = log_path(self.root, self.start, self.replica, record.created)
            with self._io_lock():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            self.handleError(record)


def _day(name: str) -> dt.date | None:
    m = _DAY.fullmatch(name)
    try:
        return dt.date.fromisoformat(m.group()) if m else None
    except ValueError:
        return None


def apply_retention(api, repo_id: str, root: Path | None, now: float, keep_days: int = KEEP_DAYS) -> dict:
    """Delete log files older than `keep_days` (by their UTC date folder) from the dataset, in one
    commit, and from `root`; squash the dataset history once its oldest commit is older than
    `keep_days`. `api` is an `HfApi` (a fake in the tests). Returns what was done."""
    from huggingface_hub import CommitOperationDelete

    cutoff = dt.datetime.fromtimestamp(now, dt.timezone.utc).date() - dt.timedelta(days=keep_days)
    old = []
    for f in api.list_repo_files(repo_id, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == PATH_IN_REPO and (d := _day(parts[1])) is not None and d < cutoff:
            old.append(f)
    if old:
        api.create_commit(repo_id, repo_type="dataset", operations=[CommitOperationDelete(f) for f in old],
                          commit_message=f"Delete logs older than {keep_days} days")
    commits = api.list_repo_commits(repo_id, repo_type="dataset")
    squash = len(commits) > 1 and commits[-1].created_at.timestamp() < now - keep_days * 86400
    if squash:
        api.super_squash_history(repo_id, repo_type="dataset",
                                 commit_message=f"Squash history (logs kept {keep_days} days)")
    local = 0
    if root is not None and root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and (day := _day(d.name)) is not None and day < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                local += 1
    return {"deleted": len(old), "squashed": squash, "local_dirs_deleted": local}


def _scheduler_class():
    """CommitScheduler that runs the retention once a day before an upload. Uploads, deletes and the
    squash all run on the scheduler's one worker thread, so they never overlap."""
    from huggingface_hub import CommitScheduler

    class RetentionScheduler(CommitScheduler):
        def __init__(self, *a, keep_days: int = KEEP_DAYS, clock: Callable[[], float] = time.time, **kw):
            self.keep_days, self.clock, self.last_retention = keep_days, clock, None
            super().__init__(*a, **kw)

        def push_to_hub(self):
            now = self.clock()
            if self.last_retention is None or now - self.last_retention >= 86400:
                self.last_retention = now
                try:
                    done = apply_retention(self.api, self.repo_id, self.folder_path, now, self.keep_days)
                    log.info("log retention", extra={"fields": {"dataset": self.repo_id, **done}})
                except Exception as e:
                    log.warning("log retention failed", extra={"fields": {"error": f"{type(e).__name__}: {e}"}})
            return super().push_to_hub()

    return RetentionScheduler


class LogShipper:
    """Local hourly JSONL under `log_dir`; with `dataset`, uploads it to that private HF dataset.
    The token is the usual one: `HF_TOKEN` (a Space secret) or the local `hf auth login`."""

    def __init__(self, log_dir: str | Path, dataset: str | None = None, replica: str | None = None,
                 every_min: float = UPLOAD_EVERY_MIN, keep_days: int = KEEP_DAYS,
                 snapshot_every_s: float = SNAPSHOT_EVERY_S, start: float | None = None,
                 scheduler_factory: Callable | None = None):
        self.root = Path(log_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset, self.every_min, self.keep_days = dataset, every_min, keep_days
        self.start = start_id(time.time() if start is None else start)
        self.replica = replica or default_replica()
        self.scheduler = None
        self._factory = scheduler_factory
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.handler = ShipHandler(self.root, self.start, self.replica, snapshot_every_s, self._io_lock)

    def _io_lock(self):
        s = self.scheduler
        return s.lock if s is not None else self._lock

    def attach(self, logger: logging.Logger | None = None) -> "LogShipper":
        (logger or logging.getLogger("tinyllm")).addHandler(self.handler)
        return self

    def start_upload(self) -> None:
        """Open the dataset in the background (the first upload and the retention run right away)."""
        if self.dataset and self._thread is None:
            self._thread = threading.Thread(target=self._connect, name="log-upload", daemon=True)
            self._thread.start()

    def _connect(self) -> None:
        factory = self._factory or _scheduler_class()
        try:
            self.scheduler = factory(repo_id=self.dataset, repo_type="dataset", private=True,
                                     folder_path=self.root, path_in_repo=PATH_IN_REPO,
                                     every=self.every_min, keep_days=self.keep_days)
            log.info("log upload on", extra={"fields": {"dataset": self.dataset, "every_min": self.every_min}})
        except Exception as e:      # no token, no access, no network: keep the logs local
            log.warning("log upload off: logs stay local", extra={"fields": {
                "dataset": self.dataset, "dir": str(self.root), "error": f"{type(e).__name__}: {e}"}})

    def close(self, timeout_s: float = 30.0) -> None:
        """Last upload (during the graceful drain), then stop the scheduler. Never raises."""
        if self._thread is not None:
            self._thread.join(timeout_s)
        s = self.scheduler
        if s is None:
            return
        try:
            s.trigger().result(timeout=timeout_s)
            log.info("logs uploaded", extra={"fields": {"dataset": self.dataset}})
        except Exception as e:
            log.warning("last log upload failed", extra={"fields": {"error": f"{type(e).__name__}: {e}"}})
        finally:
            s.stop()
        logging.getLogger("tinyllm").removeHandler(self.handler)
