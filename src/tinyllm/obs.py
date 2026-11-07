"""Step 4.13 step 2: observability for the server (D-028, docs/SERVING.md).

Standard library only, so the server and the Space need nothing extra.

- RequestRecord: one JSON log line per finished request (timings, token counts, status).
  Never the prompt or completion text; the client is a salted hash (hash_client).
- EngineSnapshot: the engine's load, sampled every few seconds.
- Metrics: thread-safe aggregator. Rolling windows (5 min, 1 h) back /stats (JSON);
  cumulative counters and histograms back /metrics (Prometheus text format, so a real
  Prometheus could scrape it unchanged). history() backs /stats/history for the status page
  (6.5b): the last hour per minute, engine snapshots, and the SLOs with their error budgets.
- JsonFormatter / setup_logging: every log line is one JSON object.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

WINDOWS = {"5m": 300.0, "1h": 3600.0}
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)

# SLOs (docs/SERVING.md, 4.13 step 3). Targets are shares of good requests; the TTFT threshold
# is per server (--slo-ttft-s: 1 s local GPU, 2 s on the Space).
SLO_TARGETS = {"ttft": 0.95, "availability": 0.99, "shed": 0.99}
SLO_WINDOW_S = 7 * 24 * 3600.0     # rolling 7 days, but only since the last restart (in memory)
FAST_BURN = 14.4                   # Google SRE fast-burn threshold, judged on the last hour
BURN_MIN_REQUESTS = 10             # fewer counted requests in the hour: no flag (too few to judge)
HISTORY_MINUTES = 60


def slo_counted(status: int) -> bool:
    """Requests the SLOs judge: served (200), shed (429) and server failures (>= 500, incl. 504).
    Client errors (400) and client disconnects (499) count for none of them."""
    return status == 200 or status == 429 or status >= 500


class _SloBucket:
    """Counters for one hour of counted requests."""
    __slots__ = ("counted", "ok", "slow", "failed", "shed")

    def __init__(self):
        self.counted = self.ok = self.slow = self.failed = self.shed = 0

    def add(self, rec: "RequestRecord", ttft_s: float) -> None:
        self.counted += 1
        if rec.status == 200:
            self.ok += 1
            self.slow += rec.ttft_s is None or rec.ttft_s > ttft_s
        elif rec.status == 429:
            self.shed += 1
        else:
            self.failed += 1

    def bad_and_total(self, slo: str) -> tuple[int, int]:
        if slo == "ttft":
            return self.slow, self.ok
        return (self.failed, self.counted) if slo == "availability" else (self.shed, self.counted)


def hash_client(ip: str, salt: str) -> str:
    """Stable, salted id for a client. The raw IP is never stored."""
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:16]


def percentile(xs, q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default); None for no data. q in [0, 100]."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def itl_stats(token_times: list[float]) -> tuple[float | None, float | None]:
    """Inter-token latency p50 / p95 from the times each token was produced."""
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    return percentile(gaps, 50), percentile(gaps, 95)


@dataclass
class RequestRecord:
    """One finished request (any outcome). Times in seconds; None when not reached."""
    request_id: str
    status: int                              # HTTP status: 200, 400, 429, 500, 504, 499 (client gone)
    variant: str = "main"
    model_version: str = ""
    client: str = ""                         # hash_client(), never the IP
    arrival_unix: float = 0.0
    queue_wait_s: float | None = None        # arrival → admitted into a slot (first token)
    ttft_s: float | None = None
    itl_p50_s: float | None = None
    itl_p95_s: float | None = None
    latency_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None         # "stop" (EOS), "length", "abort", "timeout", "error"
    stream: bool = False
    error: str | None = None
    queue_depth_at_arrival: int = 0
    busy_slots_at_arrival: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EngineSnapshot:
    t_unix: float
    queue_len: int
    busy_slots: int
    n_slots: int
    tokens_per_s: float                      # generated tokens / s since the previous snapshot
    kv_used_frac: float
    ram_mb: float | None = None
    vram_mb: float | None = None


class _Histogram:
    def __init__(self, buckets):
        self.buckets = tuple(buckets)
        self.counts = [0] * len(self.buckets)    # per bucket, not cumulative
        self.sum = 0.0
        self.count = 0

    def observe(self, x: float) -> None:
        for i, b in enumerate(self.buckets):
            if x <= b:
                self.counts[i] += 1
                break
        self.sum += x
        self.count += 1

    def lines(self, name: str, labels: str) -> list[str]:
        sep = "," if labels else ""
        out, cum = [], 0
        for b, c in zip(self.buckets, self.counts):
            cum += c
            out.append(f'{name}_bucket{{{labels}{sep}le="{b:g}"}} {cum}')
        out.append(f'{name}_bucket{{{labels}{sep}le="+Inf"}} {self.count}')
        lab = f"{{{labels}}}" if labels else ""
        out.append(f"{name}_sum{lab} {self.sum:.6f}")
        out.append(f"{name}_count{lab} {self.count}")
        return out


@dataclass
class _Window:
    span_s: float
    items: deque = field(default_factory=deque)   # (t_monotonic, RequestRecord)

    def prune(self, now: float) -> None:
        while self.items and self.items[0][0] < now - self.span_s:
            self.items.popleft()


class Metrics:
    """Thread-safe: handlers call observe(), the engine thread calls snapshot()."""

    def __init__(self, windows: dict[str, float] = WINDOWS, clock=time.monotonic, slo_ttft_s: float = 1.0,
                 wall=time.time):
        self._lock = threading.Lock()
        self._clock, self._wall = clock, wall
        self.slo_ttft_s = slo_ttft_s
        self._windows = {k: _Window(v) for k, v in windows.items()}
        self._hour = _Window(HISTORY_MINUTES * 60.0)        # per-minute history and the 1 h burn rate
        self._slo: deque = deque()                          # (hour index since start, _SloBucket)
        self._snaps: deque = deque()                        # (t_monotonic, EngineSnapshot), last hour
        self._requests: dict[tuple[str, int], int] = {}      # (variant, status) → count
        self._tokens: dict[str, int] = {}                     # variant → completion tokens
        self._ttft: dict[str, _Histogram] = {}
        self._latency: dict[str, _Histogram] = {}
        self.last_snapshot: EngineSnapshot | None = None
        self.started = clock()

    def observe(self, rec: RequestRecord, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            for w in (*self._windows.values(), self._hour):
                w.items.append((now, rec))
                w.prune(now)
            if slo_counted(rec.status):
                hour = int((now - self.started) // 3600)
                if not self._slo or self._slo[-1][0] != hour:
                    self._slo.append((hour, _SloBucket()))
                    while self._slo and self._slo[0][0] <= hour - SLO_WINDOW_S / 3600:
                        self._slo.popleft()
                self._slo[-1][1].add(rec, self.slo_ttft_s)
            v = rec.variant
            self._requests[(v, rec.status)] = self._requests.get((v, rec.status), 0) + 1
            self._tokens[v] = self._tokens.get(v, 0) + rec.completion_tokens
            if rec.ttft_s is not None:
                self._ttft.setdefault(v, _Histogram(TTFT_BUCKETS)).observe(rec.ttft_s)
            if rec.status == 200 and rec.latency_s is not None:
                self._latency.setdefault(v, _Histogram(LATENCY_BUCKETS)).observe(rec.latency_s)

    def snapshot(self, snap: EngineSnapshot, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            self.last_snapshot = snap
            self._snaps.append((now, snap))
            while self._snaps and self._snaps[0][0] < now - HISTORY_MINUTES * 60:
                self._snaps.popleft()

    def history(self, now: float | None = None) -> dict:
        """/stats/history, for the status page: the last hour per minute, the engine snapshots of the
        last hour, and each SLO over the rolling 7-day window (since the last restart)."""
        now = self._clock() if now is None else now
        wall = self._wall()
        with self._lock:
            self._hour.prune(now)
            recs = list(self._hour.items)
            snaps = [(t, s) for t, s in self._snaps if t >= now - HISTORY_MINUTES * 60]
            hour_now = int((now - self.started) // 3600)
            buckets = [b for h, b in self._slo if h > hour_now - SLO_WINDOW_S / 3600]
        uptime = now - self.started

        by_min: list[list[RequestRecord]] = [[] for _ in range(HISTORY_MINUTES)]
        for t, r in recs:
            ago = int((now - t) // 60)
            if 0 <= ago < HISTORY_MINUTES:
                by_min[ago].append(r)
        minutes = []
        for ago in range(HISTORY_MINUTES - 1, -1, -1):        # oldest first
            rs = by_min[ago]
            ok = [r for r in rs if r.status == 200]
            ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
            lat = [r.latency_s for r in ok if r.latency_s is not None]
            minutes.append({
                "minutes_ago": ago, "covered": uptime >= ago * 60,   # the server was up for this minute
                "requests": len(rs), "ok": len(ok),
                "failed": sum(r.status >= 500 for r in rs), "shed": sum(r.status == 429 for r in rs),
                "completion_tokens": sum(r.completion_tokens for r in rs),
                "ttft_p50_s": percentile(ttft, 50), "ttft_p95_s": percentile(ttft, 95),
                "latency_p95_s": percentile(lat, 95)})

        hour = _SloBucket()
        for _, r in recs:
            if slo_counted(r.status):
                hour.add(r, self.slo_ttft_s)
        slos = {}
        for name, target in SLO_TARGETS.items():
            bad, total = map(sum, zip(*(b.bad_and_total(name) for b in buckets))) if buckets else (0, 0)
            hbad, htotal = hour.bad_and_total(name)
            allowed = 1 - target
            burn = (hbad / htotal) / allowed if htotal else None
            slos[name] = {
                "target": target, "good": total - bad, "total": total,
                "compliance": round((total - bad) / total, 6) if total else None,
                "met": (total - bad) / total >= target if total else None,
                "budget_left": round(1 - bad / (allowed * total), 6) if total else None,
                "burn_rate_1h": round(burn, 3) if burn is not None else None,
                "requests_1h": htotal,
                "fast_burn": burn is not None and htotal >= BURN_MIN_REQUESTS and burn > FAST_BURN}
        slos["ttft"]["threshold_s"] = self.slo_ttft_s
        slos["shed"]["judged"] = False          # only at or below the rated load, unknown for the Space yet

        return {"now_unix": round(wall, 3), "uptime_s": round(uptime, 1),
                "slo_window_s": round(min(SLO_WINDOW_S, uptime), 1), "minutes": minutes,
                "snapshots": [{"seconds_ago": round(now - t, 1), "queue_len": s.queue_len,
                               "busy_slots": s.busy_slots, "n_slots": s.n_slots,
                               "tokens_per_s": round(s.tokens_per_s, 1)} for t, s in snaps],
                "slo": slos, "fast_burn_threshold": FAST_BURN, "burn_min_requests": BURN_MIN_REQUESTS}

    def stats(self, now: float | None = None) -> dict:
        """/stats: per rolling window, request counts, rates and latency percentiles."""
        now = self._clock() if now is None else now
        out: dict = {"uptime_s": round(now - self.started, 1), "windows": {}}
        with self._lock:
            for name, w in self._windows.items():
                w.prune(now)
                recs = [r for _, r in w.items]
                span = min(w.span_s, max(now - self.started, 1e-9))
                ok = [r for r in recs if r.status == 200]
                ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
                lat = [r.latency_s for r in ok if r.latency_s is not None]
                qw = [r.queue_wait_s for r in ok if r.queue_wait_s is not None]
                n = len(recs)
                out["windows"][name] = {
                    "requests": n,
                    "requests_per_min": round(60 * n / span, 3),
                    "ok": len(ok),
                    "error_rate": round(sum(r.status >= 500 for r in recs) / n, 4) if n else 0.0,
                    "reject_rate": round(sum(r.status == 429 for r in recs) / n, 4) if n else 0.0,
                    "clients": len({r.client for r in recs}),
                    "completion_tokens": sum(r.completion_tokens for r in recs),
                    "ttft_s": {f"p{q}": percentile(ttft, q) for q in (50, 95, 99)},
                    "latency_s": {f"p{q}": percentile(lat, q) for q in (50, 95, 99)},
                    "queue_wait_s": {f"p{q}": percentile(qw, q) for q in (50, 95)},
                }
            out["engine"] = asdict(self.last_snapshot) if self.last_snapshot else None
        return out

    def prometheus(self) -> str:
        """/metrics in the Prometheus text exposition format (version 0.0.4)."""
        L: list[str] = []
        with self._lock:
            L += ["# HELP tinyllm_requests_total Finished requests by variant and HTTP status.",
                  "# TYPE tinyllm_requests_total counter"]
            for (v, s), c in sorted(self._requests.items()):
                L.append(f'tinyllm_requests_total{{variant="{v}",status="{s}"}} {c}')
            L += ["# HELP tinyllm_completion_tokens_total Generated tokens by variant.",
                  "# TYPE tinyllm_completion_tokens_total counter"]
            for v, c in sorted(self._tokens.items()):
                L.append(f'tinyllm_completion_tokens_total{{variant="{v}"}} {c}')
            for name, hists, help_ in (
                    ("tinyllm_ttft_seconds", self._ttft, "Arrival to first token."),
                    ("tinyllm_request_latency_seconds", self._latency, "Arrival to last token, successful requests.")):
                L += [f"# HELP {name} {help_}", f"# TYPE {name} histogram"]
                for v, h in sorted(hists.items()):
                    L += h.lines(name, f'variant="{v}"')
            s = self.last_snapshot
            for name, help_, val in (
                    ("tinyllm_queue_length", "Requests waiting for a slot.", s and s.queue_len),
                    ("tinyllm_busy_slots", "Slots decoding a request.", s and s.busy_slots),
                    ("tinyllm_slots", "Total slots.", s and s.n_slots),
                    ("tinyllm_tokens_per_second", "Generated tokens per second (last snapshot).", s and s.tokens_per_s),
                    ("tinyllm_kv_used_ratio", "Share of the KV cache holding tokens.", s and s.kv_used_frac)):
                if s is not None:
                    L += [f"# HELP {name} {help_}", f"# TYPE {name} gauge", f"{name} {val:g}"]
        return "\n".join(L) + "\n"


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Structured fields go in `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": round(record.created, 3), "level": record.levelname.lower(),
             "logger": record.name, "msg": record.getMessage()}
        d.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def setup_logging(level: str = "info", path: str | None = None) -> logging.Logger:
    """The `tinyllm` logger: JSON lines to stderr, and to `path` (JSONL) if given."""
    log = logging.getLogger("tinyllm")
    log.setLevel(level.upper())
    log.handlers.clear()
    log.propagate = False
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if path:
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
    return log
