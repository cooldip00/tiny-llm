"""Step 4.13 step 2: OpenAI-compatible HTTP server over ContinuousEngine (D-025, D-028).

Design: docs/SERVING.md.

    client ──HTTP──▶ FastAPI handler ──▶ Router ──▶ EngineWorker thread: ContinuousEngine.step()
       ◀── JSON / SSE ◀── per-request asyncio.Queue ◀── tokens (call_soon_threadsafe)

- One thread owns the engine (it is not thread-safe). Handlers talk to it through an inbox.
- POST /v1/completions: prompt as text or token ids, max_tokens, temperature, top_p, top_k,
  ignore_eos, stream (SSE ending with `data: [DONE]`), finish_reason, usage.
- Hardening:
  - a bounded queue → 429 + Retry-After
  - a per-client token-bucket rate limit → 429
  - a per-request timeout → 504, slot freed
  - client disconnect → abort, slot freed
  - graceful shutdown
  - an engine error → 500 for in-flight requests and not ready
  - out of memory → 503 for in-flight requests, then recover
- Observability (obs.py): a JSON log line per request, X-Request-ID, engine snapshots,
  /stats, /metrics (Prometheus text format), --log-level debug with per-step timings,
  slow-request timelines, and /debug/state (localhost only).
- Pages (6.5b): GET / is the demo (story continuation, streamed), GET /status the status page
  (charts and SLOs from GET /stats/history: the last hour per minute, the 7-day error budget).
- Router with one variant now; X-Model-Version on every response. Release strategies (6.9)
  add routing rules.

Run the local GPU server (TinyLlama) or our 30M model:

    uv run python -m tinyllm.server --model tinyllama --slots 8 --cuda-graph
    uv run python -m tinyllm.server --ckpt checkpoints/main/best.pt --slots 8

or our 30M model in HF format (6.5b), from a folder or from the Hub (the Docker image does this):

    uv run python -m tinyllm.server --hf-model runs/export/tinystories-30m
    uv run python -m tinyllm.server --hf-model CoolDip00/tiny-llm-tinystories-30m --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import queue
import re
import secrets
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from fastapi import FastAPI
from fastapi import Request as HTTPRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from tinyllm.engine import ContinuousEngine, Request, Sampling
from tinyllm.model import norm_backend
from tinyllm.obs import EngineSnapshot, Metrics, RequestRecord, hash_client, itl_stats, setup_logging

log = logging.getLogger("tinyllm.server")


class Tokenizer(Protocol):
    eos_id: int | None

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# -- engine thread ----------------------------------------------------------------------

@dataclass
class Faults:
    """Test-only fault injection (D-028 amendment): forces the timeout / 500 / OOM paths."""
    step_delay_s: float = 0.0           # sleep before every step (slow engine)
    fail_at_step: int | None = None     # RuntimeError at step N (1-based): fatal, not ready
    oom_at_step: int | None = None      # torch.OutOfMemoryError at step N: recoverable


@dataclass
class Item:
    """What the engine thread sends a request's handler."""
    kind: str                           # "token" or "error"
    token: int = 0
    t: float = 0.0                      # time.monotonic() when the token was produced
    finish: str | None = None           # "stop" / "length" on the last token
    queue_wait: float | None = None     # on the first token: arrival → start of its step
    status: int = 200                   # errors: 400, 500, 503, 504
    message: str = ""


@dataclass
class Job:
    request_id: str
    request: Request
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue
    arrival: float                      # time.monotonic()
    deadline: float
    rid: int | None = None              # engine id, set by the engine thread
    n_tokens: int = 0

    def send(self, item: Item) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)


class EngineWorker:
    """Owns one ContinuousEngine on its own thread. submit() / abort() / debug_state() are
    safe to call from the event loop; everything else runs on the engine thread."""

    def __init__(self, engine: ContinuousEngine, metrics: Metrics, max_queue: int,
                 snapshot_s: float = 5.0, faults: Faults | None = None, name: str = "main"):
        self.engine = engine
        self.metrics = metrics
        self.max_outstanding = engine.n_slots + max_queue
        self.snapshot_s = snapshot_s
        self.faults = faults or Faults()
        self.name = name
        self.accepting = True
        self.failed: str | None = None
        self.steps = 0
        self.busy = 0
        self._outstanding = 0
        self._lock = threading.Lock()
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._jobs: dict[int, Job] = {}            # engine rid → job (engine thread only)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"engine-{name}", daemon=True)
        self._tokens_since_snap = 0
        self._last_snap = time.monotonic()

    # -- event-loop side ------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def load(self) -> tuple[int, int]:
        """(queued, busy) as last seen; a gauge, good enough for logs and admission."""
        with self._lock:
            return max(0, self._outstanding - self.busy), self.busy

    def submit(self, job: Job) -> str | None:
        """Queue a job. Returns None, or why it was refused: "failed", "draining", "full"."""
        with self._lock:
            if self.failed:
                return "failed"
            if not self.accepting:
                return "draining"
            if self._outstanding >= self.max_outstanding:
                return "full"
            self._outstanding += 1
        self._inbox.put(("add", job))
        return None

    def abort(self, job: Job) -> None:
        self._inbox.put(("abort", job))

    async def debug_state(self) -> dict:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._inbox.put(("debug", fut))
        return await asyncio.wait_for(asyncio.wrap_future(fut), 5.0)

    def stop(self, drain_s: float) -> None:
        """Stop accepting, let in-flight requests finish (up to drain_s), then stop."""
        with self._lock:
            self.accepting = False
        end = time.monotonic() + drain_s
        while time.monotonic() < end and self.outstanding() and self.alive:
            time.sleep(0.02)
        self._stop.set()
        self._thread.join(timeout=10)

    def outstanding(self) -> int:
        with self._lock:
            return self._outstanding

    # -- engine thread --------------------------------------------------------------
    def _finish(self, job: Job) -> None:
        if job.rid is not None:
            self._jobs.pop(job.rid, None)
            self.engine.results.pop(job.rid, None)
        with self._lock:
            self._outstanding -= 1

    def _fail_all(self, status: int, message: str) -> None:
        for job in list(self._jobs.values()):
            if self.failed is None:
                self.engine.abort(job.rid)
            job.send(Item("error", status=status, message=message))
            self._finish(job)

    def _handle(self, cmd: str, arg) -> None:
        if cmd == "add":
            job: Job = arg
            if self.failed or self._stop.is_set():
                job.send(Item("error", status=503 if not self.failed else 500,
                              message="engine failed" if self.failed else "shutting down"))
                self._finish(job)
                return
            try:
                job.rid = self.engine.add_request(job.request)
            except ValueError as e:
                job.send(Item("error", status=400, message=str(e)))
                self._finish(job)
                return
            self._jobs[job.rid] = job
        elif cmd == "abort":
            job = arg
            if job.rid is not None and self._jobs.get(job.rid) is job:
                if self.failed is None:
                    self.engine.abort(job.rid)
                self._finish(job)
        elif cmd == "debug":
            arg.set_result(self._debug_state())

    def _drain_inbox(self, block: bool) -> None:
        try:
            cmd, arg = self._inbox.get(timeout=0.05) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        self._handle(cmd, arg)
        while True:
            try:
                cmd, arg = self._inbox.get_nowait()
            except queue.Empty:
                return
            self._handle(cmd, arg)

    def _expire(self, now: float) -> None:
        for job in [j for j in self._jobs.values() if now > j.deadline]:
            self.engine.abort(job.rid)
            job.send(Item("error", status=504, message="request timed out"))
            self._finish(job)

    def _step(self) -> None:
        self.steps += 1
        f = self.faults
        if f.step_delay_s:
            time.sleep(f.step_delay_s)
        if f.fail_at_step == self.steps:
            raise RuntimeError(f"injected fault at step {self.steps}")
        if f.oom_at_step == self.steps:
            raise torch.OutOfMemoryError(f"injected OOM at step {self.steps}")
        t0 = time.monotonic()
        events = self.engine.step()
        t1 = time.monotonic()
        for e in events:
            job = self._jobs.get(e.rid)
            if job is None:
                continue
            job.n_tokens += 1
            finish = None
            if e.done:
                eos = job.request.eos_id
                finish = "stop" if eos is not None and e.token == eos else "length"
            job.send(Item("token", e.token, t1, finish,
                          queue_wait=t0 - job.arrival if job.n_tokens == 1 else None))
            if e.done:
                self._finish(job)
        self._tokens_since_snap += len(events)
        queued, busy = self.engine.load()
        with self._lock:
            self.busy = busy
        if log.isEnabledFor(logging.DEBUG):
            log.debug("step", extra={"fields": {"step": self.steps, "step_ms": round(1e3 * (t1 - t0), 3),
                                                "tokens": len(events), "queued": queued, "busy": busy}})

    def _snapshot(self, now: float) -> None:
        dt = max(now - self._last_snap, 1e-9)
        queued, busy = self.engine.load()
        snap = EngineSnapshot(time.time(), queued, busy, self.engine.n_slots,
                              round(self._tokens_since_snap / dt, 2), round(self.engine.kv_used_frac(), 4),
                              ram_mb=_rss_mb(), vram_mb=_vram_mb(self.engine.device))
        self.metrics.snapshot(snap)
        if busy or queued or self._tokens_since_snap:
            log.info("engine", extra={"fields": {"variant": self.name, **asdict(snap)}})
        self._tokens_since_snap, self._last_snap = 0, now

    def _run(self) -> None:
        self._snapshot(time.monotonic())
        while not self._stop.is_set():
            idle = self.failed is not None or not self.engine.has_work()
            self._drain_inbox(block=idle)
            now = time.monotonic()
            if now - self._last_snap >= self.snapshot_s and self.failed is None:
                self._snapshot(now)
            if self.failed is not None:
                continue
            self._expire(now)
            if not self.engine.has_work():
                continue
            try:
                self._step()
            except torch.OutOfMemoryError as e:
                log.error("out of memory: aborting in-flight requests", exc_info=True,
                          extra={"fields": {"variant": self.name, "in_flight": len(self._jobs)}})
                self._fail_all(503, f"out of memory, retry later ({e})")
                if self.engine.device.type == "cuda":
                    torch.cuda.empty_cache()
                if self.engine.has_work():                       # could not get back to a clean state
                    self.failed = "out of memory, engine not clean"
            except Exception as e:                              # engine state unknown: stop serving
                self.failed = f"{type(e).__name__}: {e}"
                log.error("engine step failed: not ready", exc_info=True,
                          extra={"fields": {"variant": self.name, "in_flight": len(self._jobs)}})
                self._fail_all(500, "engine error")
                with self._lock:
                    self.busy = 0
        # stopped: answer whatever is left
        self._fail_all(503, "server shutting down")
        while True:
            try:
                cmd, arg = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._handle(cmd, arg)

    def _debug_state(self) -> dict:
        now = time.monotonic()
        queued, busy = self.engine.load()
        return {"variant": self.name, "steps": self.steps, "failed": self.failed, "accepting": self.accepting,
                "queued": queued, "busy": busy, "n_slots": self.engine.n_slots,
                "kv_used_frac": round(self.engine.kv_used_frac(), 4),
                "jobs": [{"request_id": j.request_id, "rid": j.rid, "age_s": round(now - j.arrival, 3),
                          "tokens": j.n_tokens, "max_tokens": j.request.max_new_tokens,
                          "deadline_in_s": round(j.deadline - now, 3)} for j in self._jobs.values()]}


def _rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def _vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.memory_allocated(device) / 2**20, 1)


# -- routing, limits --------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    version: str                        # checkpoint hash + git commit
    worker: EngineWorker
    tokenizer: Tokenizer
    max_len: int
    vocab_size: int


class Router:
    """Picks the variant for a request. One variant now; 6.9 adds blue-green / canary /
    shadow / A/B rules here, so handlers never change."""

    def __init__(self, variants: list[Variant]):
        self.variants = {v.name: v for v in variants}
        self.primary = variants[0]

    def pick(self, client: str) -> Variant:
        return self.primary


class RateLimiter:
    """Token bucket per client: `per_min` requests per minute, bursts up to `burst`.
    Called from the event loop only. per_min <= 0 disables it."""

    def __init__(self, per_min: float, burst: int):
        self.rate = per_min / 60.0
        self.burst = max(1, burst)
        self._b: dict[str, tuple[float, float]] = {}

    def check(self, client: str, now: float) -> float:
        """0 if allowed (and counted), else seconds until the next request is allowed."""
        if self.rate <= 0:
            return 0.0
        tokens, last = self._b.get(client, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.rate)
        if tokens >= 1.0:
            self._b[client] = (tokens - 1.0, now)
            if len(self._b) > 10_000:                     # forget clients whose bucket is full again
                self._b = {k: v for k, v in self._b.items() if v[0] + (now - v[1]) * self.rate < self.burst}
            return 0.0
        self._b[client] = (tokens, now)
        return (1.0 - tokens) / self.rate


# -- HTTP layer -------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    """OpenAI /v1/completions subset, plus vLLM's top_k / ignore_eos / return_token_ids."""
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1)
    stream: bool = False
    ignore_eos: bool = False
    return_token_ids: bool = False


@dataclass
class ServerConfig:
    max_queue: int = 32                  # waiting requests beyond the slots; more → 429
    timeout_s: float = 60.0              # per request, arrival → last token; then 504
    rate_per_min: float = 0.0            # per client; 0 = off (the Space sets it)
    rate_burst: int = 10
    slow_s: float = 10.0                 # requests slower than this log their timeline (warning)
    trust_proxy: bool = False            # take the client IP from X-Forwarded-For (HF Space)
    proxy_hops: int = 1                  # with trust_proxy: trusted proxies in front, each appending to it
    salt: str | None = None              # client-hash salt; None = resolve_salt()'s order
    snapshot_s: float = 5.0
    drain_s: float = 30.0


_RID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class HeadersMiddleware:
    """Pure ASGI (keeps streaming and disconnects intact): every response gets X-Request-ID
    (the client's, if it sent a sane one) and X-Model-Version."""

    def __init__(self, app, default_version: str):
        self.app = app
        self.default_version = default_version

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _RID.match(incoming) else uuid.uuid4().hex[:16]
        state = scope.setdefault("state", {})
        state["request_id"] = rid

        async def send_with_headers(msg):
            if msg["type"] == "http.response.start":
                version = state.get("model_version", self.default_version)
                msg = {**msg, "headers": [*msg.get("headers", []), (b"x-request-id", rid.encode()),
                                          (b"x-model-version", version.encode())]}
            await send(msg)

        await self.app(scope, receive, send_with_headers)


class _ClientGone(Exception):
    pass


WEB = Path(__file__).resolve().parent / "web"


def _page(name: str) -> str:
    """The demo and status pages (6.5b): static HTML + JS in src/tinyllm/web, read per request
    (small) so an edit shows on reload. They fetch the JSON endpoints with relative URLs."""
    return (WEB / name).read_text(encoding="utf-8")


def decode_completion(tok: Tokenizer, ids: list[int], context: list[int]) -> str:
    """Completion text as it reads after the prompt. Decoded behind the prompt's last token(s),
    because a SentencePiece decoder (TinyLlama) drops a leading space at the start of the text:
    "…named Max." + " Max was" must not come back as "Max was" (byte-level BPE keeps it anyway).
    Falls back to a plain decode if the context's text isn't a prefix (a partial UTF-8 byte)."""
    if context:
        head, text = tok.decode(context), tok.decode(context + ids)
        if text.startswith(head):
            return text[len(head):]
    return tok.decode(ids)


class _Detokenizer:
    """Incremental text for streaming: decode everything so far, emit the new suffix; hold
    back while the text ends in a partial UTF-8 character."""

    def __init__(self, tok: Tokenizer, context: list[int] | None = None):
        self.tok, self.ids, self.sent, self.context = tok, [], "", list(context or [])

    def add(self, token: int) -> str:
        self.ids.append(token)
        return self._delta(final=False)

    def flush(self) -> str:
        return self._delta(final=True)

    def _delta(self, final: bool) -> str:
        text = decode_completion(self.tok, self.ids, self.context)
        if (not final and text.endswith("�")) or not text.startswith(self.sent):
            return ""
        d, self.sent = text[len(self.sent):], text
        return d


def forwarded_ip(xff: str, hops: int = 1) -> str | None:
    """The client's address in an X-Forwarded-For header behind `hops` trusted proxies.

    Each proxy appends the address the request came from, so only the last `hops` entries
    are trustworthy; anything left of them is whatever the client sent (spoofable, so never
    the leftmost). With fewer entries than hops, all came from proxies: take the leftmost.
    """
    parts = [p.strip() for p in xff.split(",") if p.strip()]
    if not parts:
        return None
    return parts[-hops] if len(parts) >= hops else parts[0]


def resolve_salt(salt: str | None = None) -> tuple[str, str]:
    """(salt, where it came from) for the client hash. The source is logged, never the salt.

    Order: an explicit salt; $TINYLLM_LOG_SALT (a Space secret, the same value in every
    replica Space, D-031); a hash of $SPACE_SIGNING_SECRET (HF's per-Space secret, stable
    across restarts, but different in each Space); else random per start.
    """
    if salt:
        return salt, "config"
    if s := os.environ.get("TINYLLM_LOG_SALT"):
        return s, "TINYLLM_LOG_SALT"
    if s := os.environ.get("SPACE_SIGNING_SECRET"):
        return hashlib.sha256(f"tinyllm-log-salt:{s}".encode()).hexdigest(), "SPACE_SIGNING_SECRET"
    return secrets.token_hex(16), "random"


def cpu_limit(cgroup: str = "/sys/fs/cgroup") -> int | None:
    """CPUs this container may use, or None if no limit is known.

    torch sizes its thread pool from the host's cores and ignores the container's CPU quota:
    under `docker run --cpus 2` on a 16-core host it starts 8 threads, and decode ran ~5x
    slower than with 2 (6.5b step 5, runs/bench/load/space-cpu/probe-threads-*). Order: the
    cgroup v2 quota (`cpu.max`, "200000 100000" = 2 CPUs), cgroup v1, then HF's $CPU_CORES.
    """
    root = Path(cgroup)
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    try:
        n = math.ceil(float(os.environ.get("CPU_CORES", "")))
        return n if n > 0 else None
    except ValueError:
        return None


class Server:
    """State shared by the handlers (app.state.srv)."""

    def __init__(self, variants: list[Variant], cfg: ServerConfig, metrics: Metrics):
        self.router = Router(variants)
        self.cfg = cfg
        self.metrics = metrics
        self.limiter = RateLimiter(cfg.rate_per_min, cfg.rate_burst)
        self.salt, self.salt_source = resolve_salt(cfg.salt)
        self.started = False

    def client_id(self, request: HTTPRequest) -> str:
        ip = request.client.host if request.client else "unknown"
        if self.cfg.trust_proxy and (xff := request.headers.get("x-forwarded-for")):
            ip = forwarded_ip(xff, self.cfg.proxy_hops) or ip
        return hash_client(ip, self.salt)

    def record(self, rec: RequestRecord, timeline: dict | None = None) -> None:
        self.metrics.observe(rec)
        fields = rec.to_dict()
        if rec.latency_s is not None and rec.latency_s > self.cfg.slow_s:
            log.warning("slow request", extra={"fields": {**fields, "timeline": timeline or {}}})
        else:
            log.info("request", extra={"fields": fields})

    def error(self, rec: RequestRecord, status: int, message: str, headers: dict | None = None,
              err_type: str = "invalid_request_error") -> JSONResponse:
        rec.status, rec.error = status, message
        rec.finish_reason = rec.finish_reason or ("timeout" if status == 504 else "error")
        self.record(rec)
        return _error_response(status, message, err_type, headers)


def _error_response(status: int, message: str, err_type: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": err_type, "code": status}},
                        status_code=status, headers=headers)


def _finish_record(rec: RequestRecord, job: Job, times: list[float], toks: list[int]) -> dict:
    rec.completion_tokens = len(toks)
    if times:
        rec.ttft_s = round(times[0] - job.arrival, 6)
        rec.latency_s = round(times[-1] - job.arrival, 6)
        p50, p95 = itl_stats(times)
        rec.itl_p50_s = None if p50 is None else round(p50, 6)
        rec.itl_p95_s = None if p95 is None else round(p95, 6)
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"queue_wait_s": rec.queue_wait_s, "ttft_s": rec.ttft_s,
            "max_gap_s": round(max(gaps), 6) if gaps else None, "tokens": len(toks)}


def create_app(variants: list[Variant], cfg: ServerConfig | None = None,
               metrics: Metrics | None = None, shipper=None) -> FastAPI:
    """`shipper`: a logship.LogShipper; its last upload runs after the drain."""
    cfg = cfg or ServerConfig()
    metrics = metrics or variants[0].worker.metrics
    srv = Server(variants, cfg, metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for v in variants:
            v.worker.start()
        srv.started = True
        log.info("server ready", extra={"fields": {"variants": {v.name: v.version for v in variants},
                                                   "slo_ttft_s": metrics.slo_ttft_s, "salt": srv.salt_source,
                                                   "proxy_hops": cfg.proxy_hops if cfg.trust_proxy else 0}})
        yield
        srv.started = False
        log.info("shutting down: draining", extra={"fields": {"drain_s": cfg.drain_s}})
        await asyncio.gather(*(asyncio.to_thread(v.worker.stop, cfg.drain_s) for v in variants))
        log.info("stopped")
        if shipper is not None:
            await asyncio.to_thread(shipper.close, cfg.drain_s)

    app = FastAPI(title="tinyllm", lifespan=lifespan)
    app.state.srv = srv
    app.add_middleware(HeadersMiddleware, default_version=srv.router.primary.version)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: HTTPRequest, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        rec = RequestRecord(request.state.request_id, 400, client=srv.client_id(request),
                            arrival_unix=time.time())
        return srv.error(rec, 400, msg or "invalid request")

    @app.get("/health")
    async def health():
        dead = [v.name for v in variants if v.worker.failed or (srv.started and not v.worker.alive)]
        if dead:
            return JSONResponse({"status": "failed", "variants": dead}, status_code=503)
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        why = []
        if not srv.started:
            why.append("not started")
        for v in variants:
            if v.worker.failed:
                why.append(f"{v.name}: {v.worker.failed}")
            elif not v.worker.accepting:
                why.append(f"{v.name}: draining")
        if why:
            return JSONResponse({"ready": False, "reasons": why}, status_code=503)
        return {"ready": True}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": v.name, "object": "model", "owned_by": "tinyllm",
                                            "version": v.version, "max_len": v.max_len} for v in variants]}

    @app.get("/stats")
    async def stats():
        out = metrics.stats()
        out["variants"] = {v.name: {"version": v.version, "outstanding": v.worker.outstanding(),
                                    "n_slots": v.worker.engine.n_slots, "failed": v.worker.failed,
                                    "norm_backend": norm_backend(v.worker.engine.model)}
                           for v in variants}
        return out

    history_cache: dict = {}

    @app.get("/stats/history")
    async def stats_history():
        """For the status page: computed at most once a second, however many pages poll it."""
        now = time.monotonic()
        if history_cache.get("t", -1.0) < now - 1.0:
            out = metrics.history()
            out["variants"] = {v.name: {"version": v.version, "n_slots": v.worker.engine.n_slots,
                                        "norm_backend": norm_backend(v.worker.engine.model)}
                               for v in variants}
            history_cache.update(t=now, out=out)
        return history_cache["out"]

    @app.get("/", response_class=HTMLResponse)
    async def demo_page():
        return HTMLResponse(_page("demo.html"))

    @app.get("/status", response_class=HTMLResponse)
    async def status_page():
        return HTMLResponse(_page("status.html"))

    @app.get("/metrics")
    async def prometheus():
        return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/debug/state")
    async def debug_state(request: HTTPRequest):
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            return _error_response(404, "not found", "not_found")
        return {v.name: await v.worker.debug_state() for v in variants}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: HTTPRequest):
        arrival = time.monotonic()
        client = srv.client_id(request)
        rec = RequestRecord(request.state.request_id, 0, client=client, arrival_unix=time.time(),
                            stream=body.stream)
        variant = srv.router.pick(client)
        rec.variant, rec.model_version = variant.name, variant.version
        request.state.model_version = variant.version
        worker = variant.worker

        wait = srv.limiter.check(client, arrival)
        if wait:
            return srv.error(rec, 429, "rate limit exceeded", {"Retry-After": str(math.ceil(wait))},
                             "rate_limit_error")
        ids = body.prompt if isinstance(body.prompt, list) else variant.tokenizer.encode(body.prompt)
        rec.prompt_tokens = len(ids)
        if not ids:
            return srv.error(rec, 400, "prompt is empty")
        if any(not 0 <= t < variant.vocab_size for t in ids):
            return srv.error(rec, 400, f"token ids must be in [0, {variant.vocab_size})")
        if len(ids) + body.max_tokens > variant.max_len:
            return srv.error(rec, 400, f"prompt ({len(ids)} tokens) + max_tokens ({body.max_tokens}) "
                                       f"exceeds the context length {variant.max_len}")
        rec.queue_depth_at_arrival, rec.busy_slots_at_arrival = worker.load()
        sampling = Sampling(body.temperature, body.top_k, body.top_p)
        eos = None if body.ignore_eos else variant.tokenizer.eos_id
        job = Job(rec.request_id, Request(ids, body.max_tokens, eos, sampling), asyncio.get_running_loop(),
                  asyncio.Queue(), arrival, arrival + cfg.timeout_s)
        refused = worker.submit(job)
        if refused == "full":
            return srv.error(rec, 429, "server busy: queue full, retry later", {"Retry-After": "1"},
                             "server_busy")
        if refused:
            return srv.error(rec, 503, "engine failed" if refused == "failed" else "server shutting down",
                             err_type="server_error")

        created, cid = int(time.time()), f"cmpl-{rec.request_id}"

        async def items(poll: HTTPRequest | None):
            """The job's items. With `poll`, also check every 0.25 s whether the client left
            (busy requests too: tokens arriving must not hide a disconnect)."""
            checked = time.monotonic()
            while True:
                if poll is None:
                    item = await job.out.get()
                else:
                    try:
                        item = await asyncio.wait_for(job.out.get(), 0.25)
                    except TimeoutError:
                        item = None
                    if time.monotonic() - checked >= 0.25:
                        checked = time.monotonic()
                        if await poll.is_disconnected():
                            raise _ClientGone
                    if item is None:
                        continue
                if item.kind == "token" and item.queue_wait is not None:
                    rec.queue_wait_s = round(item.queue_wait, 6)
                yield item
                if item.kind == "error" or item.finish:
                    return

        def usage(n: int) -> dict:
            return {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}

        if not body.stream:
            toks, times, finish = [], [], None
            try:
                async for it in items(request):
                    if it.kind == "error":
                        timeline = _finish_record(rec, job, times, toks)
                        return srv.error(rec, it.status, it.message, err_type="server_error"
                                         if it.status >= 500 else "invalid_request_error")
                    toks.append(it.token)
                    times.append(it.t)
                    finish = it.finish
            except _ClientGone:
                worker.abort(job)
                _finish_record(rec, job, times, toks)
                rec.status, rec.finish_reason = 499, "abort"
                srv.record(rec)
                return PlainTextResponse("", status_code=499)
            timeline = _finish_record(rec, job, times, toks)
            rec.status, rec.finish_reason = 200, finish
            srv.record(rec, timeline)
            choice = {"index": 0, "text": decode_completion(variant.tokenizer, toks, ids[-1:]), "logprobs": None,
                      "finish_reason": finish}
            if body.return_token_ids:
                choice["token_ids"] = toks
            return {"id": cid, "object": "text_completion", "created": created, "model": variant.name,
                    "choices": [choice], "usage": usage(len(toks))}

        def chunk(text: str, finish: str | None, token_ids: list[int] | None = None, **extra) -> str:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            if token_ids is not None:
                choice["token_ids"] = token_ids
            d = {"id": cid, "object": "text_completion", "created": created, "model": variant.name,
                 "choices": [choice], **extra}
            return f"data: {json.dumps(d)}\n\n"

        async def stream():
            toks, times, detok, done = [], [], _Detokenizer(variant.tokenizer, ids[-1:]), False
            try:
                async for it in items(None):          # disconnect: Starlette cancels this generator
                    if it.kind == "error":
                        rec.status, rec.error = it.status, it.message
                        rec.finish_reason = "timeout" if it.status == 504 else "error"
                        done = True
                        yield (f"data: {json.dumps({'error': {'message': it.message, 'type': 'server_error', 'code': it.status}})}\n\n"
                               "data: [DONE]\n\n")
                        return
                    toks.append(it.token)
                    times.append(it.t)
                    text = detok.add(it.token)
                    ids_ = [it.token] if body.return_token_ids else None
                    if it.finish:
                        rec.status, rec.finish_reason, done = 200, it.finish, True
                        yield chunk(text + detok.flush(), it.finish, ids_, usage=usage(len(toks)))
                        yield "data: [DONE]\n\n"
                        return
                    if text or ids_:
                        yield chunk(text, None, ids_)
            finally:
                if not done:
                    worker.abort(job)
                    rec.status, rec.finish_reason = 499, "abort"
                timeline = _finish_record(rec, job, times, toks)
                srv.record(rec, timeline)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


# -- loading and CLI --------------------------------------------------------------------

class StoryTokenizer:
    """Our 30M model's TinyStories BPE (no BOS; EOS separates stories)."""

    def __init__(self, path: Path | None = None):
        from tinyllm.tokenizer import EOS, TOKENIZER_PATH, load_tokenizer
        self.tok = load_tokenizer(path or TOKENIZER_PATH)
        self.eos_id = self.tok.token_to_id(EOS)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


def git_commit() -> str:
    if c := os.environ.get("TINYLLM_GIT_COMMIT"):
        return c[:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8]


HF_MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                  "tokenizer_config.json"]


def resolve_hf_model(ref: str, revision: str | None = None, local_only: bool = False) -> Path:
    """--hf-model: a local folder as is, else a Hub repo id downloaded to the HF cache (6.5b)."""
    if Path(ref).is_dir():
        if revision:
            raise ValueError("--revision is for a Hub repo id, not a local folder")
        return Path(ref)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ref, revision=revision, allow_patterns=HF_MODEL_FILES,
                                  local_files_only=local_only))


def load_story_hf(path: Path, device: str) -> tuple[torch.nn.Module, Tokenizer]:
    """Our story model as exported by scripts/export_hf.py (4.8b): fp32, as trained (like --ckpt).

    Only that export: its tokenizer is our TinyStories BPE (no BOS, EOS <|endoftext|>), which
    StoryTokenizer reads from the folder. Any other Llama folder would be tokenized wrongly.
    """
    from tinyllm.hf_llama import load_from_dir
    from tinyllm.tokenizer import EOS
    tc = json.loads((path / "tokenizer_config.json").read_text())
    if tc.get("eos_token") != EOS or tc.get("add_bos_token") or tc.get("bos_token"):
        raise ValueError(f"{path}: not a tinyllm story-model export (eos {tc.get('eos_token')!r}, "
                         f"bos {tc.get('bos_token')!r}); for TinyLlama use --model tinyllama")
    tok = StoryTokenizer(path / "tokenizer.json")
    eos_cfg = json.loads((path / "config.json").read_text()).get("eos_token_id")
    if tok.eos_id != eos_cfg:
        raise ValueError(f"{path}: tokenizer EOS id {tok.eos_id} != config.json eos_token_id {eos_cfg}")
    return load_from_dir(path, device=device, dtype=torch.float32), tok


def load_variant(args) -> tuple[torch.nn.Module, Tokenizer, str, str]:
    """(model, tokenizer, name, version) from --model tinyllama, --ckpt or --hf-model."""
    from tinyllm.model import Transformer
    device = args.device
    if args.hf_model:
        path = resolve_hf_model(args.hf_model, args.revision, args.offline)
        model, tok = load_story_hf(path, device)
        name = args.hf_model.rstrip("/").split("/")[-1]
        return model, tok, name, f"{name}@{_file_hash(path / 'model.safetensors')}+{git_commit()}"
    if args.ckpt:
        from tinyllm.config import ModelConfig
        path = Path(args.ckpt)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = Transformer(ModelConfig.from_dict(ckpt["model_cfg"])).to(device)
        model.load_state_dict(ckpt["model"])
        name = f"tinyllm-{path.parent.name}"
        return model.eval(), StoryTokenizer(), name, f"{name}@{_file_hash(path)}+{git_commit()}"
    from tinyllm.hf_llama import TINYLLAMA_REV, HFTokenizer, download, load_from_dir
    path = download(local_only=args.offline)
    model = load_from_dir(path, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    return model, HFTokenizer(path), "tinyllama-1.1b-chat", f"tinyllama-chat-v1.0@{TINYLLAMA_REV[:7]}+{git_commit()}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="tinyllm OpenAI-compatible server (/v1/completions)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=["tinyllama"])
    src.add_argument("--ckpt", help="our checkpoint, e.g. checkpoints/main/best.pt")
    src.add_argument("--hf-model", metavar="DIR_OR_REPO",
                     help="our model in HF format (scripts/export_hf.py): a folder, or a Hub repo id")
    p.add_argument("--revision", default=None, help="with a --hf-model repo id: branch, tag or commit")
    p.add_argument("--offline", action="store_true", help="Hub models from the HF cache only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--max-len", type=int, default=None, help="per-slot context; default the model's")
    p.add_argument("--kv", choices=["slots", "paged"], default="slots")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-queue", type=int, default=32)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--rate-limit", type=float, default=0.0, help="requests/min per client; 0 = off")
    p.add_argument("--rate-burst", type=int, default=10)
    p.add_argument("--slow-s", type=float, default=10.0)
    p.add_argument("--slo-ttft-s", type=float, default=1.0,
                   help="TTFT SLO threshold on /status (SERVING.md: 1 s local GPU, 2 s on the Space)")
    p.add_argument("--trust-proxy", action="store_true", help="client IP from X-Forwarded-For")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU only: torch threads; default the container's CPU limit (cpu_limit), else torch's")
    p.add_argument("--proxy-hops", type=int, default=1,
                   help="with --trust-proxy: trusted proxies in front (client = that many entries from the right)")
    p.add_argument("--drain-s", type=float, default=30.0)
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    p.add_argument("--log-file", default=None, help="also write JSON lines here")
    p.add_argument("--log-dir", default=None,
                   help="write the shipped log lines (no text, no IP) as hourly JSONL here (logship.py)")
    p.add_argument("--log-dataset", default=None, metavar="REPO_ID",
                   help="with --log-dir: upload it to this private HF dataset every 10 min")
    p.add_argument("--log-replica", default=None, help="name in the log file names; default $SPACE_ID or local")
    args = p.parse_args(argv)
    if args.log_dataset and not args.log_dir:
        p.error("--log-dataset needs --log-dir")
    if args.proxy_hops < 1:
        p.error("--proxy-hops must be at least 1")
    if args.threads is not None and args.threads < 1:
        p.error("--threads must be at least 1")
    if args.device == "cpu":
        # an explicit --threads as given; else never more than torch's own choice (no SMT oversubscription)
        limit = cpu_limit()
        n = args.threads or (min(limit, torch.get_num_threads()) if limit else None)
        if n:
            torch.set_num_threads(n)

    setup_logging(args.log_level, args.log_file)
    shipper = None
    if args.log_dir:
        from tinyllm.logship import LogShipper
        shipper = LogShipper(args.log_dir, args.log_dataset, args.log_replica).attach()
        shipper.start_upload()
    t = time.perf_counter()
    model, tok, name, version = load_variant(args)
    gen = torch.Generator(args.device).manual_seed(args.seed) if args.seed is not None else None
    engine = ContinuousEngine(model, args.slots, args.max_len, generator=gen, kv=args.kv,
                              cuda_graph=args.cuda_graph)
    log.info("model loaded", extra={"fields": {"variant": name, "version": version, "device": args.device,
                                               "slots": args.slots, "max_len": engine.max_len, "kv": args.kv,
                                               "cuda_graph": args.cuda_graph, "threads": torch.get_num_threads(),
                                               "load_s": round(time.perf_counter() - t, 2)}})
    cfg = ServerConfig(max_queue=args.max_queue, timeout_s=args.timeout, rate_per_min=args.rate_limit,
                       rate_burst=args.rate_burst, slow_s=args.slow_s, trust_proxy=args.trust_proxy,
                       proxy_hops=args.proxy_hops, drain_s=args.drain_s)
    metrics = Metrics(slo_ttft_s=args.slo_ttft_s)
    worker = EngineWorker(engine, metrics, cfg.max_queue, cfg.snapshot_s, name=name)
    app = create_app([Variant(name, version, worker, tok, engine.max_len, model.cfg.vocab_size)], cfg, metrics,
                     shipper)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False,
                timeout_graceful_shutdown=int(args.drain_s))


if __name__ == "__main__":
    main()
