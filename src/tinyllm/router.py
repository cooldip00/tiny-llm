"""The replica router (D-031): one public Space in front of two private model Spaces.

Design and limits: docs/SERVING.md "Scaling beyond one server". No torch here: the router
holds no model, so it has a small image and starts quickly.

The decisions do no I/O: `classify` says what a replica's reply means, `Picker` chooses
where a request goes. `ReplicaRouter` carries a visitor's request to a replica and back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum

import httpx
from fastapi import FastAPI
from fastapi import Request as HTTPRequest
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from tinyllm.obs import EngineSnapshot, Metrics, RequestRecord, itl_stats, setup_logging
from tinyllm.web_common import (CLIENT_HEADER, HeadersMiddleware, RateLimiter, client_hash, error_response, read_page,
                                resolve_salt)

log = logging.getLogger("tinyllm.router")


class Verdict(Enum):
    """What one reply from a replica means. See `classify`."""
    OK = "ok"                       # our server answered 2xx
    NOT_READY = "not_ready"         # 503 JSON: starting, draining or engine failed
    ASLEEP = "asleep"               # HF's paused page: wake it if traffic needs it
    DOWN = "down"                   # 502, connection error, timeout, or a page that isn't ours
    MISCONFIGURED = "misconfigured" # token or URL is wrong: never wake, log, show on the status page
    BUSY = "busy"                   # 429 server_busy: queue full, try another replica
    PASS = "pass"                   # any other answer from our server: the visitor should see it


# A request that gets one of these never reached a working model, so another replica may try it.
RETRYABLE = frozenset({Verdict.NOT_READY, Verdict.ASLEEP, Verdict.DOWN, Verdict.MISCONFIGURED, Verdict.BUSY})


@dataclass(frozen=True)
class Reply:
    """A replica's answer. `status` is None for a connection error or timeout."""
    status: int | None
    content_type: str = ""
    text: str = ""                  # the body; only read when the status isn't 2xx


def _error_type(text: str) -> str | None:
    try:
        return json.loads(text)["error"]["type"]
    except (ValueError, KeyError, TypeError):
        return None


def classify(reply: Reply) -> Verdict:
    """The reply table in docs/SERVING.md, as code. Anything unrecognised is DOWN: never ready,
    never woken, and never shown to a visitor, because it didn't come from our server."""
    status = reply.status
    if status is None or status == 502 or 300 <= status < 400:      # redirects aren't followed: the token stays here
        return Verdict.DOWN
    if status in (401, 403):        # HF refused the token, whatever the format of its reply
        return Verdict.MISCONFIGURED
    kind = reply.content_type.split(";")[0].strip().lower()
    if kind == "text/html":         # our API never answers with HTML, so this is HF's own page
        if status == 404:
            return Verdict.MISCONFIGURED
        if status == 503 and "paused" in reply.text.lower():
            return Verdict.ASLEEP
        return Verdict.DOWN
    if kind not in ("application/json", "text/event-stream"):
        return Verdict.DOWN
    if 200 <= status < 300:
        return Verdict.OK
    if status == 503:
        return Verdict.NOT_READY
    if status == 429 and _error_type(reply.text) == "server_busy":
        return Verdict.BUSY
    return Verdict.PASS


@dataclass(eq=False)
class Replica:
    """What the router knows about one replica."""
    name: str                       # shown to visitors in X-Replica; never the private URL
    url: str
    verdict: Verdict | None = None  # from the last poll; None = not polled yet
    outstanding: int = 0            # from the last poll
    n_slots: int = 0
    sent_since_poll: int = 0        # a poll can be 5 s old, so count what we have sent since
    polled_at: float | None = None
    space_id: str = ""              # owner/name on the Hub, for waking it
    woken_at: float | None = None
    model: dict | None = None       # id, version and norm_backend, from the last readable /stats

    @property
    def ready(self) -> bool:
        return self.verdict is Verdict.OK

    @property
    def load(self) -> int:
        return self.outstanding + self.sent_since_poll

    def refused(self, verdict: Verdict) -> None:
        """A request came back with a RETRYABLE verdict. Only a full queue leaves it ready."""
        self.sent_since_poll = max(0, self.sent_since_poll - 1)
        if verdict is not Verdict.BUSY:
            self.verdict = verdict

    def polled(self, verdict: Verdict, now: float, outstanding: int = 0, n_slots: int = 0) -> None:
        self.verdict, self.polled_at = verdict, now
        self.outstanding, self.n_slots, self.sent_since_poll = outstanding, n_slots, 0


class Picker:
    """Shortest queue among the ready replicas; ties go round-robin. Event loop only."""

    def __init__(self, replicas: list[Replica]):
        self.replicas = replicas
        self._last = -1

    def pick(self, exclude: tuple[Replica, ...] = ()) -> Replica | None:
        """The replica to send to, counted as sent; None if no ready replica is left."""
        n = len(self.replicas)
        in_turn = [self.replicas[(self._last + 1 + i) % n] for i in range(n)]
        ready = [r for r in in_turn if r.ready and r not in exclude]
        if not ready:
            return None
        choice = min(ready, key=lambda r: r.load)       # min keeps the first of equals: the next in turn
        self._last = self.replicas.index(choice)
        choice.sent_since_poll += 1
        return choice


# -- the request path -------------------------------------------------------------------

MAX_ATTEMPTS = 2                    # the first try and at most one retry
NO_REPLICA = "none"                 # a record's `variant` when no replica was reached
_PASSED_HEADERS = ("content-type", "cache-control", "retry-after")  # all a visitor sees of a replica's headers
_STREAM_ERROR = (b'data: {"error": {"message": "the replica failed", "type": "server_error", "code": 502}}\n\n'
                 b"data: [DONE]\n\n")


@dataclass
class RouterConfig:
    rate_per_min: float = 20.0
    rate_burst: int = 5
    trust_proxy: bool = False       # visitor address from X-Forwarded-For (on a Space)
    proxy_hops: int = 1
    salt: str | None = None
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 65.0    # longer than a replica's own 60 s limit, so the replica answers first
    max_body_bytes: int = 1 << 20
    waking_retry_s: int = 30        # Retry-After when no replica is ready; a wake measured 35 s
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 2.0     # for a whole poll of one replica
    idle_stop_s: float = 600.0      # polling stops this long after the last visitor request
    wake_cooldown_s: float = 300.0  # per replica
    model_label: str | None = None  # for /v1/models and the status page before any replica has been polled
    drain_s: float = 30.0


class _ClientGone(Exception):
    """The visitor disconnected."""


class _BodyTooLarge(Exception):
    pass


class _NoAnswer(Exception):
    """A replica accepted the request and then went silent. It keeps the request; no retry."""


def _json(data: bytes) -> dict:
    try:
        out = json.loads(data)
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def _wants_stream(body: bytes) -> bool:
    return _json(body).get("stream") is True


async def _read_body(request: HTTPRequest, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _BodyTooLarge
    parts, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _BodyTooLarge
        parts.append(chunk)
    return b"".join(parts)


async def _unless_gone(request: HTTPRequest, work: Awaitable):
    """Await `work`; if the visitor leaves first, cancel it (which closes the replica's connection)."""
    task = asyncio.ensure_future(work)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if done:
                return task.result()
            if await request.is_disconnected():
                raise _ClientGone
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class _StreamMeter:
    """Reads the events of a stream as they pass through, for the request record."""

    def __init__(self):
        self.buffer = b""
        self.times: list[float] = []
        self.prompt_tokens = self.completion_tokens = 0
        self.finish: str | None = None
        self.error: dict | None = None
        self.done = False

    def feed(self, chunk: bytes, now: float) -> None:
        self.buffer += chunk
        *events, self.buffer = self.buffer.split(b"\n\n")
        for event in events:
            self._event(event, now)

    def _event(self, event: bytes, now: float) -> None:
        if not event.startswith(b"data: "):
            return
        payload = event[len(b"data: "):]
        if payload == b"[DONE]":
            self.done = True
            return
        d = _json(payload)
        if isinstance(d.get("error"), dict):
            self.error = d["error"]
        elif d.get("choices"):
            self.times.append(now)
            self.finish = d["choices"][0].get("finish_reason") or self.finish
            usage = d.get("usage") or {}
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)


def _reply(resp: httpx.Response | None) -> Reply:
    return Reply(None) if resp is None else Reply(resp.status_code, resp.headers.get("content-type", ""), resp.text)


def _load(stats: bytes) -> tuple[int, int] | None:
    """(outstanding, n_slots) from a replica's /stats, or None if it doesn't have them."""
    variants = _json(stats).get("variants")
    if not isinstance(variants, dict) or not variants:
        return None
    try:
        return sum(int(v["outstanding"]) for v in variants.values()), sum(int(v["n_slots"]) for v in variants.values())
    except (KeyError, TypeError, ValueError):
        return None


def _model(stats: bytes) -> dict | None:
    """What a replica serves, from its /stats. A replica has one variant."""
    variants = _json(stats).get("variants")
    if not isinstance(variants, dict) or not variants:
        return None
    name, v = next(iter(variants.items()))
    return {"id": name, "version": v.get("version"), "norm_backend": v.get("norm_backend")} if isinstance(v, dict) else None


class ReplicaRouter:
    def __init__(self, replicas: list[Replica], cfg: RouterConfig, token: str | None = None,
                 metrics: Metrics | None = None, wake: Callable[[Replica], None] | None = None):
        self.cfg = cfg
        self.replicas = replicas
        self.picker = Picker(replicas)
        self.limiter = RateLimiter(cfg.rate_per_min, cfg.rate_burst)
        self.salt, self.salt_source = resolve_salt(cfg.salt)
        self.metrics = metrics or Metrics()
        self.http: httpx.AsyncClient | None = None      # opened at startup
        self._token = token
        self.wake = wake                                # None: never wake
        self.last_traffic = 0.0
        self._poller: asyncio.Task | None = None        # running only while visitors are active
        self._fresh = asyncio.Event()                   # set after the first poll round since idle
        self._background: set[asyncio.Task] = set()
        self._tokens_since_snap = 0
        self._last_snap = time.monotonic()

    def record(self, rec: RequestRecord, arrival: float) -> None:
        rec.latency_s = time.monotonic() - arrival
        self.metrics.observe(rec)
        self._tokens_since_snap += rec.completion_tokens
        log.info("request", extra={"fields": rec.to_dict()})

    def error(self, rec: RequestRecord, arrival: float, status: int, message: str, err_type: str,
              headers: dict | None = None) -> Response:
        rec.status, rec.error, rec.finish_reason = status, message, "timeout" if status == 504 else "error"
        self.record(rec, arrival)
        return error_response(status, message, err_type, headers)

    def gone(self, rec: RequestRecord, arrival: float) -> Response:
        rec.status, rec.finish_reason = 499, "abort"
        self.record(rec, arrival)
        return Response(status_code=499)

    def _auth(self) -> dict:
        return {"authorization": f"Bearer {self._token}"} if self._token else {}

    async def traffic(self, request: HTTPRequest) -> None:
        """A visitor's request is going to be routed. Only this starts polling, so an open status tab
        can't keep replicas awake. After an idle spell what we know is stale: wait for one poll round."""
        self.last_traffic = time.monotonic()
        if self._poller is None:
            self._fresh = asyncio.Event()
            self._poller = asyncio.create_task(self._poll_loop())
        if not self._fresh.is_set():
            await _unless_gone(request, self._fresh.wait())

    async def _poll_loop(self) -> None:
        try:
            while True:
                await self.poll_round()
                self._fresh.set()
                if time.monotonic() - self.last_traffic >= self.cfg.idle_stop_s:
                    return
                await asyncio.sleep(self.cfg.poll_interval_s)
        finally:
            self._poller = None     # no await since the check above, so no request can slip in between
            self._fresh.set()

    async def poll_round(self) -> None:
        results = await asyncio.gather(*(self.poll(r) for r in self.replicas), return_exceptions=True)
        for replica, result in zip(self.replicas, results):
            if isinstance(result, Exception):
                log.error("poll failed", extra={"fields": {"replica": replica.name, "error": type(result).__name__}})
                replica.polled(Verdict.DOWN, time.monotonic())
        self._snapshot()

    async def _get(self, replica: Replica, path: str) -> httpx.Response | None:
        """None if there was no answer. The exception's text names the private URL, so it's dropped."""
        try:
            return await self.http.get(f"{replica.url}{path}", headers=self._auth(), timeout=self.cfg.poll_timeout_s)
        except httpx.TransportError:
            return None

    async def poll(self, replica: Replica) -> None:
        try:
            ready, stats = await asyncio.wait_for(asyncio.gather(self._get(replica, "/ready"), self._get(replica, "/stats")),
                                                  self.cfg.poll_timeout_s)
        except asyncio.TimeoutError:
            ready = stats = None
        before, verdict, now = replica.verdict, classify(_reply(ready)), time.monotonic()
        if verdict is Verdict.OK:
            readable = classify(_reply(stats)) is Verdict.OK
            load = _load(stats.content) if readable else None
            replica.model = (_model(stats.content) if readable else None) or replica.model
            # It said it's ready but not how full: keep our estimate rather than make it look idle.
            replica.polled(verdict, now, *(load or (replica.load, replica.n_slots)))
        else:
            replica.polled(verdict, now)
        if verdict is not before:
            level = logging.ERROR if verdict is Verdict.MISCONFIGURED else logging.INFO
            log.log(level, "replica state", extra={"fields": {"replica": replica.name, "state": verdict.value}})

    def _snapshot(self) -> None:
        """For the status page's load chart. A gap in the chart means no replica was ready."""
        now = time.monotonic()
        elapsed = max(now - self._last_snap, 1e-9)
        tokens, self._tokens_since_snap, self._last_snap = self._tokens_since_snap, 0, now
        up = [r for r in self.replicas if r.ready]
        if not up:
            return
        busy = sum(min(r.outstanding, r.n_slots) for r in up)
        queued = sum(r.outstanding for r in up) - busy
        snap = EngineSnapshot(time.time(), queued, busy, sum(r.n_slots for r in up), round(tokens / elapsed, 2),
                              0.0)      # the router holds no KV cache
        self.metrics.snapshot(snap)
        if busy or queued or tokens:
            log.info("engine", extra={"fields": {"variant": "router", **asdict(snap)}})

    def _known_model(self) -> dict | None:
        return next((r.model for r in self.replicas if r.model), None)

    def models(self) -> dict:
        model, label = self._known_model(), self.cfg.model_label
        data = []
        if model or label:
            data = [{"id": model["id"] if model else label, "object": "model", "owned_by": "tinyllm",
                     "version": model["version"] if model else label}]
        return {"object": "list", "data": data}

    def describe(self, out: dict) -> dict:
        """Adds the fields the pages read. Names and states only: never a URL or a Space id."""
        model, up, now = self._known_model() or {}, [r for r in self.replicas if r.ready], time.monotonic()
        name = model.get("id") or self.cfg.model_label or "router"
        out["variants"] = {name: {"version": model.get("version") or self.cfg.model_label,
                                  "outstanding": sum(r.load for r in up), "n_slots": sum(r.n_slots for r in up),
                                  "failed": None, "norm_backend": model.get("norm_backend")}}
        out["polling"] = self._poller is not None
        out["replicas"] = [{"name": r.name, "state": r.verdict.value if r.verdict else None, "queue": r.load,
                            "n_slots": r.n_slots,
                            "polled_s_ago": None if r.polled_at is None else round(now - r.polled_at, 1)}
                           for r in self.replicas]
        return out

    def wake_asleep(self, now: float) -> None:
        """Only a replica HF says is paused: restarting a running one would kill its requests."""
        if self.wake is None:
            return
        for r in self.replicas:
            if r.verdict is Verdict.ASLEEP and (r.woken_at is None or now - r.woken_at >= self.cfg.wake_cooldown_s):
                r.woken_at = now
                task = asyncio.create_task(self._wake(r))
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    async def _wake(self, replica: Replica) -> None:
        log.info("waking replica", extra={"fields": {"replica": replica.name}})
        try:
            await asyncio.to_thread(self.wake, replica)
        except Exception as e:      # its text may name the private Space
            log.error("wake failed", extra={"fields": {"replica": replica.name, "error": type(e).__name__}})

    async def stop(self) -> None:
        tasks = [*self._background, *([self._poller] if self._poller else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def completions(self, request: HTTPRequest) -> Response:
        arrival = time.monotonic()
        client = client_hash(request, self.cfg.trust_proxy, self.cfg.proxy_hops, self.salt)
        rec = RequestRecord(request.state.request_id, 0, variant=NO_REPLICA, client=client, arrival_unix=time.time())
        wait = self.limiter.check(client, arrival)
        if wait:
            return self.error(rec, arrival, 429, "rate limit exceeded", "rate_limit_error",
                              {"Retry-After": str(math.ceil(wait))})
        try:
            body = await _read_body(request, self.cfg.max_body_bytes)
        except _BodyTooLarge:
            return self.error(rec, arrival, 413, "request body is too large", "invalid_request_error")
        except ClientDisconnect:
            return self.gone(rec, arrival)
        rec.stream = _wants_stream(body)
        try:
            await self.traffic(request)
        except _ClientGone:
            return self.gone(rec, arrival)
        return await self._route(request, rec, arrival, body)

    async def _route(self, request: HTTPRequest, rec: RequestRecord, arrival: float, body: bytes) -> Response:
        tried: tuple[Replica, ...] = ()
        busy = False
        try:
            for _ in range(MAX_ATTEMPTS):
                replica = self.picker.pick(exclude=tried)
                if replica is None:
                    break
                tried += (replica,)
                rec.variant = replica.name
                verdict, resp, data = await self._send(request, replica, body, rec.client)
                if verdict not in RETRYABLE:
                    return await self._answer(request, rec, arrival, replica, verdict, resp, data)
                replica.refused(verdict)
                busy = busy or verdict is Verdict.BUSY
        except _ClientGone:
            return self.gone(rec, arrival)
        except _NoAnswer:
            return self.error(rec, arrival, 504, "the replica did not answer in time", "server_error")
        if busy or not any(r.ready for r in self.replicas):
            self.wake_asleep(time.monotonic())
        if not tried:
            rec.variant = NO_REPLICA
        if busy:
            return self.error(rec, arrival, 429, "server busy: queue full, retry later", "server_busy",
                              {"Retry-After": "1"})
        return self.error(rec, arrival, 503, "no replica is ready: waking up, retry shortly", "waking_up",
                          {"Retry-After": str(self.cfg.waking_retry_s)})

    def _outgoing(self, request: HTTPRequest, client: str) -> dict:
        """Built from nothing, so nothing else the visitor sent (cookies, Authorization) reaches a replica."""
        headers = {"content-type": request.headers.get("content-type", "application/json"),
                   "accept-encoding": "identity", "x-request-id": request.state.request_id, CLIENT_HEADER: client}
        return {**headers, **self._auth()}

    async def _send(self, request: HTTPRequest, replica: Replica, body: bytes,
                    client: str) -> tuple[Verdict, httpx.Response | None, bytes]:
        """Until the reply's headers (and its body, unless it's a 2xx). The exception text names the
        private URL, so only its type is logged."""
        req = self.http.build_request("POST", f"{replica.url}/v1/completions", content=body,
                                      headers=self._outgoing(request, client))
        try:
            resp = await _unless_gone(request, self.http.send(req, stream=True))
            content_type = resp.headers.get("content-type", "")
            if classify(Reply(resp.status_code, content_type)) is Verdict.OK:
                return Verdict.OK, resp, b""
            data = await _unless_gone(request, resp.aread())
        except httpx.ReadTimeout as e:
            raise _NoAnswer from e
        except httpx.TransportError as e:
            log.warning("replica unreachable", extra={"fields": {"replica": replica.name, "error": type(e).__name__}})
            return classify(Reply(None)), None, b""
        return classify(Reply(resp.status_code, content_type, data.decode("utf-8", "replace"))), resp, data

    async def _answer(self, request: HTTPRequest, rec: RequestRecord, arrival: float, replica: Replica,
                      verdict: Verdict, resp: httpx.Response, data: bytes) -> Response:
        rec.model_version = resp.headers.get("x-model-version", "")
        if rec.model_version:
            request.state.model_version = rec.model_version
        headers = {k: resp.headers[k] for k in _PASSED_HEADERS if k in resp.headers}
        headers["X-Replica"] = replica.name
        if verdict is Verdict.OK and "text/event-stream" in resp.headers.get("content-type", ""):
            return StreamingResponse(self._relay(resp, rec, arrival), status_code=resp.status_code, headers=headers)
        if verdict is Verdict.OK:
            try:
                data = await _unless_gone(request, resp.aread())
            except httpx.TransportError:
                return self.error(rec, arrival, 502, "the replica failed", "server_error")
        d = _json(data)
        rec.status = resp.status_code
        if verdict is Verdict.OK:
            # Without streaming, the visitor's first token arrives with the whole response.
            rec.ttft_s = time.monotonic() - arrival
            usage, choices = d.get("usage") or {}, d.get("choices") or [{}]
            rec.prompt_tokens = usage.get("prompt_tokens", 0)
            rec.completion_tokens = usage.get("completion_tokens", 0)
            rec.finish_reason = choices[0].get("finish_reason")
        else:
            err = d.get("error")
            rec.error = err.get("message") if isinstance(err, dict) else None
            rec.finish_reason = "timeout" if rec.status == 504 else "error"
        self.record(rec, arrival)
        return Response(data, status_code=resp.status_code, headers=headers)

    async def _relay(self, resp: httpx.Response, rec: RequestRecord, arrival: float) -> AsyncIterator[bytes]:
        """Pass a stream through. Nothing can be retried now: the visitor already has part of the story."""
        meter, finished = _StreamMeter(), False
        try:
            try:
                async for chunk in resp.aiter_bytes():
                    meter.feed(chunk, time.monotonic())
                    yield chunk
            except httpx.TransportError as e:
                log.warning("replica stream failed", extra={"fields": {"replica": rec.variant, "error": type(e).__name__}})
            if meter.done:
                rec.status = meter.error.get("code", 500) if meter.error else 200
                rec.error = meter.error.get("message") if meter.error else None
                rec.finish_reason = ("timeout" if rec.status == 504 else "error") if meter.error else meter.finish
            else:
                rec.status, rec.error, rec.finish_reason = 502, "the replica failed", "error"
                finished = True
                yield (b"\n\n" if meter.buffer else b"") + _STREAM_ERROR
            finished = True
        finally:
            if not finished:
                rec.status, rec.finish_reason = 499, "abort"
            if meter.times:
                rec.ttft_s = meter.times[0] - arrival
                rec.itl_p50_s, rec.itl_p95_s = itl_stats(meter.times)
            rec.prompt_tokens, rec.completion_tokens = meter.prompt_tokens, meter.completion_tokens
            self.record(rec, arrival)
            await resp.aclose()


def hf_wake(token: str | None) -> Callable[[Replica], None]:
    """A wake function that restarts the replica's Space on the Hub. It blocks: run it in a thread."""
    def wake(replica: Replica) -> None:
        from huggingface_hub import HfApi
        HfApi(token=token).restart_space(replica.space_id)
    return wake


def create_router_app(replicas: list[Replica], cfg: RouterConfig | None = None, token: str | None = None,
                      metrics: Metrics | None = None, wake: Callable[[Replica], None] | None = None,
                      shipper=None) -> FastAPI:
    """Only the routes listed here exist, so no other path of a replica can be reached through the router."""
    cfg = cfg or RouterConfig()
    router = ReplicaRouter(replicas, cfg, token, metrics, wake)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(connect=cfg.connect_timeout_s, read=cfg.read_timeout_s, write=10.0,
                                pool=cfg.connect_timeout_s)
        # trust_env off: a proxy set in the environment must never receive the token
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            router.http = client
            log.info("server ready", extra={"fields": {"replicas": [r.name for r in replicas], "wake": wake is not None,
                                                       "slo_ttft_s": router.metrics.slo_ttft_s, "salt": router.salt_source,
                                                       "proxy_hops": cfg.proxy_hops if cfg.trust_proxy else 0}})
            yield
            await router.stop()
        log.info("stopped")
        if shipper is not None:
            await asyncio.to_thread(shipper.close, cfg.drain_s)

    app = FastAPI(title="tinyllm router", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.router = router
    app.add_middleware(HeadersMiddleware, default_version="router")

    # All of these only report what the router already knows: they never poll, wake or count as traffic.
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        return {"ready": True}

    @app.get("/v1/models")
    async def models():
        return router.models()

    @app.get("/stats")
    async def stats():
        return router.describe(router.metrics.stats())

    history_cache: dict = {}

    @app.get("/stats/history")
    async def stats_history():
        """For the status page: computed at most once a second, however many pages poll it."""
        now = time.monotonic()
        if history_cache.get("t", -1.0) < now - 1.0:
            history_cache.update(t=now, out=router.describe(router.metrics.history()))
        return history_cache["out"]

    @app.get("/", response_class=HTMLResponse)
    async def demo_page():
        return HTMLResponse(read_page("demo.html"))

    @app.get("/status", response_class=HTMLResponse)
    async def status_page():
        return HTMLResponse(read_page("status.html"))

    @app.post("/v1/completions")
    async def completions(request: HTTPRequest):
        return await router.completions(request)

    return app


# -- command line -----------------------------------------------------------------------

REPLICA_TOKEN_ENV = "TINYLLM_REPLICA_TOKEN"     # never a flag: arguments are visible to other processes
_REPLICA_NAME = re.compile(r"[A-Za-z0-9_.-]{1,32}")
_SPACE_ID = re.compile(r"[\w.-]+/[\w.-]+")


def parse_replica(spec: str) -> Replica:
    """`name=url` or `name=url,owner/space`; the Space id is what waking needs."""
    name, sep, rest = spec.partition("=")
    url, _, space = rest.partition(",")
    url = url.rstrip("/")
    if not (sep and _REPLICA_NAME.fullmatch(name) and url.startswith(("http://", "https://"))
            and (not space or _SPACE_ID.fullmatch(space))):
        raise ValueError(f"expected name=url[,owner/space], got {spec!r}")
    return Replica(name, url, space_id=space)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="tinyllm replica router (D-031)")
    p.add_argument("--replica", action="append", required=True, type=parse_replica, metavar="NAME=URL[,OWNER/SPACE]",
                   help="one per replica; waking needs the Space id")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--rate-limit", type=float, default=20.0, help="requests/min per visitor; 0 = off")
    p.add_argument("--rate-burst", type=int, default=5)
    p.add_argument("--trust-proxy", action="store_true", help="visitor address from X-Forwarded-For")
    p.add_argument("--proxy-hops", type=int, default=1,
                   help="with --trust-proxy: trusted proxies in front (visitor = that many entries from the right)")
    p.add_argument("--slo-ttft-s", type=float, default=1.0, help="TTFT SLO threshold on /status")
    p.add_argument("--model-label", default=None, help="shown as the model until a replica has been polled")
    p.add_argument("--drain-s", type=float, default=30.0)
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    p.add_argument("--log-file", default=None, help="also write JSON lines here")
    p.add_argument("--log-dir", default=None, help="write the shipped log lines as hourly JSONL here (logship.py)")
    p.add_argument("--log-dataset", default=None, metavar="REPO_ID", help="with --log-dir: upload to this private HF dataset")
    p.add_argument("--log-replica", default=None, help="name in the log file names; default $SPACE_ID or local")
    args = p.parse_args(argv)
    if len({r.name for r in args.replica}) != len(args.replica):
        p.error("--replica names must be different")
    if args.log_dataset and not args.log_dir:
        p.error("--log-dataset needs --log-dir")
    if args.proxy_hops < 1:
        p.error("--proxy-hops must be at least 1")
    token = os.environ.get(REPLICA_TOKEN_ENV) or None
    if token and not all(r.space_id for r in args.replica):
        p.error(f"with ${REPLICA_TOKEN_ENV} set, every --replica needs its Space id, so it can be woken")

    setup_logging(args.log_level, args.log_file)
    shipper = None
    if args.log_dir:
        from tinyllm.logship import LogShipper
        shipper = LogShipper(args.log_dir, args.log_dataset, args.log_replica).attach()
        shipper.start_upload()
    cfg = RouterConfig(rate_per_min=args.rate_limit, rate_burst=args.rate_burst, trust_proxy=args.trust_proxy,
                       proxy_hops=args.proxy_hops, model_label=args.model_label, drain_s=args.drain_s)
    app = create_router_app(args.replica, cfg, token, Metrics(slo_ttft_s=args.slo_ttft_s),
                            hf_wake(token) if token else None, shipper)

    import uvicorn
    # proxy_headers off: uvicorn would believe X-Forwarded-For from loopback; client_hash does its own parsing.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False,
                proxy_headers=False, timeout_graceful_shutdown=int(args.drain_s))


if __name__ == "__main__":
    main()
