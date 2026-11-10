"""The replica router (D-031)."""

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager

import httpx
import pytest
from huggingface_hub import SpaceStage
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from test_server import greedy, model, serve, sse  # noqa: F401  (model is a fixture)

from tinyllm.logship import LogShipper
from tinyllm.obs import Metrics, hash_client
from tinyllm.router import (RETRYABLE, Picker, Replica, Reply, RouterConfig, Verdict, classify, create_router_app, hf_wake,
                            parse_replica)
from tinyllm.router import main as router_main

JSON = "application/json"
HTML = "text/html; charset=utf-8"
PAUSED = "<html>The space is paused, ask a maintainer to restart it</html>"


def test_router_does_not_import_torch():
    """The router's image has no torch, so nothing it imports may need it."""
    code = "import sys, tinyllm.router, tinyllm.logship; assert 'torch' not in sys.modules, 'torch was imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def busy(err_type):
    return '{"error": {"message": "x", "type": "%s", "code": 429}}' % err_type


@pytest.mark.parametrize("reply, expected", [
    # the rows of the reply table in docs/SERVING.md
    (Reply(200, JSON, '{"ready": true}'), Verdict.OK),
    (Reply(503, JSON, '{"ready": false, "reasons": ["not started"]}'), Verdict.NOT_READY),
    (Reply(503, HTML, PAUSED), Verdict.ASLEEP),
    (Reply(502, HTML, "<html>bad gateway</html>"), Verdict.DOWN),
    (Reply(None), Verdict.DOWN),                                     # connection error or timeout
    (Reply(404, HTML, "<html>not found</html>"), Verdict.MISCONFIGURED),
    (Reply(429, JSON, busy("server_busy")), Verdict.BUSY),
    (Reply(429, JSON, busy("rate_limit_error")), Verdict.PASS),
    # around the table
    (Reply(200, "text/event-stream; charset=utf-8"), Verdict.OK),
    (Reply(503, "TEXT/HTML", PAUSED.upper()), Verdict.ASLEEP),       # case doesn't matter
    (Reply(503, HTML, "<html>something else</html>"), Verdict.DOWN), # not known to be asleep: never wake it
    (Reply(200, HTML, "<html>a landing page</html>"), Verdict.DOWN), # 200, but not from our server
    (Reply(500, "text/plain", "oops"), Verdict.DOWN),
    (Reply(401, JSON, '{"error": "bad token"}'), Verdict.MISCONFIGURED),
    (Reply(403, HTML, "<html>no</html>"), Verdict.MISCONFIGURED),
    (Reply(502, JSON, "{}"), Verdict.DOWN),
    (Reply(302, JSON, "{}"), Verdict.DOWN),                          # a redirect is never followed
    # our server's own answers go to the visitor
    (Reply(400, JSON, "{}"), Verdict.PASS),
    (Reply(500, JSON, "{}"), Verdict.PASS),
    (Reply(504, JSON, "{}"), Verdict.PASS),
    (Reply(429, JSON, "not json"), Verdict.PASS),
    (Reply(429, JSON, "[1, 2]"), Verdict.PASS),
    (Reply(404, JSON, "{}"), Verdict.PASS),
])
def test_classify(reply, expected):
    assert classify(reply) is expected


def test_a_request_is_retried_only_if_it_never_reached_a_working_model():
    assert Verdict.OK not in RETRYABLE and Verdict.PASS not in RETRYABLE
    assert RETRYABLE == set(Verdict) - {Verdict.OK, Verdict.PASS}


def replicas(*outstanding):
    """Ready replicas r1, r2, ... with these queue lengths."""
    out = [Replica(f"r{i + 1}", f"http://r{i + 1}") for i in range(len(outstanding))]
    for r, n in zip(out, outstanding):
        r.polled(Verdict.OK, 0.0, outstanding=n, n_slots=8)
    return out


def names(picker, n):
    return [picker.pick().name for _ in range(n)]


def test_pick_prefers_the_shortest_queue():
    assert Picker(replicas(5, 2)).pick().name == "r2"


def test_pick_counts_what_it_sent_since_the_last_poll():
    p = Picker(replicas(3, 0))
    assert names(p, 3) == ["r2", "r2", "r2"]          # r2's load is now 3, level with r1
    assert p.replicas[1].load == 3
    assert names(p, 2) == ["r1", "r2"]                # level: in turn after r2


def test_ties_go_round_robin():
    assert names(Picker(replicas(1, 1)), 4) == ["r1", "r2", "r1", "r2"]


def test_a_poll_resets_the_count_of_sent_requests():
    p = Picker(replicas(0, 0))
    names(p, 2)
    r = p.replicas[0]
    r.polled(Verdict.OK, 5.0, outstanding=4, n_slots=8)
    assert (r.load, r.polled_at) == (4, 5.0)


@pytest.mark.parametrize("verdict", [None, *(v for v in Verdict if v is not Verdict.OK)])
def test_only_a_ready_replica_is_picked(verdict):
    p = Picker(replicas(9, 0))
    p.replicas[1].verdict = verdict
    assert names(p, 2) == ["r1", "r1"]


def test_none_ready_picks_nothing():
    p = Picker([Replica("r1", "http://r1"), Replica("r2", "http://r2")])
    assert p.pick() is None


def test_a_retry_goes_to_a_different_replica():
    p = Picker(replicas(0, 5))
    first = p.pick()
    assert first.name == "r1" and p.pick(exclude=(first,)).name == "r2"


def test_a_retry_with_nowhere_to_go_picks_nothing():
    p = Picker(replicas(0))
    assert p.pick(exclude=(p.pick(),)) is None


# -- the request path, against fake replicas on real sockets --------------------------------

BODY = {"prompt": "Once", "max_tokens": 2}
TOKEN = "secret-router-token"
CLOSED_PORT = "http://127.0.0.1:1"


@contextmanager
def live(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical", access_log=False,
                                           lifespan="on", timeout_graceful_shutdown=1,
                                           proxy_headers=False))   # uvicorn would otherwise believe loopback's XFF
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    end = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < end and thread.is_alive(), "server did not start"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "server did not stop"


def event(text, finish=None, **extra):
    d = {"id": "x", "object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": finish}], **extra}
    return f"data: {json.dumps(d)}\n\n"


STORY = [event("Once"), event(" upon", "length", usage={"prompt_tokens": 1, "completion_tokens": 2}), "data: [DONE]\n\n"]
OK_BODY = json.dumps({"choices": [{"text": "hi", "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 3}})


def answer(status=200, body=OK_BODY, ctype=JSON, headers=None):
    async def reply(request):
        return Response(body, status, headers={"content-type": ctype, **(headers or {})})
    return reply


def stream(parts=STORY, die=False):
    async def reply(request):
        async def gen():
            for part in parts:
                yield part
            if die:
                raise RuntimeError("the replica crashed")
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"x-model-version": "v9"})
    return reply


async def hang(request):
    """Holds a non-streaming request, as the server does while generating, and notices the visitor leaving."""
    for _ in range(300):
        if await request.is_disconnected():
            request.app.state.fake.gone.set()
            return Response(status_code=499)
        await asyncio.sleep(0.02)
    return Response(status_code=500)


async def hang_streaming(request):
    fake = request.app.state.fake

    async def gen():
        try:
            yield STORY[0]
            await asyncio.sleep(6)
        finally:
            fake.gone.set()
    return StreamingResponse(gen(), media_type="text/event-stream")


def load(outstanding, n_slots):
    return answer(body=json.dumps({"variants": {"main": {"version": "v9", "outstanding": outstanding, "n_slots": n_slots,
                                                         "norm_backend": "torch"}}}))


class Fake:
    """A replica whose reply a test chooses. Its calls are recorded."""

    def __init__(self, reply=None, ready=None, stats=None):
        self.reply = reply or answer(headers={"x-model-version": "v9"})
        self.ready = ready or answer(body='{"ready": true}')
        self.stats = stats or load(0, 8)
        self.calls: list[dict] = []
        self.polls: list[tuple] = []
        self.gone = threading.Event()
        self.app = FastAPI()
        self.app.state.fake = self

        async def completions(request: Request):
            self.calls.append({"headers": dict(request.headers), "body": await request.body()})
            return await self.reply(request)
        self.app.post("/v1/completions")(completions)

        async def ready_route(request: Request):
            self.polls.append(("/ready", request.headers.get("authorization")))
            return await self.ready(request)

        async def stats_route(request: Request):
            self.polls.append(("/stats", request.headers.get("authorization")))
            return await self.stats(request)
        self.app.get("/ready")(ready_route)
        self.app.get("/stats")(stats_route)

        async def other(request: Request):
            self.calls.append({"path": request.url.path})
            return {"secret": "from the replica"}
        self.app.get("/{path:path}")(other)


class Spy(Metrics):
    def __init__(self):
        super().__init__()
        self.records = []

    def observe(self, rec, now=None):
        self.records.append(rec)
        super().observe(rec, now)


class World:
    def __init__(self, client, fakes, replicas, metrics, router):
        self.client, self.fakes, self.replicas, self.metrics, self.router = client, fakes, replicas, metrics, router

    @property
    def calls(self):
        return [len(f.calls) for f in self.fakes if f]

    def post(self, headers=None, **kw):
        return self.client.post("/v1/completions", json=kw or BODY, headers=headers)

    def record(self, timeout=5.0):
        end = time.monotonic() + timeout
        while not self.metrics.records:
            assert time.monotonic() < end, "no request was recorded"
            time.sleep(0.01)
        return self.metrics.records[-1]


@contextmanager
def world(*fakes, ready=True, poll=False, wake=None, **cfg):
    """A router in front of these fakes (None = a closed port), marked ready unless `ready` is off.
    Without `poll` that state stays as set here, so a test can reach the case where a poll was out of date."""
    cfg = RouterConfig(**{"salt": "s", "rate_per_min": 100000, "rate_burst": 1000, **cfg})

    @contextmanager
    def up(remaining, urls):
        if not remaining:
            yield urls
        elif remaining[0] is None:
            with up(remaining[1:], [*urls, CLOSED_PORT]) as out:
                yield out
        else:
            with live(remaining[0].app) as url, up(remaining[1:], [*urls, url]) as out:
                yield out

    with up(list(fakes), []) as urls:
        reps = [Replica(f"r{i + 1}", url) for i, url in enumerate(urls)]
        if ready:
            for r in reps:
                r.polled(Verdict.OK, 0.0, n_slots=8)
        metrics = Spy()
        app = create_router_app(reps, cfg, TOKEN, metrics, wake)
        if not poll:
            async def no_polling(request):
                pass
            app.state.router.traffic = no_polling
        with live(app) as url, httpx.Client(base_url=url, timeout=15) as client:
            yield World(client, fakes, reps, metrics, app.state.router)


def test_a_reply_is_passed_through_with_the_replica_named():
    with world(Fake()) as w:
        r = w.post()
        assert (r.status_code, r.text) == (200, OK_BODY)
        assert r.headers["x-replica"] == "r1" and r.headers["x-model-version"] == "v9" and r.headers["x-request-id"]
        rec = w.record()
        assert (rec.status, rec.variant, rec.completion_tokens, rec.finish_reason) == (200, "r1", 3, "stop")
        assert rec.ttft_s == pytest.approx(rec.latency_s, abs=0.05)   # not streamed: the first token comes with the reply


def test_a_stream_is_passed_through_unchanged_and_measured():
    with world(Fake(stream())) as w:
        with w.client.stream("POST", "/v1/completions", json={**BODY, "stream": True}) as r:
            text = "".join(r.iter_text())
            assert r.headers["content-type"].startswith("text/event-stream") and r.headers["x-replica"] == "r1"
            assert r.headers["x-model-version"] == "v9"
        assert text == "".join(STORY)
        rec = w.record()
        assert (rec.status, rec.stream, rec.prompt_tokens, rec.completion_tokens, rec.finish_reason) == (200, True, 1, 2, "length")
        assert rec.ttft_s is not None


BUSY = answer(429, busy("server_busy"), headers={"retry-after": "1"})
NOT_READY = answer(503, '{"ready": false}')
ASLEEP = answer(503, PAUSED, HTML)
BAD_GATEWAY = answer(502, "<html>502</html>", HTML)
NOT_FOUND = answer(404, "<html>404</html>", HTML)


@pytest.mark.parametrize("first, stays_ready, state", [
    (BUSY, True, Verdict.OK), (NOT_READY, False, Verdict.NOT_READY), (ASLEEP, False, Verdict.ASLEEP),
    (BAD_GATEWAY, False, Verdict.DOWN), (NOT_FOUND, False, Verdict.MISCONFIGURED),
])
def test_a_refused_request_is_retried_once_elsewhere(first, stays_ready, state):
    with world(Fake(first), Fake()) as w:
        r = w.post()
        assert (r.status_code, r.headers["x-replica"], w.calls) == (200, "r2", [1, 1])
        assert w.replicas[0].ready is stays_ready and w.replicas[0].verdict is state
        assert w.replicas[0].sent_since_poll == 0            # refused, so it doesn't count as load


def test_an_unreachable_replica_is_retried_elsewhere():
    with world(None, Fake()) as w:
        r = w.post()
        assert (r.status_code, r.headers["x-replica"]) == (200, "r2")
        assert w.replicas[0].verdict is Verdict.DOWN


@pytest.mark.parametrize("reply, status", [
    (answer(400, '{"error": {"message": "bad", "type": "invalid_request_error"}}'), 400),
    (answer(504, '{"error": {"message": "slow", "type": "server_error"}}'), 504),
    (answer(500, '{"error": {"message": "boom", "type": "server_error"}}'), 500),
    (answer(429, busy("rate_limit_error"), headers={"retry-after": "7"}), 429),
])
def test_an_answer_from_our_server_goes_to_the_visitor_without_a_retry(reply, status):
    with world(Fake(reply), Fake()) as w:
        r = w.post()
        assert (r.status_code, w.calls, r.headers["x-replica"]) == (status, [1, 0], "r1")
        assert w.replicas[0].ready
        if status == 429:
            assert r.headers["retry-after"] == "7"


def test_every_replica_busy_is_429_after_exactly_two_tries():
    with world(Fake(BUSY), Fake(BUSY)) as w:
        r = w.post()
        assert (r.status_code, r.json()["error"]["type"], r.headers["retry-after"], w.calls) == (429, "server_busy", "1", [1, 1])
        assert "x-replica" not in r.headers


def test_at_most_two_tries_with_three_replicas():
    with world(Fake(NOT_READY), Fake(NOT_READY), Fake()) as w:
        assert w.post().status_code == 503 and sum(w.calls) == 2


def test_no_replica_ready_is_503_with_retry_after_and_touches_nothing():
    with world(Fake(), Fake(), ready=False) as w:
        r = w.post()
        assert (r.status_code, r.json()["error"]["type"], r.headers["retry-after"], w.calls) == (503, "waking_up", "30", [0, 0])
        rec = w.record()
        assert (rec.status, rec.variant) == (503, "none")


def test_every_replica_down_is_503():
    with world(None, None) as w:
        assert w.post().status_code == 503
        assert all(r.verdict is Verdict.DOWN for r in w.replicas)


def test_a_failure_mid_stream_ends_the_stream_and_is_never_retried():
    with world(Fake(stream(STORY[:1], die=True)), Fake(stream())) as w:
        with w.client.stream("POST", "/v1/completions", json={**BODY, "stream": True}) as r:
            events = sse(r)
        assert events[0]["choices"][0]["text"] == "Once" and events[1]["error"]["code"] == 502 and events[2] == "[DONE]"
        assert len(events) == 3 and w.calls == [1, 0]
        rec = w.record()
        assert (rec.status, rec.error, rec.finish_reason) == (502, "the replica failed", "error")


def test_a_replica_error_event_is_recorded():
    err = 'data: {"error": {"message": "too slow", "type": "server_error", "code": 504}}\n\ndata: [DONE]\n\n'
    with world(Fake(stream([STORY[0], err]))) as w:
        with w.client.stream("POST", "/v1/completions", json={**BODY, "stream": True}) as r:
            assert sse(r)[-1] == "[DONE]"
        rec = w.record()
        assert (rec.status, rec.error, rec.finish_reason) == (504, "too slow", "timeout")


def test_a_silent_replica_is_504_and_keeps_the_request():
    with world(Fake(hang), Fake(), read_timeout_s=0.3) as w:
        r = w.post()
        assert (r.status_code, w.calls) == (504, [1, 0])
        assert w.record().variant == "r1"


def test_a_visitor_leaving_a_non_streaming_request_reaches_the_replica():
    fake = Fake(hang)
    with world(fake) as w:
        with pytest.raises(httpx.TimeoutException):
            w.client.post("/v1/completions", json=BODY, timeout=httpx.Timeout(0.5))
        assert fake.gone.wait(5), "the replica's request stayed open"
        rec = w.record()
        assert (rec.status, rec.finish_reason) == (499, "abort")


def test_a_visitor_leaving_a_stream_reaches_the_replica():
    fake = Fake(hang_streaming)
    with world(fake) as w:
        with w.client.stream("POST", "/v1/completions", json={**BODY, "stream": True}) as r:
            next(r.iter_lines())
        assert fake.gone.wait(5), "the replica's stream stayed open"
        rec = w.record()
        assert (rec.status, rec.finish_reason) == (499, "abort")


def test_what_reaches_a_replica_is_only_what_it_needs():
    fake = Fake()
    with world(fake) as w:
        w.post(headers={"authorization": "Bearer visitor", "cookie": "a=b", "x-request-id": "abc-1",
                        "x-forwarded-for": "9.9.9.9", "x-tinyllm-client": "f" * 16})
        call = fake.calls[0]
        assert json.loads(call["body"]) == BODY
        assert call["headers"]["authorization"] == f"Bearer {TOKEN}"
        assert call["headers"]["x-request-id"] == "abc-1"
        assert call["headers"]["x-tinyllm-client"] == hash_client("127.0.0.1", "s")      # the router's own hash
        assert "cookie" not in call["headers"] and "x-forwarded-for" not in call["headers"]


def test_a_visitor_sees_only_the_replica_headers_it_should():
    secret = {"set-cookie": "spaces-jwt=abc", "x-private": "1", "location": "http://elsewhere"}
    with world(Fake(answer(headers=secret))) as w:
        r = w.post()
        assert not set(secret) & set(r.headers)


def test_a_redirect_is_not_followed():
    other = Fake()
    with live(other.app) as url:
        fake = Fake(answer(302, "{}", headers={"location": f"{url}/v1/completions"}))
        with world(fake) as w:
            assert w.post().status_code == 503 and other.calls == []


@pytest.mark.parametrize("fakes", [(None, None), (Fake(BAD_GATEWAY), Fake(NOT_FOUND))], ids=["unreachable", "error pages"])
def test_token_and_private_urls_never_reach_a_visitor(fakes):
    with world(*fakes) as w:
        r = w.post()
        seen = r.text + str(r.headers)
        assert r.status_code == 503 and TOKEN not in seen
        assert all(rep.url not in seen and rep.url.split("//")[1] not in seen for rep in w.replicas)
        assert "<html" not in seen


@pytest.mark.parametrize("path", ["/debug/state", "/metrics", "/docs", "/openapi.json", "/anything"])
def test_only_listed_routes_exist(path):
    with world(Fake()) as w:
        assert w.client.get(path).status_code == 404 and w.fakes[0].calls == []


def test_a_large_body_is_refused():
    with world(Fake(), max_body_bytes=100) as w:
        assert w.post(prompt="x" * 200, max_tokens=1).status_code == 413
        chunked = w.client.post("/v1/completions", content=(b"x" * 80 for _ in range(3)))
        assert chunked.status_code == 413 and w.calls == [0]


def test_the_rate_limit_is_at_the_router():
    with world(Fake(), rate_per_min=60, rate_burst=2) as w:
        codes = [w.post() for _ in range(3)]
        assert [r.status_code for r in codes] == [200, 200, 429]
        assert codes[2].json()["error"]["type"] == "rate_limit_error" and codes[2].headers["retry-after"] == "1"
        assert w.calls == [2]


def test_a_spoofed_forwarded_for_cannot_dodge_the_router_limit():
    fake = Fake()
    with world(fake, rate_per_min=60, rate_burst=2, trust_proxy=True) as w:
        codes = [w.post(headers={"x-forwarded-for": f"6.6.6.{i}, 1.1.1.1"}).status_code for i in range(3)]
        assert codes == [200, 200, 429]
        assert fake.calls[0]["headers"]["x-tinyllm-client"] == hash_client("1.1.1.1", "s")
        assert w.post(headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200


def test_requests_spread_over_two_replicas():
    with world(Fake(), Fake()) as w:
        assert [w.post().headers["x-replica"] for _ in range(4)] == ["r1", "r2", "r1", "r2"]


def test_through_the_router_equals_direct_on_a_real_server(model):
    def ids(events):
        return [t for e in events if isinstance(e, dict) for t in e["choices"][0]["token_ids"]]

    with serve(model, trust_router=True) as s:
        with s.client.stream("POST", "/v1/completions", json=greedy(stream=True)) as r:
            direct = sse(r)
        replica = Replica("r1", s.url)
        replica.polled(Verdict.OK, 0.0, n_slots=3)
        metrics = Spy()
        cfg = RouterConfig(salt="s")
        with live(create_router_app([replica], cfg, TOKEN, metrics)) as url, httpx.Client(base_url=url, timeout=30) as c:
            with c.stream("POST", "/v1/completions", json=greedy(stream=True)) as r:
                routed = sse(r)
                assert r.headers["x-model-version"] == "test-v1" and r.headers["x-replica"] == "r1"
            assert ids(routed) == ids(direct) and len(ids(direct)) == 10 and routed[-1] == "[DONE]"
            assert (replica.ready, replica.n_slots) == (True, 3)      # polled the real /ready and /stats
            plain = c.post("/v1/completions", json=greedy())
            assert plain.json()["choices"][0]["token_ids"] == ids(direct)
        end = time.monotonic() + 5
        while len(metrics.records) < 2:
            assert time.monotonic() < end
            time.sleep(0.01)
        assert [(r.status, r.completion_tokens) for r in metrics.records] == [(200, 10), (200, 10)]


# -- polling and waking ---------------------------------------------------------------------

def until(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.01)


class Waker:
    """Stands in for restarting a Space on the Hub."""

    def __init__(self, hold=False, fail=False):
        self.woken: list[str] = []
        self.fail = fail
        self.release = threading.Event()
        if not hold:
            self.release.set()

    def __call__(self, replica):
        self.woken.append(replica.name)
        assert self.release.wait(10)
        if self.fail:
            raise RuntimeError("the Hub said no: http://private-url")


async def slow(request):
    await asyncio.sleep(1.5)
    return Response('{"ready": true}', media_type=JSON)


def spend_the_burst(w):
    assert w.router.limiter.check(hash_client("127.0.0.1", "s"), time.monotonic()) == 0


def test_nothing_is_polled_until_a_visitor_comes_then_the_request_waits_for_the_first_poll():
    fake = Fake()
    with world(fake, ready=False, poll=True) as w:
        time.sleep(0.1)
        assert fake.polls == []
        r = w.post()
        assert (r.status_code, r.headers["x-replica"]) == (200, "r1") and len(fake.calls) == 1


def test_a_poll_reads_the_queue_and_carries_the_token():
    fake = Fake(stats=load(3, 8))
    with world(fake, ready=False, poll=True) as w:
        w.post()
        r = w.replicas[0]
        assert (r.outstanding, r.n_slots, r.sent_since_poll, r.ready) == (3, 8, 1, True)
        assert sorted(fake.polls) == [("/ready", f"Bearer {TOKEN}"), ("/stats", f"Bearer {TOKEN}")]


def test_polling_stops_after_the_last_visitor_and_starts_again_with_the_next():
    fake = Fake()
    with world(fake, poll=True, poll_interval_s=0.02, idle_stop_s=0.4) as w:
        w.post()
        until(lambda: len(fake.polls) >= 6)
        until(lambda: w.router._poller is None)
        n = len(fake.polls)
        time.sleep(0.15)
        assert len(fake.polls) == n
        w.post()
        assert len(fake.polls) > n and w.router._poller is not None


def test_polling_continues_while_visitors_keep_coming():
    with world(Fake(), poll=True, poll_interval_s=0.02, idle_stop_s=0.5) as w:
        for _ in range(8):
            w.post()
            time.sleep(0.1)                 # 0.8 s in total, longer than idle_stop_s
            assert w.router._poller is not None


def test_a_refused_request_is_not_traffic():
    fake = Fake()
    with world(fake, poll=True, rate_per_min=60, rate_burst=1, max_body_bytes=100) as w:
        assert w.post(prompt="x" * 200, max_tokens=1).status_code == 413    # too large; this used the one token
        assert w.post().status_code == 429                                  # over the limit
        time.sleep(0.05)
        assert fake.polls == [] and w.router._poller is None


@pytest.mark.parametrize("ready, verdict, wakes", [
    (lambda: NOT_READY, Verdict.NOT_READY, False),
    (lambda: ASLEEP, Verdict.ASLEEP, True),
    (lambda: BAD_GATEWAY, Verdict.DOWN, False),
    (lambda: NOT_FOUND, Verdict.MISCONFIGURED, False),
    (lambda: answer(401, '{"error": "bad token"}'), Verdict.MISCONFIGURED, False),
    (lambda: answer(302, "{}", headers={"location": "http://elsewhere"}), Verdict.DOWN, False),
    (lambda: answer(200, "<html>a landing page</html>", HTML), Verdict.DOWN, False),
], ids=["starting", "paused", "bad-gateway", "wrong-url", "bad-token", "redirect", "landing-page"])
def test_what_a_poll_found_decides_eligibility_and_waking(ready, verdict, wakes):
    waker, fake = Waker(), Fake(ready=ready())
    with world(fake, ready=False, poll=True, wake=waker) as w:
        r = w.post()
        assert (r.status_code, r.json()["error"]["type"], len(fake.calls)) == (503, "waking_up", 0)
        assert w.replicas[0].verdict is verdict
        if wakes:
            until(lambda: waker.woken)
        else:
            time.sleep(0.1)
        assert waker.woken == (["r1"] if wakes else [])


def test_a_poll_with_no_answer_means_down():
    with world(None, ready=False, poll=True) as w:
        assert w.post().status_code == 503 and w.replicas[0].verdict is Verdict.DOWN


def test_a_hanging_replica_delays_a_visitor_only_by_the_poll_timeout():
    with world(Fake(ready=slow), ready=False, poll=True, poll_timeout_s=0.1) as w:
        start = time.monotonic()
        assert w.post().status_code == 503
        assert time.monotonic() - start < 1.0 and w.replicas[0].verdict is Verdict.DOWN


def test_ready_but_stats_unreadable_keeps_the_replica_and_its_load():
    with world(Fake(stats=answer(500, '{"oops": 1}')), poll=True) as w:
        r = w.replicas[0]
        r.polled(Verdict.OK, 0.0, outstanding=4, n_slots=8)
        r.sent_since_poll = 1
        assert w.post().status_code == 200
        assert (r.ready, r.outstanding, r.n_slots, r.sent_since_poll) == (True, 5, 8, 1)


def test_a_replica_coming_back_is_seen_by_the_next_poll():
    fake = Fake(ready=NOT_READY)
    with world(fake, ready=False, poll=True, poll_interval_s=0.02) as w:
        assert w.post().status_code == 503
        fake.ready = answer(body='{"ready": true}')
        until(lambda: w.replicas[0].ready)
        assert w.post().status_code == 200


def test_a_visitor_leaving_during_the_first_poll_is_recorded():
    fake = Fake(ready=slow)
    with world(fake, ready=False, poll=True, poll_timeout_s=3.0) as w:
        with pytest.raises(httpx.TimeoutException):
            w.client.post("/v1/completions", json=BODY, timeout=httpx.Timeout(0.3))
        assert w.record().status == 499 and fake.calls == []


def test_no_replica_ready_answers_at_once_and_wakes_in_the_background():
    waker = Waker(hold=True)
    with world(Fake(ready=ASLEEP), Fake(ready=ASLEEP), ready=False, poll=True, wake=waker) as w:
        r = w.post()
        assert (r.status_code, r.json()["error"]["type"], r.headers["retry-after"]) == (503, "waking_up", "30")
        until(lambda: sorted(waker.woken) == ["r1", "r2"])
        assert not waker.release.is_set()   # the reply above came while the wake was still running
        assert w.post().status_code == 503
        time.sleep(0.05)
        assert len(waker.woken) == 2        # at most one wake per replica per cooldown
        waker.release.set()


def test_a_replica_can_be_woken_again_after_the_cooldown():
    waker = Waker()
    with world(Fake(ready=ASLEEP), ready=False, poll=True, wake=waker, wake_cooldown_s=0.5) as w:
        w.post()
        until(lambda: len(waker.woken) == 1)
        w.post()
        time.sleep(0.05)
        assert len(waker.woken) == 1
        time.sleep(0.6)
        w.post()
        until(lambda: len(waker.woken) == 2)


def test_shedding_wakes_only_the_sleeping_replica():
    waker = Waker()
    with world(Fake(BUSY), Fake(ready=ASLEEP), ready=False, poll=True, wake=waker) as w:
        r = w.post()
        assert (r.status_code, r.json()["error"]["type"], w.calls) == (429, "server_busy", [1, 0])
        until(lambda: waker.woken)
        assert waker.woken == ["r2"]


def test_a_replica_that_turns_out_to_be_paused_is_woken():
    waker = Waker()
    with world(Fake(ASLEEP), wake=waker) as w:          # the last poll said ready; the request says paused
        assert w.post().status_code == 503
        until(lambda: waker.woken)


def test_no_wake_while_a_replica_is_serving():
    waker = Waker()
    with world(Fake(), Fake(ready=ASLEEP), ready=False, poll=True, wake=waker) as w:
        for _ in range(3):
            assert w.post().status_code == 200
        time.sleep(0.1)
        assert waker.woken == []


def test_polling_alone_never_wakes():
    waker = Waker()
    with world(Fake(ready=ASLEEP), ready=False, poll=True, wake=waker, poll_interval_s=0.02, wake_cooldown_s=0.0) as w:
        fake = w.fakes[0]
        w.post()
        n = len(fake.polls)
        until(lambda: len(fake.polls) >= n + 8)
        assert len(waker.woken) == 1


def test_a_rate_limited_request_does_not_wake():
    waker = Waker()
    with world(Fake(ready=ASLEEP), ready=False, poll=True, wake=waker, rate_per_min=60, rate_burst=1) as w:
        spend_the_burst(w)
        assert w.post().status_code == 429
        time.sleep(0.1)
        assert waker.woken == []


def test_a_failed_wake_is_contained():
    waker = Waker(fail=True)
    with world(Fake(ready=ASLEEP), ready=False, poll=True, wake=waker) as w:
        first = w.post()
        until(lambda: waker.woken)
        time.sleep(0.05)
        second = w.post()
        assert (first.status_code, second.status_code, len(waker.woken)) == (503, 503, 1)
        assert "private-url" not in first.text + second.text


@pytest.mark.parametrize("fake, cfg", [
    (None, {}),                                         # nothing at the address at all
    (lambda: Fake(ready=slow), {"poll_timeout_s": 0.1}),  # a sleeping Space: holds the poll while it wakes
], ids=["no-connection", "held-past-the-poll-timeout"])
def test_a_silent_replica_is_woken_once_per_cooldown(fake, cfg):
    waker = Waker()
    with world(fake and fake(), ready=False, poll=True, wake=waker, **cfg) as w:
        r = w.post()
        assert (r.status_code, r.json()["error"]["type"]) == (503, "waking_up")
        assert (w.replicas[0].verdict, w.replicas[0].silent) == (Verdict.DOWN, True)
        until(lambda: waker.woken)
        assert w.post().status_code == 503
        time.sleep(0.05)
        assert waker.woken == ["r1"]


def test_a_silent_replica_that_answers_again_is_no_longer_silent():
    fake = Fake(ready=slow)
    with world(fake, ready=False, poll=True, poll_timeout_s=0.1, poll_interval_s=0.02) as w:
        assert w.post().status_code == 503 and w.replicas[0].silent
        fake.ready = BAD_GATEWAY
        until(lambda: not w.replicas[0].silent)
        assert w.replicas[0].verdict is Verdict.DOWN


class Hub:
    """Stands in for HfApi: one Space at the given stage."""

    def __init__(self, stage):
        self.stage, self.seen = stage, []

    def __call__(self, token=None):
        self.seen.append(token)
        return self

    def get_space_runtime(self, repo_id, **kw):
        self.seen.append(("runtime", repo_id))
        return types.SimpleNamespace(stage=self.stage)

    def restart_space(self, repo_id, **kw):
        self.seen.append(("restart", repo_id, kw))


@pytest.mark.parametrize("stage, restarts", [
    ("SLEEPING", True), ("PAUSED", True), (SpaceStage.PAUSED, True),     # the enum has no SLEEPING: compare text
    ("RUNNING", False), ("APP_STARTING", False), ("RUNNING_APP_STARTING", False), ("BUILDING", False),
    ("RUNTIME_ERROR", False), ("STOPPED", False), (SpaceStage.RUNNING, False),
])
def test_hf_wake_restarts_only_a_space_the_hub_says_is_asleep(monkeypatch, stage, restarts):
    hub = Hub(stage)
    monkeypatch.setattr("huggingface_hub.HfApi", hub)
    seen_stage = hf_wake("t")(Replica("r1", "http://x", space_id="me/r1"))
    restart = [("restart", "me/r1", {})] if restarts else []     # no factory reboot: the image is kept
    assert hub.seen == ["t", ("runtime", "me/r1"), *restart]
    assert seen_stage == getattr(stage, "value", stage)


# -- pages, metrics and the command line ----------------------------------------------------

PASSIVE = ["/", "/status", "/stats", "/stats/history", "/v1/models", "/health", "/ready"]


@pytest.mark.parametrize("path, needle", [("/", "v1/completions"), ("/status", "replicas-panel")])
def test_the_pages_are_served(path, needle):
    with world(Fake(), ready=False) as w:
        r = w.client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and needle in r.text


def test_pages_and_stats_never_poll_or_wake():
    fake, waker = Fake(ready=ASLEEP), Waker()
    with world(fake, ready=False, poll=True, wake=waker) as w:
        for path in PASSIVE:
            assert w.client.get(path).status_code == 200
        time.sleep(0.05)
        assert (fake.polls, fake.calls, waker.woken, w.router._poller, w.router.last_traffic) == ([], [], [], None, 0.0)


def test_an_open_status_tab_does_not_keep_polling_alive():
    with world(Fake(), poll=True, poll_interval_s=0.02, idle_stop_s=0.4) as w:
        w.post()
        until(lambda: w.client.get("/stats/history").status_code == 200 and w.router._poller is None)


def test_health_and_ready_describe_the_router_itself():
    with world(None, None, ready=False) as w:
        assert w.client.get("/health").json() == {"status": "ok"} and w.client.get("/ready").json() == {"ready": True}


def test_models_before_any_poll_and_after():
    with world(Fake(), ready=False, poll=True, model_label="tiny-v1") as w:
        first = w.client.get("/v1/models").json()["data"]
        assert [(m["id"], m["version"]) for m in first] == [("tiny-v1", "tiny-v1")]
        w.post()
        after = w.client.get("/v1/models").json()["data"]
        assert [(m["id"], m["version"]) for m in after] == [("main", "v9")]


def test_models_with_nothing_known_is_empty():
    with world(Fake(), ready=False) as w:
        assert w.client.get("/v1/models").json()["data"] == []


def test_stats_describe_the_replicas_as_one_service():
    with world(Fake(stats=load(1, 8)), Fake(stats=load(3, 8)), ready=False, poll=True) as w:
        assert w.post().status_code == 200
        w.record()
        st = w.client.get("/stats").json()
        assert st["windows"]["5m"]["requests"] == 1
        assert st["variants"] == {"main": {"version": "v9", "outstanding": 5, "n_slots": 16, "failed": None,
                                           "norm_backend": "torch"}}


def test_the_status_page_gets_each_replica_state():
    with world(Fake(), Fake(ready=NOT_FOUND), ready=False, poll=True, poll_interval_s=0.02, idle_stop_s=0.3) as w:
        before = w.client.get("/stats").json()
        assert before["polling"] is False
        assert [(r["name"], r["state"], r["polled_s_ago"]) for r in before["replicas"]] == [("r1", None, None), ("r2", None, None)]
        w.post()
        after = w.client.get("/stats").json()
        assert after["polling"] is True and [r["state"] for r in after["replicas"]] == ["ok", "misconfigured"]
        assert after["replicas"][0]["n_slots"] == 8 and after["replicas"][0]["polled_s_ago"] is not None
        until(lambda: not w.client.get("/stats").json()["polling"])
        assert [r["state"] for r in w.client.get("/stats").json()["replicas"]] == ["ok", "misconfigured"]


@pytest.mark.parametrize("path", PASSIVE)
def test_public_pages_show_no_private_url_or_token(path):
    with world(Fake(), ready=False, poll=True) as w:
        replica = w.replicas[0]
        replica.space_id = "me/private-space"
        w.post()
        seen = w.client.get(path).text
        assert TOKEN not in seen and replica.url not in seen and "private-space" not in seen


def test_a_waking_503_counts_against_availability():
    with world(Fake(), ready=False) as w:
        assert w.post().status_code == 503
        w.record()
        slo = w.client.get("/stats/history").json()["slo"]["availability"]
        assert (slo["total"], slo["good"]) == (1, 0)


def test_the_load_chart_is_fed_from_polls():
    with world(Fake(stats=load(10, 8)), Fake(stats=load(2, 8)), ready=False, poll=True) as w:
        w.post()
        snap = w.client.get("/stats/history").json()["snapshots"][-1]
        assert (snap["busy_slots"], snap["queue_len"]) == (10, 2) and w.metrics.last_snapshot.n_slots == 16


def test_with_no_replica_ready_the_load_chart_has_a_gap():
    with world(Fake(ready=NOT_READY), ready=False, poll=True) as w:
        w.post()
        assert w.client.get("/stats/history").json()["snapshots"] == []


def test_the_shipped_log_has_no_prompt_address_or_token(tmp_path):
    lg = logging.getLogger("tinyllm")
    level = lg.level
    lg.setLevel(logging.INFO)
    shipper = LogShipper(tmp_path, None, replica="router").attach()
    try:
        with world(Fake(), Fake(ready=NOT_FOUND), ready=False, poll=True) as w:
            assert w.post(prompt="ZQXJ", max_tokens=1).status_code == 200
            w.record()
    finally:
        lg.removeHandler(shipper.handler)
        lg.setLevel(level)
        shipper.close()
    text = "".join(f.read_text(encoding="utf-8") for f in tmp_path.rglob("*.jsonl"))
    assert '"request"' in text and "server ready" in text and "stopped" in text
    assert "ZQXJ" not in text and TOKEN not in text and "127.0.0.1" not in text


def test_the_log_shipper_is_closed_on_shutdown():
    class Shipper:
        closed = None

        def close(self, timeout_s):
            self.closed = timeout_s

    shipper = Shipper()
    with live(create_router_app([Replica("r1", "http://x")], RouterConfig(salt="s", drain_s=7.0), shipper=shipper)):
        pass
    assert shipper.closed == 7.0


@pytest.mark.parametrize("spec, expected", [
    ("r1=http://127.0.0.1:9001", ("r1", "http://127.0.0.1:9001", "")),
    ("r1=https://me-r1.hf.space/,me/r1", ("r1", "https://me-r1.hf.space", "me/r1")),
])
def test_parse_replica(spec, expected):
    r = parse_replica(spec)
    assert (r.name, r.url, r.space_id) == expected


@pytest.mark.parametrize("spec", ["r1", "=http://x", "r1=x.hf.space", "r1=ftp://x", "bad name=http://x", "r1=http://x,notaspace"])
def test_parse_replica_refuses_a_bad_spec(spec):
    with pytest.raises(ValueError):
        parse_replica(spec)


@pytest.fixture
def cli(monkeypatch):
    """Runs main() up to the point where the server would start."""
    def run(*argv, token=None):
        seen = {}
        monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(app=app, **kw))
        monkeypatch.setattr("tinyllm.router.setup_logging", lambda *a: None)
        if token:
            monkeypatch.setenv("TINYLLM_REPLICA_TOKEN", token)
        else:
            monkeypatch.delenv("TINYLLM_REPLICA_TOKEN", raising=False)
        router_main(list(argv))
        return seen
    return run


def test_cli_builds_the_router(cli):
    seen = cli("--replica", "r1=http://a,me/r1", "--replica", "r2=http://b,me/r2", "--rate-limit", "30", "--trust-proxy",
               "--model-label", "tiny", token="tok", )
    router = seen["app"].state.router
    assert [(r.name, r.url, r.space_id) for r in router.replicas] == [("r1", "http://a", "me/r1"), ("r2", "http://b", "me/r2")]
    assert (router.cfg.rate_per_min, router.cfg.rate_burst, router.cfg.trust_proxy, router.cfg.model_label) == (30.0, 5, True, "tiny")
    assert router._token == "tok" and router.wake is not None
    assert seen["proxy_headers"] is False               # or uvicorn would believe X-Forwarded-For from loopback


def test_cli_without_a_token_never_wakes(cli):
    router = cli("--replica", "r1=http://a")["app"].state.router
    assert router._token is None and router.wake is None


@pytest.mark.parametrize("argv, token", [
    (["--replica", "r1=http://a", "--replica", "r1=http://b"], None),       # the same name twice
    (["--replica", "r1=http://a"], "tok"),                                  # a token, but nothing to restart
    (["--replica", "r1=http://a,me/r1", "--token", "tok"], None),           # the token is never a flag
    (["--replica", "r1=http://a", "--log-dataset", "me/logs"], None),
    ([], None),
])
def test_cli_refuses(cli, argv, token):
    with pytest.raises(SystemExit):
        cli(*argv, token=token)
