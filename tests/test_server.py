"""4.13 step 2: the HTTP server (server.py) on a tiny CPU model, through a real uvicorn server.

httpx's in-process ASGI transport buffers whole responses, so streaming, disconnects and
shutdown are tested against uvicorn on 127.0.0.1 (random port) in a background thread.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
import torch
import uvicorn

from tinyllm.config import ModelConfig
from tinyllm.engine import ContinuousEngine
from tinyllm.generate import generate
from tinyllm.model import Transformer
from tinyllm.obs import Metrics
from tinyllm.server import EngineWorker, Faults, ServerConfig, Variant, cpu_limit, create_app, resolve_salt
from tinyllm.web_common import forwarded_ip

CFG = ModelConfig(vocab_size=128, max_seq_len=48, d_model=64, n_layers=3, n_heads=4, ffn_hidden=160)
PROMPT = [5, 17, 99, 3, 42, 8]


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(CFG).eval()
    with torch.no_grad():                         # well-separated logits (see test_kv_cache)
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return m


def solo(model, prompt, n) -> list[int]:
    out = generate(model, torch.tensor([prompt]), n, temperature=0.0)
    return out[0, len(prompt):].tolist()


class AsciiTokenizer:
    """1 char = 1 id (mod 128); id 0 is EOS and decodes to nothing."""
    eos_id = 0

    def encode(self, text):
        return [ord(c) % 128 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != 0)


class Live:
    def __init__(self, server, thread, worker, port):
        self.server, self.thread, self.worker = server, thread, worker
        self.url = f"http://127.0.0.1:{port}"
        self.client = httpx.Client(base_url=self.url, timeout=30)

    def post(self, **body):
        headers = body.pop("headers", None)
        return self.client.post("/v1/completions", json=body, headers=headers)

    def wait(self, cond, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False


@contextmanager
def serve(model, n_slots=3, faults=None, shipper=None, **cfg):
    cfg = ServerConfig(**{"snapshot_s": 0.2, "drain_s": 10.0, **cfg})
    metrics = Metrics()
    worker = EngineWorker(ContinuousEngine(model, n_slots), metrics, cfg.max_queue, cfg.snapshot_s, faults)
    app = create_app([Variant("main", "test-v1", worker, AsciiTokenizer(), CFG.max_seq_len, CFG.vocab_size)],
                     cfg, metrics, shipper)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           access_log=False, lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    end = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < end and thread.is_alive(), "server did not start"
        time.sleep(0.01)
    live = Live(server, thread, worker, server.servers[0].sockets[0].getsockname()[1])
    try:
        yield live
    finally:
        live.client.close()
        server.should_exit = True
        thread.join(timeout=15)
        assert not thread.is_alive(), "server did not stop"


def greedy(**kw):
    return {"prompt": PROMPT, "max_tokens": 10, "temperature": 0.0, "ignore_eos": True,
            "return_token_ids": True, **kw}


def sse(resp) -> list:
    out = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


# -- correctness ------------------------------------------------------------------------

def test_greedy_completion_matches_generate(model):
    want = solo(model, PROMPT, 10)
    with serve(model) as s:
        r = s.post(**greedy(), headers={"X-Request-ID": "abc-1"})
        assert r.status_code == 200, r.text
        d = r.json()
        c = d["choices"][0]
        assert c["token_ids"] == want and c["text"] == AsciiTokenizer().decode(want)
        assert c["finish_reason"] == "length" and d["object"] == "text_completion"
        assert d["usage"] == {"prompt_tokens": 6, "completion_tokens": 10, "total_tokens": 16}
        assert r.headers["x-request-id"] == "abc-1" and r.headers["x-model-version"] == "test-v1"
        assert d["id"] == "cmpl-abc-1"
        # a text prompt goes through the tokenizer
        r2 = s.post(**greedy(prompt="hi there"))
        assert r2.json()["choices"][0]["token_ids"] == solo(model, AsciiTokenizer().encode("hi there"), 10)
        assert len(r2.headers["x-request-id"]) == 16          # generated when the client sends none
        assert s.client.get("/health", headers={"X-Request-ID": "bad id!"}).headers["x-request-id"] != "bad id!"


def test_stream_equals_non_stream(model):
    with serve(model) as s:
        whole = s.post(**greedy(max_tokens=12)).json()["choices"][0]
        with s.client.stream("POST", "/v1/completions", json=greedy(max_tokens=12, stream=True)) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            assert r.headers["x-model-version"] == "test-v1"
            events = sse(r)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert "".join(c["choices"][0]["text"] for c in chunks) == whole["text"]
    assert [t for c in chunks for t in c["choices"][0]["token_ids"]] == whole["token_ids"]
    assert [c["choices"][0]["finish_reason"] for c in chunks][-1] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert chunks[-1]["usage"]["completion_tokens"] == 12


def test_eos_stops_with_finish_reason_stop(model):
    want = solo(model, PROMPT, 30)
    with serve(model) as s:
        tok = AsciiTokenizer()
        tok.eos_id = want[4]                                  # make the 5th greedy token the EOS
        s.server.config.app.state.srv.router.primary.tokenizer = tok
        c = s.post(**greedy(max_tokens=30, ignore_eos=False)).json()["choices"][0]
    first = want.index(want[4])
    assert c["token_ids"] == want[:first + 1] and c["finish_reason"] == "stop"


def test_concurrent_requests_match_solo(model):
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(1, 128, (int(torch.randint(2, 12, (1,), generator=g)),), generator=g).tolist()
               for _ in range(7)]
    news = [int(torch.randint(1, 16, (1,), generator=g)) for _ in prompts]
    got = [None] * len(prompts)
    with serve(model, n_slots=3) as s:
        def run(i):
            got[i] = s.post(**greedy(prompt=prompts[i], max_tokens=news[i])).json()["choices"][0]["token_ids"]
        ts = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert got == [solo(model, p, n) for p, n in zip(prompts, news)]


@pytest.mark.parametrize("body", [
    {"prompt": []},
    {"prompt": ""},
    {"prompt": [1, 2, 300]},
    {"prompt": [1, 2], "max_tokens": 47},                     # 2 + 47 > 48
    {"prompt": [1, 2], "max_tokens": 0},
    {"prompt": [1, 2], "temperature": -1},
    {"prompt": [1, 2], "top_p": 0},
    {"prompt": [1, 2], "top_k": 0},
    {"max_tokens": 3},
])
def test_bad_requests_are_400(model, body):
    with serve(model) as s:
        r = s.post(**body)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["type"] == "invalid_request_error" and r.json()["error"]["message"]
        assert "x-request-id" in r.headers
        assert 'tinyllm_requests_total{variant="main",status="400"} 1' in s.client.get("/metrics").text


# -- hardening --------------------------------------------------------------------------

def test_queue_full_is_429_with_retry_after(model):
    with serve(model, n_slots=1, max_queue=1, faults=Faults(step_delay_s=0.03)) as s:
        res = []
        ts = [threading.Thread(target=lambda: res.append(s.post(**greedy(max_tokens=15)).status_code))
              for _ in range(2)]
        for t in ts:
            t.start()
        assert s.wait(lambda: s.worker.outstanding() == 2)
        r = s.post(**greedy())
        assert r.status_code == 429 and r.headers["retry-after"] == "1"
        assert r.json()["error"]["type"] == "server_busy"
        for t in ts:
            t.join()
        assert res == [200, 200]
        assert s.post(**greedy()).status_code == 200            # room again


def test_rate_limit_per_client(model):
    with serve(model, rate_per_min=60, rate_burst=2) as s:
        codes = [s.post(**greedy(max_tokens=2)) for _ in range(3)]
        assert [r.status_code for r in codes] == [200, 200, 429]
        assert codes[2].headers["retry-after"] == "1" and codes[2].json()["error"]["type"] == "rate_limit_error"
        stats = s.client.get("/stats").json()["windows"]["5m"]
        assert stats["requests"] == 3 and stats["reject_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_forwarded_ip_ignores_what_the_client_sent():
    assert forwarded_ip("1.1.1.1") == "1.1.1.1"
    assert forwarded_ip("6.6.6.6, 1.1.1.1") == "1.1.1.1"                  # spoofed entry on the left
    assert forwarded_ip("6.6.6.6, 1.1.1.1, 10.0.0.2", hops=2) == "1.1.1.1"
    assert forwarded_ip("1.1.1.1", hops=2) == "1.1.1.1"                  # fewer entries than hops
    assert forwarded_ip(" , ") is None


def test_spoofed_forwarded_for_cannot_dodge_the_rate_limit(model):
    with serve(model, rate_per_min=60, rate_burst=2, trust_proxy=True) as s:
        def post(xff):
            return s.client.post("/v1/completions", json=greedy(max_tokens=2),
                                 headers={"x-forwarded-for": xff}).status_code
        # one visitor (1.1.1.1, appended by the proxy) faking a new address each time
        assert [post(f"6.6.6.{i}, 1.1.1.1") for i in range(3)] == [200, 200, 429]
        assert post("2.2.2.2") == 200                                    # another visitor has its own bucket


def test_router_header_is_ignored_by_default(model):
    """The public Space runs this image: a visitor must not be able to pick their own identity."""
    with serve(model, rate_per_min=60, rate_burst=2) as s:
        def post(i):
            return s.client.post("/v1/completions", json=greedy(max_tokens=2),
                                 headers={"x-tinyllm-client": f"{i:016x}"}).status_code
        assert [post(i) for i in range(3)] == [200, 200, 429]


def test_trust_router_takes_the_visitor_hash_from_the_header(model):
    with serve(model, rate_per_min=60, rate_burst=1, trust_router=True) as s:
        def post(value):
            return s.client.post("/v1/completions", json=greedy(max_tokens=2),
                                 headers={"x-tinyllm-client": value}).status_code
        assert [post("a" * 16), post("a" * 16)] == [200, 429]            # one visitor, one bucket
        assert post("b" * 16) == 200                                     # another visitor
        assert post("not a hash") == 200                                 # malformed: falls back to the address...
        assert post("A" * 16) == 429                                     # ...and shares that one bucket


def test_cpu_limit(tmp_path, monkeypatch):
    monkeypatch.delenv("CPU_CORES", raising=False)
    assert cpu_limit(str(tmp_path)) is None                              # nothing known: torch decides
    (tmp_path / "cpu.max").write_text("200000 100000\n")                 # docker run --cpus 2 (cgroup v2)
    assert cpu_limit(str(tmp_path)) == 2
    (tmp_path / "cpu.max").write_text("150000 100000\n")                 # --cpus 1.5 → 2 threads
    assert cpu_limit(str(tmp_path)) == 2
    (tmp_path / "cpu.max").write_text("max 100000\n")                    # no cap: fall through
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")           # cgroup v1, no cap either
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert cpu_limit(str(tmp_path)) is None
    monkeypatch.setenv("CPU_CORES", "2")                                 # HF's variable
    assert cpu_limit(str(tmp_path)) == 2
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("400000\n")       # a v1 cap beats CPU_CORES
    assert cpu_limit(str(tmp_path)) == 4
    monkeypatch.setenv("CPU_CORES", "lots")
    assert cpu_limit(str(tmp_path / "missing")) is None


def test_salt_sources(monkeypatch):
    for k in ("TINYLLM_LOG_SALT", "SPACE_SIGNING_SECRET"):
        monkeypatch.delenv(k, raising=False)
    a, b = resolve_salt(), resolve_salt()
    assert a[1] == b[1] == "random" and a[0] != b[0]
    monkeypatch.setenv("SPACE_SIGNING_SECRET", "hf-secret")
    salt, src = resolve_salt()
    assert src == "SPACE_SIGNING_SECRET" and salt == resolve_salt()[0] and "hf-secret" not in salt
    monkeypatch.setenv("TINYLLM_LOG_SALT", "mine")
    assert resolve_salt() == ("mine", "TINYLLM_LOG_SALT")                # the user's salt wins
    assert resolve_salt("explicit") == ("explicit", "config")


def test_timeout_is_504_and_frees_the_slot(model):
    with serve(model, n_slots=1, timeout_s=0.3, faults=Faults(step_delay_s=0.03)) as s:
        r = s.post(**greedy(max_tokens=40))
        assert r.status_code == 504, r.text
        assert s.wait(lambda: s.worker.outstanding() == 0)
        assert s.client.get("/debug/state").json()["main"]["jobs"] == []
        with s.client.stream("POST", "/v1/completions", json=greedy(max_tokens=40, stream=True)) as rs:
            events = sse(rs)
        assert events[-2]["error"]["code"] == 504 and events[-1] == "[DONE]"
        s.worker.faults.step_delay_s = 0.0
        assert s.post(**greedy(max_tokens=3)).status_code == 200
        assert 'tinyllm_requests_total{variant="main",status="504"} 2' in s.client.get("/metrics").text


def test_client_disconnect_aborts_the_request(model):
    with serve(model, n_slots=1, faults=Faults(step_delay_s=0.03)) as s:
        # non-streaming: the client gives up after 0.3 s of a ~1.2 s request
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(s.url + "/v1/completions", json=greedy(max_tokens=40), timeout=0.3)
        assert s.wait(lambda: s.worker.outstanding() == 0, timeout=1.0)
        # streaming: read two chunks, then hang up
        with httpx.Client(base_url=s.url, timeout=10) as c:
            with c.stream("POST", "/v1/completions", json=greedy(max_tokens=40, stream=True)) as r:
                lines = r.iter_lines()
                assert next(lines).startswith("data: ")
        assert s.wait(lambda: s.worker.outstanding() == 0, timeout=1.0)
        assert s.wait(lambda: 'tinyllm_requests_total{variant="main",status="499"} 2'
                      in s.client.get("/metrics").text, timeout=1.0)
        assert s.client.get("/debug/state").json()["main"]["jobs"] == []


def test_graceful_shutdown_finishes_in_flight_requests(model):
    want = solo(model, PROMPT, 20)
    with serve(model, faults=Faults(step_delay_s=0.02)) as s:
        res = {}
        t = threading.Thread(target=lambda: res.update(r=s.post(**greedy(max_tokens=20))))
        t.start()
        assert s.wait(lambda: s.worker.outstanding() == 1)
        s.server.should_exit = True
        t.join(timeout=10)
        assert res["r"].status_code == 200 and res["r"].json()["choices"][0]["token_ids"] == want
        s.thread.join(timeout=10)
        assert not s.worker.accepting and not s.worker.alive


# -- observability ----------------------------------------------------------------------

def test_health_ready_stats_metrics_models(model):
    with serve(model) as s:
        assert s.client.get("/health").json() == {"status": "ok"}
        assert s.client.get("/ready").json() == {"ready": True}
        s.post(**greedy())
        m = s.client.get("/metrics")
        assert m.headers["content-type"].startswith("text/plain")
        assert 'tinyllm_requests_total{variant="main",status="200"} 1' in m.text
        assert 'tinyllm_completion_tokens_total{variant="main"} 10' in m.text
        assert 'tinyllm_ttft_seconds_count{variant="main"} 1' in m.text
        assert s.wait(lambda: "tinyllm_slots 3" in s.client.get("/metrics").text)
        st = s.client.get("/stats").json()
        w = st["windows"]["5m"]
        assert w["requests"] == 1 and w["ok"] == 1 and w["completion_tokens"] == 10
        assert w["ttft_s"]["p50"] > 0 and w["latency_s"]["p95"] >= w["ttft_s"]["p50"]
        assert st["engine"]["n_slots"] == 3 and st["variants"]["main"]["version"] == "test-v1"
        assert s.client.get("/v1/models").json()["data"][0]["id"] == "main"
        dbg = s.client.get("/debug/state").json()["main"]
        assert dbg["steps"] >= 9 and dbg["jobs"] == [] and dbg["failed"] is None


def test_request_log_line_has_timings_and_no_text(model, caplog):
    with caplog.at_level(logging.INFO, logger="tinyllm"):
        with serve(model) as s:
            r = s.post(**greedy(prompt="secret words"))
            rid = r.headers["x-request-id"]
            assert s.wait(lambda: any(getattr(x, "fields", {}).get("request_id") == rid for x in caplog.records))
    rec = next(x for x in caplog.records if getattr(x, "fields", {}).get("request_id") == rid)
    f = rec.fields
    assert rec.getMessage() == "request" and f["status"] == 200 and f["finish_reason"] == "length"
    assert f["prompt_tokens"] == 12 and f["completion_tokens"] == 10 and f["model_version"] == "test-v1"
    assert f["ttft_s"] > 0 and f["latency_s"] >= f["ttft_s"] and f["queue_wait_s"] >= 0
    assert f["itl_p50_s"] is not None and f["queue_depth_at_arrival"] == 0
    assert len(f["client"]) == 16 and "127.0.0.1" not in json.dumps(f)
    assert "secret" not in json.dumps(f) and r.json()["choices"][0]["text"] not in json.dumps(f)


# -- failure injection (D-028 amendment) ------------------------------------------------

def test_engine_error_fails_requests_and_readiness(model, caplog):
    with caplog.at_level(logging.ERROR, logger="tinyllm"):
        with serve(model, faults=Faults(fail_at_step=3)) as s:
            r = s.post(**greedy())
            assert r.status_code == 500 and r.json()["error"]["type"] == "server_error"
            assert s.client.get("/health").status_code == 503
            ready = s.client.get("/ready")
            assert ready.status_code == 503 and "injected fault" in ready.json()["reasons"][0]
            assert s.post(**greedy()).status_code == 503          # refused at the door
            assert 'tinyllm_requests_total{variant="main",status="500"} 1' in s.client.get("/metrics").text
    assert any(x.getMessage() == "engine step failed: not ready" and x.exc_info for x in caplog.records)


def test_oom_fails_in_flight_requests_then_recovers(model):
    want = solo(model, PROMPT, 10)
    with serve(model, faults=Faults(oom_at_step=2)) as s:
        res = []
        ts = [threading.Thread(target=lambda: res.append(s.post(**greedy()))) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert {r.status_code for r in res} <= {200, 503}         # a late arrival may miss the OOM
        assert any(r.status_code == 503 and "out of memory" in r.json()["error"]["message"] for r in res)
        assert s.wait(lambda: s.worker.outstanding() == 0)
        assert s.client.get("/ready").status_code == 200 and s.client.get("/health").status_code == 200
        r = s.post(**greedy())
        assert r.status_code == 200 and r.json()["choices"][0]["token_ids"] == want
