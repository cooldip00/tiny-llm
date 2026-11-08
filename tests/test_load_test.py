"""4.13 step 3: the load generator (scripts/load_test.py)."""

import json
import random
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from load_test import (BANDWIDTH, Result, main, poisson_arrivals, rated_load, summarize,  # noqa: E402
                       summarize_sequential)
import load_test  # noqa: E402
from test_server import model, serve  # noqa: E402, F401  (model is a fixture)

from tinyllm.server import Faults  # noqa: E402


def test_poisson_arrivals_have_the_requested_rate():
    a = poisson_arrivals(10.0, 1000.0, random.Random(0))
    assert a == sorted(a) and 0 <= a[0] and a[-1] < 1000.0
    assert len(a) / 1000.0 == pytest.approx(10.0, rel=0.03)
    gaps = [y - x for x, y in zip(a, a[1:])]
    assert sum(gaps) / len(gaps) == pytest.approx(0.1, rel=0.03)
    assert a == poisson_arrivals(10.0, 1000.0, random.Random(0))       # seeded


def ok(ttft, tokens=4):
    return Result(0.0, 200, ttft_s=ttft, latency_s=ttft + 0.3, tokens=tokens, itl_s=[0.1] * (tokens - 1))


def test_summary_and_slo_checks():
    rs = [ok(0.1), ok(0.2), ok(0.3), Result(0.0, 429), Result(0.0, 504, error="timed out"),
          Result(0.0, -1, error="ConnectError")]
    s = summarize(rs, rate=2.0, wall_s=2.0, ttft_slo_s=1.0)
    assert (s["sent"], s["ok"], s["shed_429"], s["failed"], s["other"]) == (6, 3, 1, 2, 0)
    assert s["completion_tokens"] == 12 and s["tok_s"] == 6.0 and s["achieved_rps"] == 1.5
    assert s["ttft_s"]["p50"] == pytest.approx(0.2) and s["itl_s"]["p50"] == pytest.approx(0.1)
    assert s["availability"] == pytest.approx(4 / 6, abs=1e-4) and s["shed_rate"] == pytest.approx(1 / 6, abs=1e-4)
    assert s["slo"] == {"ttft_p95": True, "availability": False, "shed": False} and not s["meets_slo"]
    good = summarize([ok(0.1)] * 10, 1.0, 10.0, 1.0)
    slow = summarize([ok(2.0)] * 10, 4.0, 10.0, 1.0)
    assert good["meets_slo"] and slow["slo"]["ttft_p95"] is False
    # rated load stops at the first miss, even if a higher rate happens to pass
    assert rated_load([{**good, "rate": 1.0}, {**good, "rate": 2.0}, {**slow, "rate": 4.0},
                       {**good, "rate": 8.0}]) == 2.0
    assert rated_load([{**slow, "rate": 1.0}]) is None


def test_load_test_against_our_server(model, tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("once upon\nthe dog\na big red ball\n")
    out = tmp_path / "load.json"
    # 2 slots + 1 queued, ~0.2 s per request: rate 2 fits, rate 100 must shed
    with serve(model, n_slots=2, max_queue=1, faults=Faults(step_delay_s=0.02)) as s:
        main(["--url", s.url, "--rates", "2,100", "--duration", "1.5", "--max-tokens", "8",
              "--prompts", str(prompts), "--warmup", "1", "--pause", "0.2", "--power-mode", "test",
              "--out", str(out), "--keep-requests"])
    rec = json.loads(out.read_text())
    low, high = (r["summary"] for r in rec["rows"])
    assert rec["model"] == "main" and rec["provenance"]["power_mode"] == "test"
    assert low["sent"] > 0 and low["ok"] == low["sent"] and low["completion_tokens"] == 8 * low["ok"]
    assert low["ttft_s"]["p50"] > 0 and low["itl_s"]["p50"] > 0 and low["meets_slo"]
    assert high["shed_429"] > 0 and high["failed"] == 0 and high["ok"] > 0
    assert high["ok"] + high["shed_429"] == high["sent"] and not high["slo"]["shed"]
    assert rec["rated_load_rps"] == 2.0
    req = rec["rows"][0]["requests"][0]
    assert req["status"] == 200 and req["tokens"] == 8 and len(req["itl_s"]) >= 1


def test_existing_output_is_never_overwritten(tmp_path):
    out = tmp_path / "load.json"
    out.write_text("earlier result")
    with pytest.raises(SystemExit):                          # before any request: no server needed
        main(["--url", "http://127.0.0.1:9", "--power-mode", "test", "--out", str(out)])
    assert out.read_text() == "earlier result"


FAKE_TOKEN = "fake-login-for-tests"


def test_hf_login_is_sent_counted_by_replica_and_never_saved(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        body = ('data: {"choices":[{"text":"a"}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, headers={"X-Replica": "r1" if len(seen) % 2 else "r2"}, text=body)

    real = httpx.AsyncClient
    monkeypatch.setattr(load_test.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: FAKE_TOKEN)
    out = tmp_path / "load.json"
    main(["--url", "https://x-y.hf.space", "--hf-login", "--rates", "40", "--duration", "0.5", "--max-tokens", "1",
          "--warmup", "1", "--pause", "0", "--power-mode", "test", "--out", str(out), "--keep-requests"])
    assert len(seen) > 3 and set(seen) == {f"Bearer {FAKE_TOKEN}"}
    text = out.read_text()
    assert FAKE_TOKEN not in text
    rec = json.loads(text)
    assert rec["workload"]["hf_login"] is True
    s = rec["rows"][0]["summary"]
    assert set(s["by_replica"]) == {"r1", "r2"} and sum(s["by_replica"].values()) == s["ok"] == s["sent"]
    assert {r["replica"] for r in rec["rows"][0]["requests"]} == {"r1", "r2"}


@pytest.mark.parametrize("url", ["http://x.hf.space", "https://example.com", "https://x.hf.space.evil.com",
                                 "https://x.hf.space@evil.com", "https://hf.space", "http://127.0.0.1:8000"])
def test_hf_login_refused_for_any_other_url(url, tmp_path, monkeypatch):
    def no_token():
        raise AssertionError("the login must not even be read")

    monkeypatch.setattr("huggingface_hub.get_token", no_token)
    with pytest.raises(SystemExit) as e:
        main(["--url", url, "--hf-login", "--power-mode", "test", "--out", str(tmp_path / "x.json")])
    assert e.value.code == 2


def test_hf_login_needs_a_login(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    with pytest.raises(SystemExit) as e:
        main(["--url", "https://x.hf.space", "--hf-login", "--power-mode", "test", "--out", str(tmp_path / "x.json")])
    assert e.value.code == 2


def test_by_replica_is_empty_on_a_single_server():
    assert summarize([ok(0.1)], 1.0, 1.0, 1.0)["by_replica"] == {}


def test_sequential_summary_excludes_prefill():
    # 11 tokens: first at 0.5 s (prefill), last at 1.5 s -> 10 decode tokens in 1 s
    rs = [Result(0.0, 200, ttft_s=0.5, latency_s=1.5, tokens=11, itl_s=[0.1] * 10),
          Result(0.0, 200, ttft_s=0.1, latency_s=0.6, tokens=11, itl_s=[0.05] * 10),   # 20 tok/s
          Result(0.0, 503, error="busy")]
    s = summarize_sequential(rs, wall_s=2.1, weight_bytes=BANDWIDTH / 40)            # ceiling 40 tok/s
    assert (s["sent"], s["ok"], s["tokens"], s["errors"]) == (3, 2, [11], ["busy"])
    assert s["decode_tok_s"]["p50"] == pytest.approx(15.0) and s["ceiling_tok_s"] == 40.0
    assert s["pct_of_ceiling"] == pytest.approx(37.5)
    assert "pct_of_ceiling" not in summarize_sequential(rs, 2.1, None)


def test_sequential_mode_against_our_server(model, tmp_path):
    out, prompts = tmp_path / "seq.json", tmp_path / "prompts.txt"
    prompts.write_text("once upon\nthe dog\n")                 # the tiny model's context is 48
    with serve(model, n_slots=2, faults=Faults(step_delay_s=0.01)) as s:
        main(["--url", s.url, "--sequential", "3", "--max-tokens", "6", "--warmup", "1", "--prompts", str(prompts),
              "--power-mode", "test", "--weight-bytes", "4.23e9", "--out", str(out), "--keep-requests"])
    rec = json.loads(out.read_text())
    seq = rec["sequential"]
    assert seq["sent"] == seq["ok"] == 3 and seq["tokens"] == [6] and "rows" not in rec
    # a 0.01 s delay per engine step -> roughly 100 tok/s; the ceiling is set to 100 tok/s
    assert 0 < seq["decode_tok_s"]["p50"] < 150 and seq["ceiling_tok_s"] == 100.0
    assert seq["pct_of_ceiling"] == pytest.approx(seq["decode_tok_s"]["p50"], abs=0.1)
    assert rec["workload"]["sequential"] == 3 and len(rec["requests"]) == 3
    sent = [r["sent_s"] for r in rec["requests"]]
    assert sent == sorted(sent) and all(b - a >= r["latency_s"] for a, b, r in zip(sent, sent[1:], rec["requests"]))
