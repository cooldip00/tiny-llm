"""4.13 step 2: observability helpers (obs.py)."""

import json
import logging
import re

import pytest

from tinyllm.obs import (EngineSnapshot, JsonFormatter, Metrics, RequestRecord, hash_client,
                         itl_stats, percentile)


def test_percentile_matches_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([3.0], 95) == 3.0
    xs = [4, 1, 3, 2]                                     # numpy: 2.5, 3.85, 1.0, 4.0
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile(xs, 95) == pytest.approx(3.85)
    assert percentile(xs, 0) == 1 and percentile(xs, 100) == 4


def test_itl_from_token_times():
    assert itl_stats([1.0]) == (None, None)
    p50, p95 = itl_stats([0.0, 0.1, 0.2, 0.5])            # gaps 0.1, 0.1, 0.3
    assert p50 == pytest.approx(0.1) and p95 == pytest.approx(0.28)


def test_client_hash_is_salted_and_hides_the_ip():
    a = hash_client("203.0.113.7", "s1")
    assert a == hash_client("203.0.113.7", "s1") and len(a) == 16
    assert a != hash_client("203.0.113.7", "s2") and a != hash_client("203.0.113.8", "s1")
    assert "203" not in a


def test_request_record_never_holds_text():
    keys = set(RequestRecord("r1", 200).to_dict())
    assert not keys & {"prompt", "text", "completion", "ip"}


def rec(status=200, ttft=0.2, lat=1.0, toks=10, variant="main", client="c1"):
    return RequestRecord("r", status, variant=variant, client=client, ttft_s=ttft,
                         latency_s=lat, queue_wait_s=0.05, completion_tokens=toks)


def test_rolling_windows_drop_old_requests():
    m = Metrics(windows={"5m": 300.0, "1h": 3600.0}, clock=lambda: 0.0)
    m.observe(rec(ttft=0.1), now=0.0)
    m.observe(rec(ttft=0.3, client="c2"), now=200.0)
    m.observe(rec(status=429, ttft=None, lat=None, toks=0), now=400.0)
    m.observe(rec(status=500, ttft=None, lat=None, toks=0), now=401.0)
    s = m.stats(now=410.0)
    w5, w1 = s["windows"]["5m"], s["windows"]["1h"]
    assert w5["requests"] == 3 and w1["requests"] == 4       # the t=0 request left the 5 min window
    assert w5["ok"] == 1 and w5["ttft_s"]["p50"] == pytest.approx(0.3)
    assert w5["reject_rate"] == pytest.approx(1 / 3, abs=1e-4) and w5["error_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert w1["clients"] == 2 and w1["completion_tokens"] == 20
    assert w1["requests_per_min"] == pytest.approx(60 * 4 / 410.0, rel=1e-3)
    assert s["engine"] is None


LINE = re.compile(r'^[a-z_]+(\{[a-z_]+="[^"]*"(,[a-z_]+="[^"]*")*\})? -?[0-9.e+]+$')


def test_prometheus_text_format():
    m = Metrics()
    for t in (0.03, 0.2, 0.2, 7.0):
        m.observe(rec(ttft=t, lat=t + 1))
    m.observe(rec(status=429, ttft=None, lat=None, toks=0))
    m.observe(rec(variant="canary", ttft=0.5))
    m.snapshot(EngineSnapshot(0.0, queue_len=3, busy_slots=2, n_slots=8, tokens_per_s=512.5,
                              kv_used_frac=0.25))
    text = m.prometheus()
    lines = text.strip().split("\n")
    for ln in lines:
        assert ln.startswith("# HELP ") or ln.startswith("# TYPE ") or LINE.match(ln), ln
    assert 'tinyllm_requests_total{variant="main",status="200"} 4' in lines
    assert 'tinyllm_requests_total{variant="main",status="429"} 1' in lines
    assert 'tinyllm_requests_total{variant="canary",status="200"} 1' in lines
    # histogram buckets are cumulative and end with +Inf == count
    b = [int(ln.rsplit(" ", 1)[1]) for ln in lines if ln.startswith('tinyllm_ttft_seconds_bucket{variant="main"')]
    assert b == sorted(b) and b[-1] == 4
    assert 'tinyllm_ttft_seconds_bucket{variant="main",le="0.05"} 1' in lines
    assert 'tinyllm_ttft_seconds_count{variant="main"} 4' in lines
    assert "tinyllm_queue_length 3" in lines and "tinyllm_tokens_per_second 512.5" in lines


def test_json_log_lines():
    fmt = JsonFormatter()
    r = logging.LogRecord("tinyllm", logging.INFO, __file__, 1, "request done", None, None)
    r.fields = {"request_id": "abc", "ttft_s": 0.12}
    d = json.loads(fmt.format(r))
    assert d["msg"] == "request done" and d["request_id"] == "abc" and d["level"] == "info"
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        r2 = logging.LogRecord("tinyllm", logging.ERROR, __file__, 1, "step failed", None, sys.exc_info())
    d2 = json.loads(fmt.format(r2))
    assert "RuntimeError: boom" in d2["exc"] and "\n" not in fmt.format(r2)
