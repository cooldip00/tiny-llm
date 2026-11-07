"""6.5b step 3: SLO / error-budget maths (Metrics.history) and the demo and status pages."""

import pytest

from test_server import greedy, model, serve  # noqa: F401  (model is a fixture)
from tinyllm.obs import BURN_MIN_REQUESTS, HISTORY_MINUTES, EngineSnapshot, Metrics, RequestRecord, slo_counted


def rec(status=200, ttft=0.2, lat=1.0, toks=10):
    return RequestRecord("r", status, variant="main", client="c1", ttft_s=ttft, latency_s=lat,
                         completion_tokens=toks)


def snap(queue, busy, tps):
    return EngineSnapshot(t_unix=0.0, queue_len=queue, busy_slots=busy, n_slots=4, tokens_per_s=tps,
                          kv_used_frac=0.0)


def metrics(**kw):
    return Metrics(windows={"5m": 300.0}, clock=lambda: 0.0, wall=lambda: 1.7e9, **kw)


def test_counted_statuses():
    assert [s for s in (200, 400, 429, 499, 500, 503, 504) if slo_counted(s)] == [200, 429, 500, 503, 504]


def test_empty_history_has_the_full_shape():
    h = metrics().history(now=0.0)
    assert len(h["minutes"]) == HISTORY_MINUTES
    assert [m["minutes_ago"] for m in h["minutes"]] == list(range(HISTORY_MINUTES - 1, -1, -1))
    assert [m["covered"] for m in h["minutes"]] == [False] * (HISTORY_MINUTES - 1) + [True]
    assert h["snapshots"] == []
    for s in h["slo"].values():
        assert s["total"] == 0 and s["compliance"] is None and s["met"] is None
        assert s["budget_left"] is None and s["burn_rate_1h"] is None and not s["fast_burn"]
    assert h["slo"]["shed"]["judged"] is False and h["slo"]["ttft"]["threshold_s"] == 1.0


def test_slo_compliance_and_error_budget():
    m = metrics(slo_ttft_s=2.0)
    for i in range(196):
        m.observe(rec(ttft=0.5), now=float(i))
    m.observe(rec(ttft=2.5), now=200.0)                            # slow: TTFT SLO bad
    m.observe(rec(ttft=None), now=201.0)                           # 200 with no TTFT counts as slow
    m.observe(rec(status=500, ttft=None, lat=None, toks=0), now=202.0)
    m.observe(rec(status=429, ttft=None, lat=None, toks=0), now=203.0)
    m.observe(rec(status=400, ttft=None, lat=None, toks=0), now=204.0)   # client errors and
    m.observe(rec(status=499, ttft=None, lat=None, toks=0), now=205.0)   # disconnects count for none
    s = m.history(now=300.0)["slo"]

    t = s["ttft"]                                                  # judged on the 198 served
    assert (t["good"], t["total"]) == (196, 198)
    assert t["compliance"] == pytest.approx(196 / 198, abs=1e-6) and t["met"]
    assert t["budget_left"] == pytest.approx(1 - 2 / (0.05 * 198), abs=1e-6)
    a = s["availability"]                                          # 200 counted: 198 + 500 + 429
    assert (a["good"], a["total"]) == (199, 200) and a["met"]
    assert a["budget_left"] == pytest.approx(1 - 1 / (0.01 * 200), abs=1e-6)     # half the budget
    assert a["burn_rate_1h"] == pytest.approx((1 / 200) / 0.01)
    assert a["requests_1h"] == 200 and not a["fast_burn"]
    assert (s["shed"]["good"], s["shed"]["total"]) == (199, 200)


def test_missed_slo_and_fast_burn():
    m = metrics()
    for i in range(BURN_MIN_REQUESTS):
        m.observe(rec(status=500 if i < 3 else 200), now=float(i))
    a = m.history(now=60.0)["slo"]["availability"]
    assert not a["met"] and a["budget_left"] < 0                   # over budget: negative, the page clamps
    assert a["burn_rate_1h"] == pytest.approx(30.0) and a["fast_burn"]


def test_too_few_requests_never_flag_a_fast_burn():
    m = metrics()
    for i in range(BURN_MIN_REQUESTS - 1):
        m.observe(rec(status=500), now=float(i))
    a = m.history(now=60.0)["slo"]["availability"]
    assert a["burn_rate_1h"] == pytest.approx(100.0) and not a["fast_burn"]


def test_burn_rate_uses_the_last_hour_the_budget_the_whole_window():
    m = metrics()
    m.observe(rec(status=500), now=10.0)                           # an old failure, hour 0
    for i in range(20):
        m.observe(rec(), now=7200.0 + i)                           # hour 2: all good
    a = m.history(now=7300.0)["slo"]["availability"]
    assert a["total"] == 21 and a["good"] == 20
    assert a["burn_rate_1h"] == 0 and a["requests_1h"] == 20


def test_seven_day_window_drops_old_hours():
    m = metrics()
    m.observe(rec(status=500), now=0.0)
    day = 24 * 3600.0
    m.observe(rec(), now=7 * day - 1)                              # hour 167: still in the window
    a = m.history(now=7 * day - 1)["slo"]["availability"]
    assert a["total"] == 2
    m.observe(rec(), now=7 * day + 10)                             # hour 168: hour 0 falls out
    a = m.history(now=7 * day + 10)["slo"]["availability"]
    assert (a["good"], a["total"]) == (2, 2)
    assert len(m._slo) == 2                                        # pruned, not just filtered
    assert m.history(now=7 * day + 10)["slo_window_s"] == 7 * day


def test_minutes_and_snapshots():
    m = metrics()
    m.observe(rec(), now=0.0)                                      # over an hour old: gone
    m.snapshot(snap(0, 0, 0.0), now=0.0)                           # likewise
    m.observe(rec(ttft=0.1, lat=1.0, toks=5), now=3600.0 - 125)    # 2 min ago
    m.observe(rec(ttft=0.3, lat=2.0, toks=7), now=3600.0 - 121)
    m.observe(rec(status=429, ttft=None, lat=None, toks=0), now=3600.0 - 10)
    m.snapshot(snap(2, 4, 31.25), now=3590.0)
    h = m.history(now=3600.5)
    by_ago = {x["minutes_ago"]: x for x in h["minutes"]}
    two = by_ago[2]
    assert (two["requests"], two["ok"], two["completion_tokens"]) == (2, 2, 12)
    assert two["ttft_p50_s"] == pytest.approx(0.2) and two["latency_p95_s"] == pytest.approx(1.95)
    assert (by_ago[0]["shed"], by_ago[0]["ttft_p50_s"]) == (1, None)
    assert sum(x["requests"] for x in h["minutes"]) == 3
    assert all(x["covered"] for x in h["minutes"])
    assert h["snapshots"] == [{"seconds_ago": 10.5, "queue_len": 2, "busy_slots": 4, "n_slots": 4,
                               "tokens_per_s": 31.2}]
    assert h["uptime_s"] == 3600.5 and h["now_unix"] == 1.7e9


# -- live server -------------------------------------------------------------------------

def test_pages_and_history_endpoint(model):
    with serve(model) as live:
        assert live.post(**greedy()).status_code == 200
        assert live.post(prompt="", max_tokens=1).status_code == 400
        for path, needle in (("/", "v1/completions"), ("/status", "stats/history")):
            r = live.client.get(path)
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
            assert needle in r.text and "http" not in r.text.split(needle)[0][-20:]   # relative URL
        h = live.client.get("/stats/history").json()
        assert h["variants"]["main"]["n_slots"] == 3 and h["variants"]["main"]["version"] == "test-v1"
        assert h["minutes"][-1]["requests"] == 2 and h["minutes"][-1]["ok"] == 1
        assert h["slo"]["availability"]["total"] == 1                 # the 400 is not counted
        assert h["slo"]["ttft"]["good"] == 1


def test_polling_the_pages_is_not_counted(model):
    with serve(model) as live:
        assert live.post(**greedy()).status_code == 200
        for _ in range(3):
            for path in ("/", "/status", "/stats/history", "/stats"):
                assert live.client.get(path).status_code == 200
        win = next(iter(live.client.get("/stats").json()["windows"].values()))
        assert win["requests"] == 1
        assert live.client.get("/stats/history").json()["slo"]["availability"]["total"] == 1
