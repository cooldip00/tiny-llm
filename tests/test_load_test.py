"""4.13 step 3: the load generator (scripts/load_test.py)."""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from load_test import Result, main, poisson_arrivals, rated_load, summarize  # noqa: E402
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
