"""scripts/analyze_logs.py on synthetic logs with known answers (6.5b step 4)."""

import calendar
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_logs as A  # noqa: E402

D1 = calendar.timegm((2026, 11, 6, 10, 0, 0))            # 2026-11-06 10:00 UTC
D2 = calendar.timegm((2026, 11, 7, 13, 0, 0))            # 2026-11-07 13:00 UTC


def req(ts, status=200, client="a", ttft=0.5, lat=2.0, toks=10, finish="length", error=None, msg="request"):
    return {"ts": ts, "level": "info", "msg": msg, "status": status, "client": client, "ttft_s": ttft,
            "latency_s": lat, "queue_wait_s": 0.1 if status == 200 else None, "prompt_tokens": 4,
            "completion_tokens": toks if status == 200 else 0, "finish_reason": finish, "error": error,
            "stream": status == 200 and client == "b"}


def write(root: Path, rel: str, rows: list) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))


@pytest.fixture
def logs(tmp_path):
    # run 1 (day 1): clean stop; run 2 (day 2): no stop line (hard kill or still up), over two hours
    write(tmp_path, "2026-11-06/20261106T095900Z-sp-10.jsonl", [
        {"ts": D1, "msg": "server ready", "slo_ttft_s": 2.0},
        {"ts": D1 + 1, "msg": "engine", "queue_len": 0, "busy_slots": 1, "n_slots": 4, "tokens_per_s": 20.0},
        req(D1 + 10, ttft=0.4), req(D1 + 20, client="b", ttft=0.6), req(D1 + 30, ttft=3.0),   # one slow
        req(D1 + 40, status=400, ttft=None, lat=None, finish="error", error="prompt: too long"),
        {"ts": D1 + 50, "msg": "stopped"},
    ])
    write(tmp_path, "2026-11-07/20261107T125500Z-sp-13.jsonl", [
        {"ts": D2, "msg": "server ready", "slo_ttft_s": 2.0},
        {"ts": D2 + 1, "msg": "engine", "queue_len": 3, "busy_slots": 4, "n_slots": 4, "tokens_per_s": 50.0},
        {"ts": D2 + 31, "msg": "engine", "queue_len": 1, "busy_slots": 4, "n_slots": 4, "tokens_per_s": 40.0},
        req(D2 + 10, client="c", ttft=1.0), req(D2 + 11, client="c", ttft=1.5, lat=12.0, msg="slow request"),
        req(D2 + 12, status=429, client="d", ttft=None, lat=None, finish="error", error="server busy"),
        req(D2 + 13, status=504, client="c", ttft=None, lat=None, finish="timeout", error="request timed out"),
        req(D2 + 14, status=499, client="c", ttft=None, lat=None, finish="abort"),
        "{not json", '"a string"', '{"msg": "no ts"}',
        {"ts": D2 + 15, "msg": "something new"},
        {"ts": D2 + 16, "level": "error", "msg": "engine step failed: not ready"},
    ])
    write(tmp_path, "2026-11-07/20261107T125500Z-sp-14.jsonl", [
        {"ts": D2 + 3600, "msg": "engine", "queue_len": 0, "busy_slots": 2, "n_slots": 4, "tokens_per_s": 30.0},
        req(D2 + 3610, client="a", ttft=0.2),
    ])
    return tmp_path


def test_report_numbers(logs):
    records, counts = A.read_logs(logs)
    assert counts == {"files": 3, "lines": 22, "corrupt": 3}
    rep = A.analyze(records)
    assert rep["runs"] == {"starts": 2, "runs": 2, "clean_stops": 1, "uptime_h": round((50 + 3610) / 3600, 3)}
    assert rep["unknown_lines"] == {"something new": 1}
    assert rep["requests"]["total"] == 10
    assert rep["requests"]["per_day"] == {"2026-11-06": {"200": 3, "400": 1},
                                          "2026-11-07": {"200": 3, "429": 1, "504": 1, "499": 1}}
    assert rep["visitors"] == {"per_day": {"2026-11-06": 2, "2026-11-07": 3}, "total": 4}
    ttft = rep["ttft_s"]                                   # 200s: 0.4 0.6 3.0 1.0 1.5 0.2
    assert ttft["n"] == 6 and ttft["p50"] == pytest.approx(0.8)
    assert rep["tokens"] == {"prompt": 24, "completion": 60, "completion_per_request": 10.0}
    assert rep["finish_reasons"] == {"length": 6, "error": 2, "timeout": 1, "abort": 1}
    slo = rep["slo"]
    # TTFT ≤ 2 s: 5 of 6 served; availability: 1 failure (504) of 8 counted (6 + 429 + 504)
    assert (slo["ttft"]["good"], slo["ttft"]["total"], slo["ttft"]["met"]) == (5, 6, False)
    assert slo["ttft"]["budget_left"] == pytest.approx(1 - 1 / (0.05 * 6), abs=1e-6)
    assert (slo["availability"]["good"], slo["availability"]["total"]) == (7, 8)
    assert (slo["shed"]["good"], slo["shed"]["total"], slo["shed"]["judged"]) == (7, 8, False)
    assert slo["ttft"]["threshold_s"] == 2.0
    assert rep["errors"][0]["count"] == 1 and {e["status"] for e in rep["errors"]} == {400, 429, 504}
    assert rep["engine_errors"] == {"engine step failed: not ready": 1}
    assert rep["busiest_hours_utc"][0] == {"hour": 13, "requests": 5}
    assert rep["peak_hour"] == {"hour": "2026-11-07 13:00", "requests": 5}
    sn = rep["snapshots"]
    assert sn["n"] == 4 and sn["n_slots"] == 4 and sn["busy_slots_max"] == 4 and sn["queue_len"]["max"] == 3
    assert sn["all_slots_busy_share"] == 0.5 and sn["busy_slots_mean"] == 2.75 and sn["tokens_per_s_max"] == 50.0


def test_since_and_threshold_override(logs):
    records, _ = A.read_logs(logs)
    rep = A.analyze(records, since=calendar.timegm((2026, 11, 7, 0, 0, 0)), slo_ttft_s=1.0)
    assert list(rep["requests"]["per_day"]) == ["2026-11-07"] and rep["runs"]["starts"] == 1
    assert (rep["slo"]["ttft"]["good"], rep["slo"]["ttft"]["total"]) == (2, 3)   # 1.0, 0.2 ≤ 1 s; 1.5 not


def test_empty_folder(tmp_path):
    records, counts = A.read_logs(tmp_path)
    rep = A.analyze(records)
    assert counts["files"] == 0 and rep["requests"]["total"] == 0 and rep["slo"]["ttft"]["met"] is None
    assert "Requests 0" in A.render(rep, "folder x", counts)


def test_cli_markdown_and_unique_json(logs, tmp_path, capsys):
    out = tmp_path / "out"
    for _ in range(2):
        A.main(["--dir", str(logs), "--json", "--json-dir", str(out)])
    md = capsys.readouterr().out
    assert "# Usage report: folder" in md and "| 2026-11-07 | 3 | 0 | 1 | 1 | 1 | 3 |" in md
    assert "| TTFT ≤ 2 s | 95% | 5 / 6 | 83.33% |" in md and "shown, not judged" in md
    files = sorted(out.iterdir())
    assert len(files) == 2                                  # never overwritten
    d = json.loads(files[0].read_text())
    assert d["requests"]["total"] == 10 and d["files"]["corrupt"] == 3
