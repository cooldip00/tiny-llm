"""6.5b step 4: usage report from the shipped server logs (src/tinyllm/logship.py).

Reads the hourly JSONL files from a local folder (`--dir`, e.g. a server's `--log-dir`) or from the
private HF dataset (`--dataset`, downloaded into the HF cache with your `hf auth login`), and prints
a markdown report: period, restarts and uptime, requests per day by status, unique visitors per
day, TTFT / latency percentiles, tokens, finish reasons, the SLOs and their error budget over the
period (same maths and targets as the status page, src/tinyllm/obs.py), errors grouped by message,
the busiest hours (UTC) and queue / slot use from the engine snapshots (input for sizing, D-031).

With `--json`, the report also goes to a new file under runs/logs/ (never overwritten).
Corrupt lines and unknown messages are skipped and counted, never fatal.

    uv run python scripts/analyze_logs.py --dataset CoolDip00/tiny-llm-space-logs --days 7 --json
    uv run python scripts/analyze_logs.py --dir /tmp/tinyllm-logs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyllm.obs import SLO_TARGETS, _SloBucket, percentile, slo_counted  # noqa: E402

REQUEST = ("request", "slow request")
KNOWN = {*REQUEST, "engine", "model loaded", "server ready", "shutting down: draining", "stopped"}
DEFAULT_TTFT_S = 2.0          # the Space's threshold (Dockerfile SLO_TTFT_S) when no ready line says


def utc(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc)


def read_logs(root: Path) -> tuple[list[dict], dict]:
    """All lines of the *.jsonl files under `root`, each tagged with its run (start-replica, from
    the file name). Returns (records, counts of files / lines / corrupt)."""
    records, n = [], {"files": 0, "lines": 0, "corrupt": 0}
    for p in sorted(root.rglob("*.jsonl")):
        n["files"] += 1
        run = p.stem.rsplit("-", 1)[0]
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            n["lines"] += 1
            try:
                r = json.loads(line)
                if not isinstance(r, dict) or not isinstance(r.get("ts"), (int, float)):
                    raise ValueError("no ts")
            except ValueError:
                n["corrupt"] += 1
                continue
            r["_run"] = run
            records.append(r)
    records.sort(key=lambda r: r["ts"])
    return records, n


def _q(xs) -> dict:
    xs = [x for x in xs if isinstance(x, (int, float))]
    out = {f"p{q}": (round(v, 4) if (v := percentile(xs, q)) is not None else None) for q in (50, 95, 99)}
    return {"n": len(xs), **out}


def analyze(records: list[dict], since: float | None = None, until: float | None = None,
            slo_ttft_s: float | None = None) -> dict:
    recs = [r for r in records if (since is None or r["ts"] >= since) and (until is None or r["ts"] < until)]
    unknown = Counter(r.get("msg") for r in recs if r.get("msg") not in KNOWN and r.get("level") != "error")
    reqs = [r for r in recs if r.get("msg") in REQUEST and isinstance(r.get("status"), int)]
    snaps = [r for r in recs if r.get("msg") == "engine"]
    ready = [r for r in recs if r.get("msg") == "server ready"]
    if slo_ttft_s is None:
        slo_ttft_s = next((r["slo_ttft_s"] for r in reversed(ready) if "slo_ttft_s" in r), DEFAULT_TTFT_S)

    # runs: one per server start (file name), uptime = first to last line seen
    runs = defaultdict(list)
    for r in recs:
        runs[r["_run"]].append(r)
    run_info = {k: {"first": v[0]["ts"], "last": v[-1]["ts"], "stopped": any(x.get("msg") == "stopped" for x in v)}
                for k, v in sorted(runs.items())}

    per_day = defaultdict(Counter)
    visitors = defaultdict(set)
    for r in reqs:
        day = utc(r["ts"]).date().isoformat()
        per_day[day][str(r["status"])] += 1
        if r.get("client"):
            visitors[day].add(r["client"])
    ok = [r for r in reqs if r["status"] == 200]

    bucket = _SloBucket()
    for r in reqs:
        if slo_counted(r["status"]):
            bucket.add(SimpleNamespace(status=r["status"], ttft_s=r.get("ttft_s")), slo_ttft_s)
    slos = {}
    for name, target in SLO_TARGETS.items():
        bad, total = bucket.bad_and_total(name)
        slos[name] = {"target": target, "good": total - bad, "total": total,
                      "compliance": round((total - bad) / total, 6) if total else None,
                      "met": (total - bad) / total >= target if total else None,
                      "budget_left": round(1 - bad / ((1 - target) * total), 6) if total else None}
    slos["ttft"]["threshold_s"] = slo_ttft_s
    slos["shed"]["judged"] = False

    errors = Counter((r["status"], r.get("error")) for r in reqs if r.get("error"))
    engine_errors = Counter(r.get("msg") for r in recs if r.get("level") == "error")
    hour_of_day = Counter(utc(r["ts"]).hour for r in reqs)
    hours = Counter(utc(r["ts"]).strftime("%Y-%m-%d %H:00") for r in reqs)
    busy = [s.get("busy_slots") for s in snaps if isinstance(s.get("busy_slots"), int)]
    n_slots = max((s.get("n_slots") or 0 for s in snaps), default=None) or None
    return {
        "period": {"first": utc(recs[0]["ts"]).isoformat() if recs else None,
                   "last": utc(recs[-1]["ts"]).isoformat() if recs else None,
                   "since": utc(since).isoformat() if since is not None else None},
        "unknown_lines": dict(unknown),
        "runs": {"starts": len(ready), "runs": len(run_info),
                 "clean_stops": sum(v["stopped"] for v in run_info.values()),
                 "uptime_h": round(sum(v["last"] - v["first"] for v in run_info.values()) / 3600, 3)},
        "requests": {"total": len(reqs), "by_status": dict(Counter(str(r["status"]) for r in reqs)),
                     "per_day": {d: dict(c) for d, c in sorted(per_day.items())}},
        "visitors": {"per_day": {d: len(v) for d, v in sorted(visitors.items())},
                     "total": len(set().union(*visitors.values())) if visitors else 0},
        "ttft_s": _q(r.get("ttft_s") for r in ok),
        "latency_s": _q(r.get("latency_s") for r in ok),
        "queue_wait_s": _q(r.get("queue_wait_s") for r in ok),
        "tokens": {"prompt": sum(r.get("prompt_tokens") or 0 for r in ok),
                   "completion": sum(r.get("completion_tokens") or 0 for r in ok),
                   "completion_per_request": round(sum(r.get("completion_tokens") or 0 for r in ok) / len(ok), 2)
                   if ok else None},
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in reqs)),
        "stream_share": round(sum(bool(r.get("stream")) for r in ok) / len(ok), 4) if ok else None,
        "slo": slos,
        "errors": [{"status": s, "error": e, "count": c} for (s, e), c in errors.most_common()],
        "engine_errors": dict(engine_errors),
        "busiest_hours_utc": [{"hour": h, "requests": c} for h, c in hour_of_day.most_common(5)],
        "peak_hour": ({"hour": hours.most_common(1)[0][0], "requests": hours.most_common(1)[0][1]}
                      if hours else None),
        "snapshots": {"n": len(snaps), "n_slots": n_slots,
                      "queue_len": {**_q(s.get("queue_len") for s in snaps),
                                    "max": max((s.get("queue_len") or 0 for s in snaps), default=None)},
                      "busy_slots_mean": round(sum(busy) / len(busy), 3) if busy else None,
                      "busy_slots_max": max(busy, default=None),
                      "all_slots_busy_share": round(sum(b == n_slots for b in busy) / len(busy), 4)
                      if busy and n_slots else None,
                      "tokens_per_s_max": max((s.get("tokens_per_s") or 0 for s in snaps), default=None)},
    }


def _f(x, unit="") -> str:
    return "–" if x is None else f"{x:g}{unit}"


def render(rep: dict, source: str, counts: dict) -> str:
    p, rq, s = rep["period"], rep["requests"], rep["slo"]
    L = [f"# Usage report: {source}", "",
         f"- Period: {p['first'] or '–'} → {p['last'] or '–'}" + (f" (since {p['since']})" if p["since"] else ""),
         f"- Files {counts['files']}, lines {counts['lines']}, corrupt {counts['corrupt']}, "
         f"unknown {sum(rep['unknown_lines'].values())}",
         f"- Server starts {rep['runs']['starts']} ({rep['runs']['clean_stops']} clean stops), "
         f"uptime seen {rep['runs']['uptime_h']:g} h",
         f"- Requests {rq['total']}, unique visitors {rep['visitors']['total']} "
         "(across days only with a fixed TINYLLM_LOG_SALT)", ""]
    statuses = sorted({k for c in rq["per_day"].values() for k in c})
    if statuses:
        L += ["## Requests per day (UTC)", "", "| Day | " + " | ".join(statuses) + " | Visitors |",
              "|---|" + "---:|" * (len(statuses) + 1)]
        for d, c in rq["per_day"].items():
            L.append(f"| {d} | " + " | ".join(str(c.get(k, 0)) for k in statuses)
                     + f" | {rep['visitors']['per_day'].get(d, 0)} |")
        L.append("")
    L += ["## Latency (served requests)", "", "| | n | p50 | p95 | p99 |", "|---|---:|---:|---:|---:|"]
    for k, name in (("ttft_s", "TTFT"), ("latency_s", "Whole request"), ("queue_wait_s", "Queue wait")):
        q = rep[k]
        L.append(f"| {name} | {q['n']} | {_f(q['p50'], ' s')} | {_f(q['p95'], ' s')} | {_f(q['p99'], ' s')} |")
    t = rep["tokens"]
    L += ["", f"Tokens: prompt {t['prompt']}, completion {t['completion']} "
              f"({_f(t['completion_per_request'])} per request); streamed share {_f(rep['stream_share'])}.",
          "Finish reasons: " + (", ".join(f"{k} {v}" for k, v in sorted(rep["finish_reasons"].items())) or "–"), "",
          "## SLOs over the period", "", "| SLO | Target | Good / total | Compliance | Budget left | |",
          "|---|---:|---:|---:|---:|---|"]
    names = {"ttft": f"TTFT ≤ {s['ttft']['threshold_s']:g} s", "availability": "Availability",
             "shed": "Not shed (429)"}
    for k, v in s.items():
        verdict = "shown, not judged" if v.get("judged") is False else (
            "–" if v["met"] is None else ("met" if v["met"] else "missed"))
        comp = "–" if v["compliance"] is None else f"{v['compliance']:.2%}"
        left = "–" if v["budget_left"] is None else f"{v['budget_left']:.0%}"
        L.append(f"| {names[k]} | {v['target']:.0%} | {v['good']} / {v['total']} | {comp} | {left} | {verdict} |")
    if rep["errors"] or rep["engine_errors"]:
        L += ["", "## Errors", "", "| Status | Message | Count |", "|---:|---|---:|"]
        L += [f"| {e['status']} | {e['error']} | {e['count']} |" for e in rep["errors"]]
        L += [f"| engine | {m} | {c} |" for m, c in rep["engine_errors"].items()]
    if rep["busiest_hours_utc"]:
        L += ["", "## Busiest hours (UTC, all days)", "",
              ", ".join(f"{h['hour']:02d}:00 ({h['requests']})" for h in rep["busiest_hours_utc"])
              + f". Peak single hour: {rep['peak_hour']['hour']} ({rep['peak_hour']['requests']} requests)."]
    sn = rep["snapshots"]
    if sn["n"]:
        L += ["", "## Queue and slots (engine snapshots, every 30 s)", "",
              f"{sn['n']} snapshots, {_f(sn['n_slots'])} slots: busy mean {_f(sn['busy_slots_mean'])}, "
              f"max {_f(sn['busy_slots_max'])}; all slots busy {_f(sn['all_slots_busy_share'])} of the time; "
              f"queue p95 {_f(sn['queue_len']['p95'])}, max {_f(sn['queue_len']['max'])}; "
              f"peak {_f(sn['tokens_per_s_max'])} tok/s."]
    return "\n".join(L) + "\n"


def json_path(out_dir: Path) -> Path:
    """runs/logs/report_<UTC now>[_<n>].json: a new file every time."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    p, n = out_dir / f"report_{stamp}.json", 2
    while p.exists():
        p, n = out_dir / f"report_{stamp}_{n}.json", n + 1
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path, help="local log folder (a server's --log-dir)")
    src.add_argument("--dataset", metavar="REPO_ID", help="the private HF dataset the server uploads to")
    when = ap.add_mutually_exclusive_group()
    when.add_argument("--since", help="UTC date YYYY-MM-DD")
    when.add_argument("--days", type=int, help="the last N days (UTC midnight N days ago until now)")
    ap.add_argument("--slo-ttft-s", type=float, default=None,
                    help="TTFT threshold; default from the latest 'server ready' line, else 2 s")
    ap.add_argument("--json", action="store_true", help="also write the report to a new file in --json-dir")
    ap.add_argument("--json-dir", type=Path, default=Path("runs/logs"))
    args = ap.parse_args(argv)

    since = None
    if args.since:
        since = dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.timezone.utc).timestamp()
    elif args.days is not None:
        today = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        since = (today - dt.timedelta(days=args.days)).timestamp()

    if args.dataset:
        from huggingface_hub import snapshot_download
        patterns = ["logs/**"]
        if since is not None:              # only the day folders in the period
            d, end = utc(since).date(), dt.datetime.now(dt.timezone.utc).date()
            patterns = []
            while d <= end:
                patterns.append(f"logs/{d.isoformat()}/*")
                d += dt.timedelta(days=1)
        root = Path(snapshot_download(args.dataset, repo_type="dataset", allow_patterns=patterns))
        source = f"dataset {args.dataset}"
    else:
        root, source = args.dir, f"folder {args.dir}"
    records, counts = read_logs(root)
    rep = analyze(records, since=since, slo_ttft_s=args.slo_ttft_s)
    print(render(rep, source, counts), end="")
    if args.json:
        args.json_dir.mkdir(parents=True, exist_ok=True)
        path = json_path(args.json_dir)
        path.write_text(json.dumps({"source": source, "files": counts, "generated_utc":
                                    utc(time.time()).isoformat(), **rep}, indent=2) + "\n")
        print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
