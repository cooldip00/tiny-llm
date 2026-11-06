"""Step 4.13 step 3: load generator for any OpenAI-style /v1/completions server.

The same client drives our server (`python -m tinyllm.server`) and vLLM (4.8a), so their
numbers compare directly.

For each request rate: Poisson arrivals (exponential gaps, fixed seed) for `--duration`
seconds. Every request streams, ignores EOS and asks for exactly `--max-tokens`, so the work
is the same at every rate and on every server. Then the next rate.

Measured on the client (SLOs in docs/SERVING.md are server-side; this is the cross-check):
- TTFT: request sent → first content chunk. Includes queueing.
- inter-token latency (ITL): gaps between content chunks. A server that holds back a partial
  character sends two tokens in one chunk; the token count comes from `usage`.
- latency: sent → last content chunk.
- throughput: completion tokens / wall time (first send → last completion).
- 429s (shed load) and failures (status >= 500, error chunks, connection errors).

`--sequential N` instead sends N requests one after another (one in flight, closed loop): the
single-request (B=1) decode speed through the HTTP path, for comparing engines in 4.8a. Decode
tok/s per request = (tokens - 1) / (last chunk - first chunk), so prefill and queueing are out
of it. With `--weight-bytes`, it is also shown as a % of the bandwidth ceiling (423 GB/s /
weight bytes, docs/HARDWARE.md).

Rated load (goodput) = the highest rate at which the run, and every lower rate, meets all
three SLOs: TTFT p95 <= target, availability >= 99%, 429s <= 1%. Server-side queue wait is in
the server's request log and `/stats`.

    uv run python scripts/load_test.py --url http://127.0.0.1:8000 --rates 1,2,4,8,16 \\
        --duration 60 --max-tokens 128 --slo local --power-mode Performance --label ours-8slots
    uv run python scripts/load_test.py --sequential 20 --max-tokens 256 --weight-bytes 2.2e9 \\
        --power-mode Performance --label ours-b1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from tinyllm.obs import percentile

SLO_TTFT_P95_S = {"local": 1.0, "space": 2.0}      # docs/SERVING.md, "SLOs"
SLO_AVAILABILITY = 0.99
SLO_SHED_MAX = 0.01
BANDWIDTH = 423e9           # bytes/s, Phase 0.5 measurement (docs/HARDWARE.md)

PROMPTS = [
    "Once upon a time, there was a little dog named Max.",
    "Lily went to the park with her mom. She saw",
    "The sun was hot and the sky was blue. Tom wanted to",
    "In a small house near the forest lived an old cat.",
    "One day, a boy named Sam found a shiny red ball under his bed.",
    "Mia had a big box of crayons. She liked to draw",
    "The little bird could not fly yet. Every morning it",
    "Ben and his sister were building a tall tower of blocks when",
]


@dataclass
class Result:
    sent_s: float                       # arrival offset within the rate's run
    status: int = 0                     # HTTP status; error chunks set their code; -1 = no response
    ttft_s: float | None = None
    latency_s: float | None = None
    tokens: int = 0
    itl_s: list[float] = field(default_factory=list)
    error: str = ""


def poisson_arrivals(rate: float, duration: float, rng: random.Random) -> list[float]:
    """Arrival offsets in [0, duration) with exponential gaps of mean 1 / rate."""
    out, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return out
        out.append(t)


async def one_request(client: httpx.AsyncClient, body: dict, sent_s: float) -> Result:
    r = Result(sent_s)
    start = time.perf_counter()
    last, chunks, usage = None, 0, None
    try:
        async with client.stream("POST", "/v1/completions", json=body) as resp:
            r.status = resp.status_code
            if resp.status_code != 200:
                r.error = (await resp.aread()).decode(errors="replace")[:200]
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if "error" in d:
                    r.status, r.error = int(d["error"].get("code") or 500), str(d["error"].get("message"))[:200]
                    break
                if d.get("usage"):
                    usage = d["usage"].get("completion_tokens")
                if not d.get("choices"):            # vLLM's usage-only final chunk
                    continue
                now = time.perf_counter()
                if last is None:
                    r.ttft_s = now - start
                else:
                    r.itl_s.append(now - last)
                last, chunks = now, chunks + 1
    except httpx.HTTPError as e:
        r.status, r.error = -1, f"{type(e).__name__}: {e}"[:200]
        return r
    if last is not None:
        r.latency_s = last - start
    r.tokens = usage if usage is not None else chunks
    return r


def summarize(results: list[Result], rate: float, wall_s: float, ttft_slo_s: float) -> dict:
    n = len(results)
    ok = [r for r in results if r.status == 200]
    shed = sum(r.status == 429 for r in results)
    failed = sum(r.status >= 500 or r.status < 0 for r in results)
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    itl = [x for r in ok for x in r.itl_s]
    lat = [r.latency_s for r in ok if r.latency_s is not None]
    tokens = sum(r.tokens for r in ok)
    rnd = lambda x: None if x is None else round(x, 4)       # noqa: E731
    s = {
        "rate": rate, "sent": n, "ok": len(ok), "shed_429": shed, "failed": failed,
        "other": n - len(ok) - shed - failed, "wall_s": round(wall_s, 3),
        "achieved_rps": round(len(ok) / wall_s, 3) if wall_s else None,
        "tok_s": round(tokens / wall_s, 1) if wall_s else None, "completion_tokens": tokens,
        "ttft_s": {f"p{q}": rnd(percentile(ttft, q)) for q in (50, 95, 99)},
        "itl_s": {f"p{q}": rnd(percentile(itl, q)) for q in (50, 95, 99)},
        "latency_s": {f"p{q}": rnd(percentile(lat, q)) for q in (50, 95, 99)},
        "availability": round(1 - failed / n, 4) if n else None,
        "shed_rate": round(shed / n, 4) if n else None,
        "errors": sorted({r.error for r in results if r.error})[:5],
    }
    p95 = s["ttft_s"]["p95"]
    s["slo"] = {"ttft_p95": p95 is not None and p95 <= ttft_slo_s,
                "availability": s["availability"] is not None and s["availability"] >= SLO_AVAILABILITY,
                "shed": s["shed_rate"] is not None and s["shed_rate"] <= SLO_SHED_MAX}
    s["meets_slo"] = n > 0 and all(s["slo"].values())
    return s


def summarize_sequential(results: list[Result], wall_s: float, weight_bytes: float | None) -> dict:
    """One request at a time: per-request decode speed, excluding prefill (first chunk)."""
    ok = [r for r in results if r.status == 200 and r.ttft_s is not None and r.latency_s is not None]
    decode = [(r.tokens - 1) / (r.latency_s - r.ttft_s) for r in ok if r.tokens > 1 and r.latency_s > r.ttft_s]
    rnd = lambda x: None if x is None else round(x, 4)       # noqa: E731
    s = {
        "sent": len(results), "ok": len(ok), "wall_s": round(wall_s, 3),
        "tokens": sorted({r.tokens for r in ok}),             # should be just max_tokens
        "decode_tok_s": {f"p{q}": rnd(percentile(decode, q)) for q in (10, 50, 90)},
        "ttft_s": {f"p{q}": rnd(percentile([r.ttft_s for r in ok], q)) for q in (50, 95)},
        "itl_s": {f"p{q}": rnd(percentile([x for r in ok for x in r.itl_s], q)) for q in (50, 99)},
        "errors": sorted({r.error for r in results if r.error})[:5],
    }
    if weight_bytes and decode:
        s["ceiling_tok_s"] = round(BANDWIDTH / weight_bytes, 1)
        s["pct_of_ceiling"] = round(100 * s["decode_tok_s"]["p50"] / s["ceiling_tok_s"], 1)
    return s


def rated_load(summaries: list[dict]) -> float | None:
    """Highest rate such that it and every lower rate meet all SLOs (None if the lowest fails)."""
    best = None
    for s in sorted(summaries, key=lambda s: s["rate"]):
        if not s["meets_slo"]:
            break
        best = s["rate"]
    return best


async def run_rate(client: httpx.AsyncClient, body_for, rate: float, duration: float,
                   rng: random.Random) -> tuple[list[Result], float]:
    arrivals = poisson_arrivals(rate, duration, rng)
    t0 = time.perf_counter()

    async def fire(i: int, at: float) -> Result:
        await asyncio.sleep(max(0.0, at - (time.perf_counter() - t0)))
        return await one_request(client, body_for(i), at)

    results = await asyncio.gather(*(fire(i, at) for i, at in enumerate(arrivals)))
    return list(results), time.perf_counter() - t0


async def run(args) -> dict:
    prompts = (Path(args.prompts).read_text().splitlines() if args.prompts else PROMPTS)
    prompts = [p for p in prompts if p.strip()]
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, limits=limits) as client:
        model = args.model
        if model is None:
            model = (await client.get("/v1/models")).json()["data"][0]["id"]

        def body_for(i: int) -> dict:
            return {"model": model, "prompt": prompts[i % len(prompts)], "max_tokens": args.max_tokens,
                    "temperature": args.temperature, "ignore_eos": True, "stream": True,
                    "stream_options": {"include_usage": True}}

        for i in range(args.warmup):                          # compile / capture / first-touch costs
            w = await one_request(client, body_for(i), 0.0)
            if w.status != 200:
                raise SystemExit(f"warmup request failed: {w.status} {w.error}")
        if args.sequential:
            t0 = time.perf_counter()
            results = [await one_request(client, body_for(i), time.perf_counter() - t0)
                       for i in range(args.sequential)]
            s = summarize_sequential(results, time.perf_counter() - t0, args.weight_bytes)
            d = s["decode_tok_s"]
            print(f"sequential x{s['sent']} | ok {s['ok']} tokens {s['tokens']} | decode tok/s p10 "
                  f"{d['p10']} p50 {d['p50']} p90 {d['p90']} | TTFT p50 {_ms(s['ttft_s']['p50'])} | "
                  f"ITL p50 {_ms(s['itl_s']['p50'])} p99 {_ms(s['itl_s']['p99'])}"
                  + (f" | {s['pct_of_ceiling']}% of {s['ceiling_tok_s']} tok/s ceiling"
                     if "pct_of_ceiling" in s else ""), flush=True)
            return {"model": model, "sequential": s,
                    "requests": [asdict(r) for r in results] if args.keep_requests else None}
        rng = random.Random(args.seed)
        ttft_slo = SLO_TTFT_P95_S[args.slo]
        rows = []
        for rate in args.rates:
            results, wall = await run_rate(client, body_for, rate, args.duration, rng)
            s = summarize(results, rate, wall, ttft_slo)
            rows.append({"summary": s, "requests": [asdict(r) for r in results] if args.keep_requests else None})
            t, it, la = s["ttft_s"], s["itl_s"], s["latency_s"]
            print(f"rate {rate:6.2f}/s | sent {s['sent']:4d} ok {s['ok']:4d} 429 {s['shed_429']:3d} "
                  f"fail {s['failed']:3d} | {s['achieved_rps']:6.2f} req/s {s['tok_s']:8.1f} tok/s | "
                  f"TTFT p50 {_ms(t['p50'])} p95 {_ms(t['p95'])} p99 {_ms(t['p99'])} | ITL p50 {_ms(it['p50'])} "
                  f"p99 {_ms(it['p99'])} | lat p99 {_ms(la['p99'])} | SLO {'ok' if s['meets_slo'] else 'MISS'} "
                  f"{'' if s['meets_slo'] else [k for k, v in s['slo'].items() if not v]}", flush=True)
            if args.pause:
                await asyncio.sleep(args.pause)
    summaries = [r["summary"] for r in rows]
    return {"model": model, "rated_load_rps": rated_load(summaries), "rows": rows}


def _ms(x: float | None) -> str:
    return "    -" if x is None else f"{1e3 * x:5.0f}ms"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Poisson load test for /v1/completions (ours or vLLM)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="model name in requests; default: first of /v1/models")
    ap.add_argument("--rates", type=lambda s: [float(x) for x in s.split(",")], default=[1, 2, 4, 8],
                    help="requests/s, comma-separated")
    ap.add_argument("--sequential", type=int, default=0, metavar="N",
                    help="instead of the rate sweep: N requests one at a time (B=1 decode speed)")
    ap.add_argument("--weight-bytes", type=float, default=None,
                    help="stored weight bytes, for the %% of the bandwidth ceiling in --sequential")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals per rate")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between rates")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slo", choices=sorted(SLO_TTFT_P95_S), default="local")
    ap.add_argument("--power-mode", required=True, help="NitroSense mode for the record, e.g. Performance")
    ap.add_argument("--label", default="run", help="names the output file")
    ap.add_argument("--keep-requests", action="store_true", help="store every request in the JSON")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/bench/load/<label>.json")
    args = ap.parse_args(argv)

    from tinyllm.provenance import Provenance                 # imports torch: keep it out of the client path
    prov = Provenance(power_mode=args.power_mode, url=args.url).start()
    out = asyncio.run(run(args))
    if not args.sequential:
        print(f"rated load (all SLOs met at it and below): {out['rated_load_rps']} req/s")
    record = {"workload": {k: getattr(args, k) for k in ("rates", "duration", "max_tokens", "temperature",
                                                         "warmup", "seed", "slo", "prompts", "sequential",
                                                         "weight_bytes")},
              "slo": {"ttft_p95_s": SLO_TTFT_P95_S[args.slo], "availability": SLO_AVAILABILITY,
                      "shed_max": SLO_SHED_MAX},
              "provenance": prov.finish(), **out}
    path = args.out or Path("runs/bench/load") / f"{args.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
