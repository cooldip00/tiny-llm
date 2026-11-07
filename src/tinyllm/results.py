"""Step 6.8 (D-029): the results tables, rendered from one benchmark run's JSON files.

scripts/bench_all.py runs every benchmark in BENCHES into one directory (runs/bench/v1b/), then calls
render_results() for docs/RESULTS.md and render_readme() for the README's <!-- bench:NAME --> blocks.
Everything here is pure (JSON in, markdown out), so the CPU tests cover it without a GPU.

Rules this module enforces:
  - every table is built from one run directory: nothing is mixed in from other sessions
  - a missing file is an error naming the `--only` command that makes it, never an empty table
  - each table carries a heat line from its files' provenance, since absolute tok/s on this laptop
    moves 10-20% between sessions with temperature
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# A sustained clock below this, or a hardware slowdown, means the numbers are not this GPU's normal
# (11-04 runs: busy median ~1,600 MHz at 85 °C, with sw_power_cap / sw_thermal_slowdown only).
MIN_CLOCK_MHZ = 1400
HARD_THROTTLE = {"hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
PAGED_ROW = ("paged/16", 32)        # the README's paged row: 16-token blocks, 32 slots, the 8-slot budget


@dataclass(frozen=True)
class Bench:
    name: str
    script: str
    args: tuple[str, ...]
    out: str                         # path inside the run directory
    minutes: int                     # rough, for the run plan printout
    env: tuple[tuple[str, str], ...] = ()   # extra environment for this benchmark, recorded in the manifest

    @property
    def out_dir_flag(self) -> bool:  # profile_decode takes a directory, the others a file
        return self.script == "profile_decode.py"


TL = ("--model", "tinyllama")
# The naive engine runs a full forward per token, so its activations and logits grow by one position every
# step. No cached block fits the next request, the caching allocator keeps reserving new ones until the GPU
# is full (7,991 MiB in use for a 475 MiB peak at 30M B=32 P=128), and a cuBLAS call then fails
# (6.8, 2026-11-06). Expandable segments let a block grow in place: 695 MiB, same speed. Only the naive
# benchmarks need it; the other engines use fixed-size caches.
GROWING = (("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),)
BENCHES = [
    Bench("30m-naive", "bench_generate.py", ("--engine", "naive", "--dtype", "bf16"), "30m/naive.json", 4, GROWING),
    Bench("30m-kv", "bench_generate.py", ("--engine", "kv", "--dtype", "bf16"), "30m/kv.json", 2),
    Bench("30m-kv-graph", "bench_generate.py", ("--engine", "kv-graph", "--dtype", "bf16"), "30m/kv-graph.json", 2),
    Bench("30m-serving", "bench_serving.py", ("--cuda-graph", "--paged"), "30m/serving.json", 12),
    Bench("30m-int8", "bench_quant.py", (), "30m/int8.json", 4),
    Bench("30m-profile", "profile_decode.py", ("--batch", "1"), "30m/profile/decode_main_B1.json", 1),
    Bench("tl-logits", "tinyllama_vs_hf.py", (), "tinyllama/logits_vs_hf.json", 4),
    Bench("tl-hf", "bench_generate.py", (*TL, "--engine", "hf"), "tinyllama/hf.json", 12),
    Bench("tl-naive", "bench_generate.py", (*TL, "--engine", "naive"), "tinyllama/naive.json", 6, GROWING),
    Bench("tl-kv", "bench_generate.py", (*TL, "--engine", "kv"), "tinyllama/kv.json", 10),
    Bench("tl-kv-graph", "bench_generate.py", (*TL, "--engine", "kv-graph"), "tinyllama/kv-graph.json", 6),
    Bench("tl-serving", "bench_serving.py", (*TL, "--cuda-graph"), "tinyllama/serving.json", 15),
    Bench("tl-int8", "bench_quant.py", TL, "tinyllama/int8.json", 10),
    # 4.10b: the Triton INT8 kernel, end to end with CUDA graphs, and its matmuls alone
    Bench("tl-int8-graph", "bench_quant.py", (*TL, "--engine", "kv-graph", "--triton"),
          "tinyllama/int8_graph_triton.json", 12),
    Bench("tl-int8-gemv", "bench_int8_gemv.py", (), "tinyllama/int8_gemv.json", 3),
    # 4.10a: the Triton RMSNorm + residual add, alone and end to end with CUDA graphs
    Bench("tl-rmsnorm", "bench_rmsnorm.py", (), "tinyllama/rmsnorm_graph.json", 10),
    Bench("tl-profile", "profile_decode.py", (*TL, "--batch", "1"), "tinyllama/profile/decode_tinyllama_B1.json", 2),
]
BY_NAME = {b.name: b for b in BENCHES}
MANIFEST = "manifest.json"
# Optional results run by hand, read when present (None otherwise). 4.10c: tl-logits with our RMSNorm forced
# to the PyTorch code (tinyllama_vs_hf.py --norm-backend torch), which shows where D-032's logit gap comes from.
EXTRAS = {"tl-logits-torchnorm": "tinyllama/logits_vs_hf_torchnorm.json"}


class MissingResult(Exception):
    pass


def missing(run_dir: str | Path) -> list[Bench]:
    return [b for b in BENCHES if not (Path(run_dir) / b.out).exists()]


def load_run(run_dir: str | Path) -> dict:
    """{bench name: its JSON, "manifest": manifest or {}}; raises MissingResult listing every gap."""
    run_dir = Path(run_dir)
    gaps = missing(run_dir)
    if gaps:
        lines = "\n".join(f"  {run_dir / b.out}  ->  scripts/bench_all.py --only {b.name}" for b in gaps)
        raise MissingResult(f"{len(gaps)} result file(s) missing from {run_dir}:\n{lines}")
    data = {b.name: json.loads((run_dir / b.out).read_text()) for b in BENCHES}
    for name, out in EXTRAS.items():
        data[name] = json.loads((run_dir / out).read_text()) if (run_dir / out).exists() else None
    m = run_dir / MANIFEST
    data["manifest"] = json.loads(m.read_text()) if m.exists() else {}
    return data


# ---------------------------------------------------------------- formatting


def n(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def row(rows: list[dict], **kv) -> dict | None:
    return next((r for r in rows if all(r.get(k) == v for k, v in kv.items())), None)


def need(rows: list[dict], what: str, **kv) -> dict:
    r = row(rows, **kv)
    if r is None or "skipped" in r:
        raise MissingResult(f"{what}: no row with {kv}")
    return r


# ---------------------------------------------------------------- heat


def heat(prov: dict | None) -> dict | None:
    """Busy SM clock median / min, peak temperature, throttle reasons (gpu_idle dropped), warnings."""
    if not prov or not prov.get("during"):
        return None
    d = prov["during"]
    clock = d.get("clocks.sm_busy") or d.get("clocks.sm") or {}
    reasons = [r for r in d.get("throttle_reasons_seen") or [] if r != "gpu_idle"]
    warn = sorted(HARD_THROTTLE.intersection(reasons))
    # A low clock alone is not throttling: small or host-bound steps leave the GPU idle and it clocks down by
    # itself (30M KV decode: 1,095 MHz at 65 °C). Flag it only when the driver also reports a thermal slowdown.
    if (clock.get("median") is not None and clock["median"] < MIN_CLOCK_MHZ
            and "sw_thermal_slowdown" in reasons):
        warn.append(f"busy clock median {clock['median']:.0f} MHz < {MIN_CLOCK_MHZ}")
    return {"clock_median": clock.get("median"), "clock_min": clock.get("min"),
            "temp_max": (d.get("temperature.gpu") or {}).get("max"), "reasons": reasons, "warn": warn}


def heat_text(h: dict | None) -> str:
    if h is None:
        return "not recorded"
    parts = []
    if h["clock_median"] is not None:
        parts.append(f"{n(h['clock_median'])} MHz (min {n(h['clock_min'])})")
    if h["temp_max"] is not None:
        parts.append(f"{h['temp_max']:.0f} °C")
    parts.append(" + ".join(h["reasons"]) or "no throttling")
    return ("⚠ " if h["warn"] else "") + ", ".join(parts)


def heat_line(data: dict, names: list[str], labels: list[str] | None = None) -> str:
    labels = labels or names
    body = "; ".join(f"{lab}: {heat_text(heat(data[nm].get('provenance')))}" for nm, lab in zip(names, labels))
    return f"_Heat (busy SM clock median, peak GPU temperature, throttle reasons): {body}._"


def warnings(data: dict) -> list[str]:
    out = []
    for b in BENCHES:
        h = heat(data[b.name].get("provenance"))
        if h and h["warn"]:
            out.append(f"{b.name}: {', '.join(h['warn'])} -> consider scripts/bench_all.py --only {b.name}")
    return out


# ---------------------------------------------------------------- tables


def ceiling_note(d: dict) -> str:
    bw, wb = d["bandwidth_bytes_s"], d["weight_bytes"]
    return (f"{bw / 1e9:.0f} GB/s ÷ {n(wb / 2**20)} MiB = **{n(bw / wb)} tok/s per sequence**")


def repeats_note(d: dict) -> str:
    r = (d.get("provenance") or {}).get("repeats")
    return f"median of {r} runs" if r else "median of repeated runs"


TL_ENGINES = [("tl-hf", "HF Transformers `generate` (bf16, SDPA)"),
              ("tl-naive", "Ours, naive (full forward per token)"),
              ("tl-kv", "Ours, KV cache"),
              ("tl-kv-graph", "Ours, KV cache + CUDA graphs")]


def tl_b1_table(data: dict, prompt_len: int = 128) -> str:
    rows = []
    for name, label in TL_ENGINES:
        r = need(data[name]["rows"], name, batch=1, prompt_len=prompt_len)
        cells = [label, n(r["decode_tok_s"]), f"{r['pct_of_ceiling']:.0f}%", f"{r['ms_per_step']:.1f}"]
        rows.append([f"**{c}**" for c in cells] if name == "tl-kv-graph" else cells)
    return table(["Engine", "tok/s", "% of ceiling", "ms / step"], rows)


def decode_grid(data: dict, engines: list[tuple[str, str]]) -> str:
    plens = sorted({r["prompt_len"] for nm, _ in engines for r in data[nm]["rows"]})
    out, slides = [], False
    for name, label in engines:
        rows = data[name]["rows"]
        for b in sorted({r["batch"] for r in rows}):
            cells = [label, str(b)]
            for p in plens:
                r = row(rows, batch=b, prompt_len=p)
                if r is None or "skipped" in r:
                    cells.append("—")
                    continue
                mark = " ‡" if r.get("window_slides") else ""
                slides |= bool(mark)
                cells.append(f"{n(r['decode_tok_s'])} ({r['pct_of_ceiling']:.0f}%){mark}")
            out.append(cells)
    text = table(["Engine", "Batch", *[f"P={p}" for p in plens]], out)
    text += "\n\nDecode tok/s, with the % of the bandwidth ceiling at that batch size. — = skipped: " \
            "prompt + new tokens exceed the context, which a KV cache can't slide."
    if slides:
        text += " ‡ = prompt + new tokens exceed the context, so naive slides its window."
    return text


def int8_section(d: dict) -> str:
    base = d["baseline"]
    variants = list(dict.fromkeys(r["variant"] for r in d["rows"]))
    others = [v for v in variants if v != base]
    rows = []
    for b in sorted({r["batch"] for r in d["rows"]}):
        tok = {v: need(d["rows"], "int8", variant=v, batch=b)["decode_tok_s"] for v in variants}
        rows.append([str(b), *[n(tok[v]) for v in variants], *[f"{tok[v] / tok[base]:.2f}×" for v in others]])
    text = table(["Batch", *[f"{v} tok/s" for v in variants], *[f"{v} / {base}" for v in others]], rows)
    mib = {v: sum(d["weights_mib"][v].values()) for v in variants}
    loss, tf, same = d["val_loss"], d["teacher_forced"], d["greedy_rows_identical"]
    n_seq = same[base]
    text += "\n\n" + "\n".join([
        "- Weights: " + ", ".join(f"{v} {n(mib[v])} MiB" for v in variants) + ".",
        "- Val loss: " + ", ".join(f"{v} {loss[v]:.4f}" for v in variants)
        + " (" + ", ".join(f"{v} {loss[v] - loss[base]:+.4f}" for v in others) + f" vs {base}).",
        f"- Teacher-forced on {base}'s greedy tokens: argmax agreement "
        + ", ".join(f"{v} {tf[v]['argmax_agree']:.2%}" for v in variants)
        + f" (median top-1 margin {tf['_median_gap']:.2f}).",
        f"- Greedy rows identical to {base}: " + ", ".join(f"{v} {same[v]}/{n_seq}" for v in variants) + ".",
    ])
    return text


GEMV_VARIANTS = [("bf16", "bf16"), ("torch", "PyTorch int8"), ("triton", "Triton int8")]


def gemv_bandwidth(d: dict, m: int, v: str) -> tuple[float, float]:
    """(bytes one decode step's matmuls must read, % of the measured bandwidth) for variant v at m rows."""
    rows = [r for r in d["rows"] if r["m"] == m]
    nbytes = sum(r[f"{v}_gb_s"] * r[f"{v}_us"] * 1e3 * r["per_step"] for r in rows)    # GB/s × µs = kB
    return nbytes, 100 * nbytes / (d["step_us"][str(m)][v] * 1e-6) / d["bandwidth"]


def gemv_table(d: dict) -> str:
    ms = sorted(map(int, d["step_us"]))
    rows = []
    for m in ms:
        s = d["step_us"][str(m)]
        rows.append([str(m), *[f"{s[v] / 1e3:.2f}" for v, _ in GEMV_VARIANTS],
                     _ratio(s["bf16"] / s["triton"]), _ratio(s["torch"] / s["triton"], 1)])
    text = table(["Rows (batch)", *[f"{lab} ms" for _, lab in GEMV_VARIANTS], "Triton vs bf16",
                  "Triton vs PyTorch int8"], rows)
    (b_bf, p_bf), (b_tr, p_tr) = gemv_bandwidth(d, ms[0], "bf16"), gemv_bandwidth(d, ms[0], "triton")
    calls = sum(r["per_step"] for r in d["rows"] if r["m"] == ms[0])
    return text + (
        f"\n\nTime for every matmul of one decode step (the {calls} Linears), replayed in CUDA graphs over weight "
        f"copies larger than the L2 cache, so each call reads its weight from memory. At {ms[0]} row, a step must "
        f"read {n(b_bf / 2**20)} MiB of bf16 weights or {n(b_tr / 2**20)} MiB of int8 weights and scales: bf16 "
        f"runs at {p_bf:.0f}% of the measured {d['bandwidth'] / 1e9:.0f} GB/s, the Triton kernel at {p_tr:.0f}%. "
        "PyTorch int8 casts each weight back to bf16 first, so it moves more bytes than bf16, not fewer.")


def rmsnorm_section(d: dict) -> str:
    batches = sorted({r["batch"] for r in d["rows"]})
    rows = []
    for b in batches:
        t, k = need(d["rows"], "rmsnorm", variant="torch", batch=b), need(d["rows"], "rmsnorm", variant="triton", batch=b)
        rows.append([str(b), n(t["decode_tok_s"]), n(k["decode_tok_s"]), _ratio(k["decode_tok_s"] / t["decode_tok_s"]),
                     f"{t['ms_per_step']:.2f} → {k['ms_per_step']:.2f}"])
    text = table(["Batch", "PyTorch norm tok/s", "Triton norm tok/s", "Triton vs PyTorch", "ms per step"], rows)
    p, tf, alone = d["profile_b1"], d["teacher_forced"], d["kernels_alone"]
    fused = sorted((r for r in alone if r["op"] == "add+norm"), key=lambda r: r["rows"])
    if not fused or fused[0]["rows"] != 1:
        raise MissingResult("rmsnorm alone: no add+norm row with 1 row")
    one, big = fused[0], fused[-1]
    return text + (
        f"\n\nKV cache + CUDA graphs, the two norms alternating at each batch size. At batch 1 a decode step runs "
        f"{n(p['torch']['gpu_ops_per_step'])} GPU ops with the PyTorch norm and {n(p['triton']['gpu_ops_per_step'])} "
        f"with the Triton kernels (GPU busy {p['torch']['gpu_busy_ms_per_step']:.2f} → "
        f"{p['triton']['gpu_busy_ms_per_step']:.2f} ms per step). Residual add + norm alone: one row takes "
        f"{one['torch_us']:.1f} µs in PyTorch and {one['triton_us']:.1f} µs in Triton; {n(big['rows'])} rows run at "
        f"{big['triton_pct_bw']:.0f}% of the measured {d['bandwidth'] / 1e9:.0f} GB/s. Greedy rows identical to the "
        f"PyTorch norm: {d['greedy_rows_identical']}/{d['n_greedy_rows']}; teacher-forced argmax agreement "
        f"{tf['argmax_agree']:.2%}.")


def serving_baseline(rows: list[dict], r: dict) -> tuple[dict, str] | None:
    """Static has none; continuous vs static at the same size; graphs vs continuous; paged vs 8 contiguous slots."""
    e, s = r["engine"], r["size"]
    if e == "continuous":
        return row(rows, engine="static", size=s), "static"
    if e == "cont+graph":
        return row(rows, engine="continuous", size=s), "continuous"
    if e.startswith("paged/"):
        return row(rows, engine="continuous", size=8), "continuous 8 (same KV memory)"
    return None


def serving_table(d: dict) -> str:
    rows = []
    for r in d["rows"]:
        base = serving_baseline(d["rows"], r)
        vs = f"{r['tok_s'] / base[0]['tok_s']:.2f}× {base[1]}" if base and base[0] else "—"
        alloc = r.get("kv_alloc_used_mean")
        c = r.get("correctness") or {}
        rows.append([r["engine"], str(r["size"]), n(r["tok_s"]), vs,
                     f"{r['latency_mean_s']:.2f} / {r['latency_p95_s']:.2f}",
                     f"{r['ttft_mean_s']:.2f} / {r['ttft_p95_s']:.2f}",
                     f"{r['kv_reserved_mib']:.0f}", "—" if alloc is None else f"{alloc:.0%}",
                     str(r["peak_active"] or "—"), str(r["preemptions"]),
                     f"{c['identical']}/{c['of']}" if c else "—"])
    w = d["workload"]
    text = table(["Engine", "Slots", "tok/s", "vs", "Latency mean / p95 (s)", "TTFT mean / p95 (s)",
                  "KV MiB", "KV claimed in use", "Peak active", "Preempt", "Same as solo"], rows)
    reps = (d.get("provenance") or {}).get("repeats")
    text += (f"\n\n{w['requests']} requests, all queued at t=0 (seed {w['seed']}); "
             f"{'median-wall run of ' + str(reps) if reps else 'median run'}. \"Same as solo\": requests whose "
             "tokens equal solo `generate()`; bf16 near-ties flip some, so it is reported, not asserted.")
    return text


def logits_section(d: dict, torchnorm: dict | None = None) -> str:
    """`torchnorm`: the same run with our RMSNorm forced to PyTorch (EXTRAS); adds its rows, labelled by norm."""
    rows = []
    runs = [(d, "")] if torchnorm is None else [
        (d, f", {(d.get('provenance') or {}).get('norm_backend', '?')} norm"), (torchnorm, ", torch norm")]
    for (r, label), phase in [(x, p) for x in runs for p in ("prefill", "decode")]:
        s = r["stats"][phase]
        flips = s["flips"]["ours_bf16"]
        rows.append([phase + label, str(s["ours_vs_hf_bf16"]["positions"]),
                     "yes" if s["bit_identical_to_hf_bf16"] else "no",
                     f"{s['ours_vs_hf_bf16']['max_abs']:.3g}",
                     f"{s['ours_bf16']['top1']:.2%}", f"{s['hf_bf16']['top1']:.2%}",
                     ", ".join(f"{f:.3f}" for f in flips) or "none"])
    text = table(["Phase", "Positions", "Ours == HF bf16 (`torch.equal`)", "Max abs Δ, ours vs HF",
                  "Top-1 vs fp32: ours", "Top-1 vs fp32: HF", "Our margin at each fp32 flip"], rows)
    div = d["greedy_divergence"]
    same = sum(g["ours_vs_hf_bf16"] is None for g in div)
    at = lambda i: "never" if i is None else f"token {i}"                    # noqa: E731
    text += (f"\n\nGreedy, {len(div)} prompts × {d['new_tokens']} tokens: ours equals HF bf16 on {same}/{len(div)}. "
             "First divergence from the fp32 greedy text, ours / HF: "
             + "; ".join(f"{at(g['ours_vs_fp32'])} / {at(g['hf_bf16_vs_fp32'])}" for g in div) + ".")
    return text


def profile_table(profiles: list[tuple[str, dict]]) -> str:
    rows = []
    for label, d in profiles:
        for mode in ("eager", "graph"):
            m = d["modes"][mode]
            rows.append([label, mode, f"{m['ms_per_step']:.2f}", f"{m['host_issue_ms_per_step']:.2f}",
                         f"{m['gpu_busy_ms_per_step']:.2f}", f"{m['gpu_idle_frac']:.0%}",
                         n(m["decode_tok_s"]), f"{m['pct_of_ceiling']:.0f}%"])
    return table(["Model (B=1)", "Decode step", "ms / step", "Host issue ms", "GPU busy ms", "GPU idle",
                  "tok/s", "% of ceiling"], rows)


# ---------------------------------------------------------------- pages


def run_header(data: dict) -> str:
    m = data["manifest"]
    rec = m.get("benches") or {}
    # the Run line describes the main run only; --only reruns get their own line, grouped by session
    provs = [data[b.name].get("provenance") or {} for b in BENCHES if not (rec.get(b.name) or {}).get("rerun")]
    commits = sorted({(p.get("git") or {}).get("commit") or "?" for p in provs})
    dirty = any((p.get("git") or {}).get("dirty") for p in provs)
    first = next((p for p in provs if p.get("gpu_name")), {})
    times = sorted(p["time_utc"] for p in provs if p.get("time_utc"))
    commit = (f"commit `{commits[0][:7]}`" if len(commits) == 1 else
              f"⚠ {len(commits)} commits ({', '.join(c[:7] for c in commits)})")
    lines = [
        f"- **Run:** {times[0] if times else '?'} → {m.get('finished_utc', times[-1] if times else '?')} UTC, "
        f"{commit}{' with uncommitted changes' if dirty else ''}, NitroSense **{m.get('power_mode', 'not recorded')}**.",
        f"- **Machine:** {first.get('gpu_name', '?')}, driver {first.get('driver', '?')}, "
        f"torch {first.get('torch', '?')}, CUDA {first.get('cuda', '?')}.",
    ]
    sessions: dict[tuple, list[str]] = {}
    for n, r in sorted(rec.items(), key=lambda kv: str(kv[1].get("finished_utc", ""))):
        if r.get("rerun"):
            fin = str(r.get("finished_utc", "?"))
            key = (fin[:10], ((r.get("git") or {}).get("commit") or "?")[:7], r.get("power_mode") or "not recorded")
            sessions.setdefault(key, []).append(
                f"{n} (finished {fin[11:16]} UTC, started at {r.get('start_temp_c', '?')} °C)")
    if sessions:
        lines.append("- **Rerun later with `--only`:** " + "; ".join(
            f"{day}, commit `{c}`, NitroSense **{mode}**: {', '.join(names)}"
            for (day, c, mode), names in sessions.items()) + ". Every other benchmark is from the run above.")
    return "\n".join(lines)


def render_results(data: dict, run_dir: str = "runs/bench/v1b") -> str:
    tl_gen = [(nm, lab) for nm, lab in TL_ENGINES]
    w = warnings(data)
    parts = [
        "# Results",
        "",
        f"<!-- Generated by scripts/bench_all.py from {run_dir}/. Do not edit by hand: change the script "
        "(src/tinyllm/results.py) and re-render. -->",
        "",
        "Every number on this page comes from **one** run of `scripts/bench_all.py --run`: one commit, "
        "one session, one GPU job at a time, with a cool-down before each benchmark"
        + (" (a benchmark rerun later with `--only` is listed below)." if any(
            r.get("rerun") for r in (data["manifest"].get("benches") or {}).values()) else "."),
        "",
        run_header(data),
        "- **How to read it:** ratios between rows of one table are the reliable result; they share a "
        "session and its heat. Absolute tok/s on this laptop moves 10–20% between sessions with "
        "temperature, so each table ends with its heat line. ⚠ marks a hardware slowdown, or a thermal "
        f"slowdown with the busy clock median below {n(MIN_CLOCK_MHZ)} MHz.",
        "- **Elsewhere:** dated results and the reasoning behind each step are in "
        "[INFERENCE.md](INFERENCE.md); ours vs vLLM vs llama.cpp under load (4.8a, rerun in 4.10c) is in "
        "[SERVING.md](SERVING.md).",
        "- **Reproduce:** `uv run python scripts/bench_all.py` re-renders this page and the README "
        "tables from the saved JSON in seconds, with no GPU. `--run --power-mode <mode>` reruns "
        "everything (~1.5–2 h); `--only <bench>` reruns one benchmark.",
    ]
    if w:
        parts += ["", "**⚠ Throttling warnings:**", *[f"- {x}" for x in w]]
    hf = data["tl-hf"]
    parts += [
        "", "## TinyLlama 1.1B (bf16)", "",
        "### Decode at batch 1", "",
        f"128-token prompt, 256 new tokens, {repeats_note(hf)}. Ceiling: {ceiling_note(hf)}.", "",
        tl_b1_table(data), "", heat_line(data, [n_ for n_, _ in tl_gen], ["hf", "naive", "kv", "kv-graph"]),
        "", img("img/decode_b1.svg", b1_alt(data)),
        "", "### Decode by batch size and prompt length", "",
        decode_grid(data, tl_gen), "", img("img/batch_scaling.svg", scaling_alt(data)), "",
        "", "### INT8 weights (per-row scales)", "",
        "Two ways to run an int8 matmul: `int8` casts the weight back to bf16 on every call (plain PyTorch); "
        "`int8-triton` is our Triton kernel, which reads the int8 weight and dequantizes in registers (4.10b).",
        "", "#### Eager decode (KV cache, no graphs)", "",
        int8_section(data["tl-int8"]), "", heat_line(data, ["tl-int8"]),
        "", "#### With CUDA graphs: PyTorch path vs Triton kernel", "",
        "Each variant is timed with only its own weights on the GPU.", "",
        int8_section(data["tl-int8-graph"]), "", heat_line(data, ["tl-int8-graph"]),
        "", "#### The matmuls alone", "",
        gemv_table(data["tl-int8-gemv"]), "", heat_line(data, ["tl-int8-gemv"]),
        "", "### Fused RMSNorm + residual add (Triton, 4.10a)", "",
        rmsnorm_section(data["tl-rmsnorm"]), "", heat_line(data, ["tl-rmsnorm"]),
        "", "### Serving: static vs continuous batching, with and without CUDA graphs", "",
        serving_table(data["tl-serving"]), "", heat_line(data, ["tl-serving"]),
        "", "### Logits vs HF Transformers", "",
        "Ours (bf16) and HF Transformers (bf16, eager) against an fp32 HF reference, on the same prompts.",
        "", logits_section(data["tl-logits"], data.get("tl-logits-torchnorm")), "", heat_line(data, ["tl-logits"]),
        "", "## Our 30M model (bf16 weights)", "",
        "### Decode by batch size and prompt length", "",
        f"256 new tokens, {repeats_note(data['30m-kv'])}. Ceiling: {ceiling_note(data['30m-kv'])}.", "",
        decode_grid(data, [("30m-naive", "Naive"), ("30m-kv", "KV cache"), ("30m-kv-graph", "KV + CUDA graphs")]),
        "", heat_line(data, ["30m-naive", "30m-kv", "30m-kv-graph"], ["naive", "kv", "kv-graph"]),
        "", "### Serving: static, continuous, CUDA graphs, paged KV", "",
        "Paged rows: `paged/<block size>`, all within the 8-slot contiguous cache's KV memory.", "",
        serving_table(data["30m-serving"]), "", heat_line(data, ["30m-serving"]),
        "", img("img/serving.svg", serving_alt(data)),
        "", "### INT8 weights", "",
        int8_section(data["30m-int8"]), "", heat_line(data, ["30m-int8"]),
        "", "## Decode-step profile (batch 1, 128-token context)", "",
        profile_table([("TinyLlama 1.1B", data["tl-profile"]), ("30M", data["30m-profile"])]), "",
        heat_line(data, ["tl-profile", "30m-profile"], ["TinyLlama", "30M"]),
        "", img("img/decode_profile.svg", profile_alt(data)),
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)) + "\n"


# ---------------------------------------------------------------- README blocks


def _ratio(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}×"


# ---------------------------------------------------------------- plot embeds (6.2, D-030)
# The SVGs come from tinyllm.plots via bench_all.py; the alt text carries the plot's numbers.


def img(path: str, alt: str) -> str:
    return f"![{alt}]({path})"


def _tl(data: dict, nm: str, b: int) -> dict:
    return need(data[nm]["rows"], nm, batch=b, prompt_len=128)


def b1_alt(data: dict) -> str:
    hf = data["tl-hf"]
    return ("Bar chart, TinyLlama 1.1B decode at batch 1: " + "; ".join(
        f"{lab.split(' (')[0].replace('`', '')} {n(_tl(data, nm, 1)['decode_tok_s'])} tok/s" for nm, lab in TL_ENGINES)
        + f"; bandwidth ceiling {n(hf['bandwidth_bytes_s'] / hf['weight_bytes'])} tok/s")


def scaling_alt(data: dict) -> str:
    bs = sorted({r["batch"] for r in data["tl-kv-graph"]["rows"]} & {r["batch"] for r in data["tl-hf"]["rows"]})
    per = lambda nm: ", ".join(n(_tl(data, nm, b)["decode_tok_s"]) for b in bs)   # noqa: E731
    return (f"Line chart, decode tok/s at batch {', '.join(map(str, bs))}: ours with CUDA graphs {per('tl-kv-graph')}; "
            f"HF Transformers {per('tl-hf')}; log scales, with the bandwidth ceiling times batch size")


def serving_alt(data: dict) -> str:
    tls, ms = data["tl-serving"]["rows"], data["30m-serving"]["rows"]
    tok = lambda rows, nm, **kv: n(need(rows, nm, **kv)["tok_s"])                 # noqa: E731
    pg = need(ms, "30m-serving", engine=PAGED_ROW[0], size=PAGED_ROW[1])
    return ("Bar chart, serving tok/s. TinyLlama, 8 slots: static " + tok(tls, "tl-serving", engine="static", size=8)
            + ", continuous " + tok(tls, "tl-serving", engine="continuous", size=8)
            + ", continuous with CUDA graphs " + tok(tls, "tl-serving", engine="cont+graph", size=8)
            + ". 30M model, same KV memory: contiguous 8 sequences " + tok(ms, "30m-serving", engine="continuous", size=8)
            + f", paged {pg['peak_active']} sequences {n(pg['tok_s'])}")


def profile_alt(data: dict) -> str:
    m = data["tl-profile"]["modes"]
    return ("Bar chart, share of a TinyLlama decode step the GPU is idle: "
            f"eager {m['eager']['gpu_idle_frac']:.0%} ({m['eager']['ms_per_step']:.1f} ms per step), "
            f"CUDA graph {m['graph']['gpu_idle_frac']:.0%} ({m['graph']['ms_per_step']:.1f} ms per step)")


def readme_blocks(data: dict) -> dict[str, str]:
    g = {nm: data[nm]["rows"] for nm, _ in TL_ENGINES}
    at = lambda nm, b: need(g[nm], nm, batch=b, prompt_len=128)              # noqa: E731
    hf1, kv1, kvg1 = at("tl-hf", 1), at("tl-kv", 1), at("tl-kv-graph", 1)
    tls, ms = data["tl-serving"]["rows"], data["30m-serving"]["rows"]
    st8 = need(tls, "tl-serving", engine="static", size=8)
    co8 = need(tls, "tl-serving", engine="continuous", size=8)
    gr8 = need(tls, "tl-serving", engine="cont+graph", size=8)
    c8 = need(ms, "30m-serving", engine="continuous", size=8)
    pg = need(ms, "30m-serving", engine=PAGED_ROW[0], size=PAGED_ROW[1])
    lg = data["tl-logits"]["stats"]
    identical = lg["prefill"]["bit_identical_to_hf_bf16"] and lg["decode"]["bit_identical_to_hf_bf16"]
    max_d = max(lg[p]["ours_vs_hf_bf16"]["max_abs"] for p in ("prefill", "decode"))
    positions = sum(lg[p]["ours_vs_hf_bf16"]["positions"] for p in ("prefill", "decode"))
    flips = [f for p in ("prefill", "decode") for f in lg[p]["flips"]["ours_bf16"]]
    top1 = {p: lg[p]["ours_vs_hf_bf16"]["top1"] for p in ("prefill", "decode")}
    tn = (data.get("tl-logits-torchnorm") or {}).get("stats")
    torch_identical = bool(tn) and all(tn[p]["bit_identical_to_hf_bf16"] for p in ("prefill", "decode"))
    gap_cause = (" The difference is our Triton RMSNorm (D-032): with the PyTorch norm the logits are "
                 "bit-identical to HF's." if torch_identical else "")

    logits_hl = ("**Our engine's logits are bit-identical to HF Transformers** in bf16 on TinyLlama, "
                 "prefill and cached decode." if identical else
                 f"**Our engine's bf16 logits match HF Transformers to within {max_d:.2g}** on TinyLlama "
                 f"(same top token at {top1['prefill']:.1%} of prefill and {top1['decode']:.1%} of cached-decode "
                 "positions)." + gap_cause)
    highlights = "\n".join([
        img("docs/img/decode_b1.svg", b1_alt(data)),
        "",
        f"- **TinyLlama 1.1B decodes at {n(kvg1['decode_tok_s'])} tok/s at batch 1: "
        f"{kvg1['pct_of_ceiling']:.0f}% of the bandwidth ceiling, "
        f"{_ratio(kvg1['decode_tok_s'] / hf1['decode_tok_s'], 1)} HF `generate`** (KV cache + CUDA graphs).",
        f"- {logits_hl}",
        f"- **Serving:** continuous batching gives {_ratio(co8['tok_s'] / st8['tok_s'])} static batching. "
        f"A paged KV cache fits {pg['peak_active']} sequences where contiguous slots fit 8, for "
        f"{_ratio(pg['tok_s'] / c8['tok_s'])} the throughput in the same memory.",
    ])

    hf = data["tl-hf"]
    prof = data["tl-profile"]["modes"]
    kv32 = at("tl-kv", 32)
    host_bound = prof["eager"]["gpu_idle_frac"] >= 0.5
    i8 = data["tl-int8-graph"]
    vs_bf16 = lambda v, b: (need(i8["rows"], "tl-int8-graph", variant=v, batch=b)["decode_tok_s"]   # noqa: E731
                            / need(i8["rows"], "tl-int8-graph", variant="bf16", batch=b)["decode_tok_s"])
    tri, cast = {b: vs_bf16("int8-triton", b) for b in (1, 32)}, {b: vs_bf16("int8", b) for b in (1, 32)}
    int8_text = (f"- **INT8 weights:** val loss {i8['val_loss']['int8-triton'] - i8['val_loss']['bf16']:+.4f}, "
                 f"{i8['teacher_forced']['int8-triton']['argmax_agree']:.1%} argmax agreement with bf16. Our "
                 "Triton kernel reads the int8 weight and dequantizes inside the matmul; with CUDA graphs, decode "
                 f"is {_ratio(tri[1])} bf16 at batch 1 and {_ratio(tri[32])} at batch 32. Casting the weight back "
                 f"to bf16 on every call (plain PyTorch) gives {_ratio(cast[1])} and {_ratio(cast[32])}")
    int8_text += (": it moves more bytes than bf16, not fewer." if cast[1] < 1 and cast[32] < 1 else ".")
    eager_text = (f"- **Eager decode is {'host-bound' if host_bound else 'slow'}.** At batch 1, HF and our eager "
                  f"engine take {hf1['ms_per_step']:.1f} and {kv1['ms_per_step']:.1f} ms per step "
                  f"(ours {kv32['ms_per_step']:.1f} ms at batch 32). The profiler shows the GPU idle "
                  f"{prof['eager']['gpu_idle_frac']:.0%} of an eager step"
                  + (", waiting for kernel launches" if host_bound else "")
                  + f". Capturing the decode step as a CUDA graph removes that overhead; the GPU is then idle "
                  f"{prof['graph']['gpu_idle_frac']:.0%} of the step.")
    corr = (f"- **Correctness:** bf16 logits equal HF's bf16 logits exactly (`torch.equal`) at all {positions} "
            "test positions." if identical else
            f"- **Correctness:** bf16 logits differ from HF's bf16 logits by at most {max_d:.2g} over "
            f"{positions} test positions." + gap_cause)
    corr += (f" Every top-1 disagreement with fp32 is a near-tie (margin ≤ {max(flips):.3f} logits)." if flips
             else " No top-1 disagreement with fp32.")
    tinyllama = "\n".join([
        f"Decode speed at batch 1 with a 128-token prompt, 256 new tokens, {repeats_note(hf)}. The ceiling is "
        f"the measured bandwidth divided by the weight size: {ceiling_note(hf)}, because each decoded token "
        "has to read every weight once.",
        "",
        tl_b1_table(data),
        "",
        eager_text,
        f"- **Batching:** with graphs, {n(at('tl-kv-graph', 8)['decode_tok_s'])} tok/s at batch 8 and "
        f"{n(at('tl-kv-graph', 32)['decode_tok_s'])} tok/s at batch 32 (HF: {n(at('tl-hf', 8)['decode_tok_s'])} "
        f"and {n(at('tl-hf', 32)['decode_tok_s'])}).",
        int8_text,
        corr,
        "",
        img("docs/img/batch_scaling.svg", scaling_alt(data)),
    ])

    w = data["tl-serving"]["workload"]
    reps = (data["tl-serving"].get("provenance") or {}).get("repeats")
    serving = "\n".join([
        f"{w['requests']} requests of mixed lengths ({n(st8['useful_tokens'])} tokens to generate), "
        f"{'median of ' + str(reps) + ' runs' if reps else 'median run'}.",
        "",
        table(["Scheduler / cache", "Setting", "tok/s", "vs baseline"], [
            ["Static batching (TinyLlama)", "8 slots", n(st8["tok_s"]), "—"],
            ["Continuous batching (TinyLlama)", "8 slots", n(co8["tok_s"]), _ratio(co8["tok_s"] / st8["tok_s"])],
            ["Continuous batching + CUDA graphs (TinyLlama)", "8 slots", n(gr8["tok_s"]),
             f"{_ratio(gr8['tok_s'] / co8['tok_s'])} vs eager continuous"],
            ["Contiguous KV slots (30M model)", f"{c8['kv_reserved_mib']:.0f} MiB of KV, 8 sequences",
             n(c8["tok_s"]), "—"],
            ["Paged KV cache (30M model)", f"same {pg['kv_reserved_mib']:.0f} MiB, {pg['peak_active']} sequences",
             n(pg["tok_s"]), _ratio(pg["tok_s"] / c8["tok_s"])],
        ]),
        "",
        "The paged cache stores KV in fixed-size blocks through a per-sequence block table, as in vLLM, so a "
        "sequence holds only the blocks it has filled. The share of reserved KV holding real tokens goes from "
        f"{c8['kv_alloc_used_mean']:.0%} to {pg['kv_alloc_used_mean']:.0%}.",
    ])
    return {"highlights": highlights, "tinyllama": tinyllama, "serving": serving}


def replace_block(text: str, name: str, body: str) -> str:
    """Replace what sits between <!-- bench:NAME --> and <!-- /bench:NAME -->; the rest stays byte-identical."""
    start, end = f"<!-- bench:{name} -->", f"<!-- /bench:{name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one {start} and one {end}")
    i, j = text.index(start) + len(start), text.index(end)
    if j < i:
        raise ValueError(f"{end} comes before {start}")
    return text[:i] + "\n" + body.strip("\n") + "\n" + text[j:]


def render_readme(text: str, data: dict) -> str:
    for name, body in readme_blocks(data).items():
        text = replace_block(text, name, body)
    return text


def block_names(text: str) -> list[str]:
    return re.findall(r"<!-- bench:([\w-]+) -->", text)
