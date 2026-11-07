"""Step 6.8: results rendering (tinyllm.results) and the benchmark output guard, on CPU with made-up JSON.

The JSON shapes copy what bench_generate / bench_serving / bench_quant / tinyllama_vs_hf /
profile_decode write; the numbers are invented, so the tests check plumbing, not performance.
"""

import json
import sys
from pathlib import Path

import pytest

from tinyllm import results as R
from tinyllm.provenance import claim_output

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def prov(clock=1600.0, temp=80.0, reasons=("sw_power_cap",), **extra):
    return {"time_utc": "2026-11-07T10:00:00+00:00", "git": {"commit": "abcdef1234567", "dirty": False},
            "gpu_name": "Test GPU", "driver": "1.0", "torch": "2.x", "cuda": "13.0", **extra,
            "during": {"clocks.sm_busy": {"min": clock - 200, "median": clock, "max": clock + 100, "n": 10},
                       "temperature.gpu": {"min": 50.0, "median": temp - 2, "max": temp},
                       "throttle_reasons_seen": ["gpu_idle", *reasons]}}


def gen(scale, batches=(1, 8, 32), plens=(32, 128, 512), max_len=2048, weight_bytes=2_200_000_000):
    rows = []
    for b in batches:
        for p in plens:
            if p + 256 > max_len and scale != "naive":
                rows.append({"batch": b, "prompt_len": p, "skipped": "exceeds max_seq_len"})
                continue
            tok = scale * b * 10 if isinstance(scale, (int, float)) else 33.0 * b
            ceiling = 423e9 / weight_bytes * b
            rows.append({"batch": b, "prompt_len": p, "decode_tok_s": tok, "ms_per_step": 1e3 * b / tok,
                         "pct_of_ceiling": 100 * tok / ceiling, "window_slides": p + 256 > max_len})
    return {"bandwidth_bytes_s": 423e9, "weight_bytes": weight_bytes, "provenance": prov(repeats=5), "rows": rows}


def serving(paged):
    def r(engine, size, tok, active=0, alloc=None):
        return {"engine": engine, "size": size, "tok_s": tok, "useful_tokens": 9004, "latency_mean_s": 1.0,
                "latency_p95_s": 2.0, "ttft_mean_s": 0.5, "ttft_p95_s": 1.0, "kv_reserved_mib": 64.0,
                "kv_alloc_used_mean": alloc, "peak_active": active, "preemptions": 0,
                "correctness": {"identical": 60, "of": 64}}
    rows = [r("static", 8, 200.0), r("continuous", 8, 300.0, 8, 0.5), r("cont+graph", 8, 600.0, 8, 0.5),
            r("static", 32, 700.0), r("continuous", 32, 800.0, 32, 0.55), r("cont+graph", 32, 1500.0, 32, 0.55)]
    if paged:
        rows += [r("paged/16", 8, 290.0, 8, 0.97), r("paged/16", 32, 450.0, 25, 0.97)]
    return {"workload": {"requests": 64, "seed": 0}, "provenance": prov(repeats=3), "rows": rows}


def int8(base, variants):
    speed = {"int8": 0.8, "int8-triton": 1.4}
    q = lambda v: v.startswith("int8")                                          # noqa: E731
    rows = [{"variant": v, "batch": b, "decode_tok_s": 100.0 * b * speed.get(v, 1.0)}
            for b in (1, 8, 32) for v in variants]
    return {"baseline": base, "provenance": prov(), "rows": rows,
            "weights_mib": {v: {"matmul": 100.0, "embedding": 10.0, "other": 0.1} for v in variants},
            "val_loss": {v: 1.6 + (0.001 if q(v) else 0.0) for v in variants},
            "teacher_forced": {**{v: {"argmax_agree": 0.985 if q(v) else 1.0} for v in variants},
                               "_median_gap": 2.0},
            "greedy_rows_identical": {v: (0 if q(v) else 8) for v in variants}}


def gemv():
    # One shape, 10 calls per step, 1 MiB of bf16 weight per call: a 10 MiB step for bf16, 5 MiB for int8.
    us = {"bf16": 50.0, "torch": 150.0, "triton": 25.0}
    mib = {"bf16": 2**20, "torch": 2**19, "triton": 2**19}
    rows = [{"shape": "w", "m": m, "per_step": 10, **{f"{v}_us": us[v] for v in us},
             **{f"{v}_gb_s": mib[v] / (us[v] * 1e3) for v in us}} for m in (1, 32)]
    return {"bandwidth": 423e9, "provenance": prov(), "rows": rows,
            "step_us": {str(m): {v: 10 * us[v] for v in us} for m in (1, 32)}}


def logits():
    def phase(pos, flips):
        s = {"max_abs": 0.0, "top1": 1.0, "positions": pos}
        return {"ours_vs_hf_bf16": s, "ours_bf16": {**s, "top1": 0.99}, "hf_bf16": {**s, "top1": 0.99},
                "bit_identical_to_hf_bf16": True, "flips": {"ours_bf16": flips, "hf_bf16": flips}}
    return {"new_tokens": 64, "provenance": prov(),
            "stats": {"prefill": phase(151, []), "decode": phase(264, [0.0022, 0.019])},
            "greedy_divergence": [{"ours_vs_hf_bf16": None, "ours_vs_fp32": 43, "hf_bf16_vs_fp32": 43}]}


def profile():
    m = lambda ms, idle: {"ms_per_step": ms, "host_issue_ms_per_step": ms * 0.9,      # noqa: E731
                          "gpu_busy_ms_per_step": ms * 0.3, "gpu_idle_frac": idle,
                          "decode_tok_s": 1e3 / ms, "pct_of_ceiling": 50.0}
    return {"modes": {"eager": m(22.0, 0.7), "graph": m(7.8, 0.03)}, "provenance": prov()}


def fake_run(tmp_path: Path) -> Path:
    files = {
        "30m-naive": gen("naive", max_len=512, weight_bytes=60_000_000),
        "30m-kv": gen(50, max_len=512, weight_bytes=60_000_000),
        "30m-kv-graph": gen(80, max_len=512, weight_bytes=60_000_000),
        "30m-serving": serving(paged=True), "30m-int8": int8("fp32", ["fp32", "bf16", "int8"]),
        "30m-profile": profile(), "tl-logits": logits(),
        "tl-hf": gen(4.5), "tl-naive": gen("naive", batches=(1,)), "tl-kv": gen(4.5), "tl-kv-graph": gen(12.8),
        "tl-serving": serving(paged=False), "tl-int8": int8("bf16", ["bf16", "int8"]), "tl-profile": profile(),
        "tl-int8-graph": int8("bf16", ["bf16", "int8", "int8-triton"]), "tl-int8-gemv": gemv(),
    }
    for name, d in files.items():
        p = tmp_path / R.BY_NAME[name].out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    (tmp_path / R.MANIFEST).write_text(json.dumps({
        "power_mode": "performance", "finished_utc": "2026-11-07T12:00",
        "benches": {"tl-hf": {"rerun": False}, "tl-int8": {"rerun": True, "finished_utc": "2026-11-07T13:42:20+00:00",
                                                           "start_temp_c": 49.0}}}))
    return tmp_path


README = """# x

intro stays
<!-- bench:highlights -->
old highlights
<!-- /bench:highlights -->
- training bullet stays

## Results: TinyLlama
<!-- bench:tinyllama -->
old table
<!-- /bench:tinyllama -->
Full tables: docs/RESULTS.md

<!-- bench:serving -->
<!-- /bench:serving -->
tail stays
"""


def test_render_results_from_one_run(tmp_path):
    page = R.render_results(R.load_run(fake_run(tmp_path)))
    assert page.startswith("# Results\n") and "Do not edit by hand" in page
    assert "NitroSense **performance**" in page and "commit `abcdef1`" in page
    assert "| **Ours, KV cache + CUDA graphs** | **128** |" in page        # 12.8 * 1 * 10 tok/s at B=1
    assert "1.50× static" in page                                        # continuous 300 / static 200
    assert "1.50× continuous 8 (same KV memory)" in page                 # paged/16 at 32: 450 / 300
    assert "0.80×" in page and "int8 +0.0010" in page
    assert "| 1 | 100 | 80 | 140 | 0.80× | 1.40× |" in page                # graphs table: bf16 / int8 / int8-triton
    assert "| 1 | 0.50 | 1.50 | 0.25 | 2.00× | 6.0× |" in page              # matmuls alone, ms per step
    assert "(the 10 Linears)" in page and "read 10 MiB of bf16 weights or 5 MiB" in page
    assert "bf16 runs at 5% of the measured 423 GB/s, the Triton kernel at 5%" in page   # 5 MiB / 250 µs
    assert "1,600 MHz (min 1,400), 80 °C, sw_power_cap" in page and "gpu_idle" not in page
    assert "None" not in page and "\n\n\n" not in page and "Throttling warnings" not in page
    assert "token 43 / token 43" in page and " ‡" in page                # 30M naive at P=512 slides its window
    assert "**Rerun later with `--only`:** tl-int8 (finished 13:42 UTC, started at 49.0 °C)." in page
    assert "tl-hf (" not in page


def test_readme_blocks_replace_only_inside_markers(tmp_path):
    data = R.load_run(fake_run(tmp_path))
    out = R.render_readme(README, data)
    for keep in ("# x\n\nintro stays\n<!-- bench:highlights -->\n", "<!-- /bench:highlights -->\n- training bullet stays\n",
                 "<!-- /bench:tinyllama -->\nFull tables: docs/RESULTS.md\n", "<!-- /bench:serving -->\ntail stays\n"):
        assert keep in out
    assert "old highlights" not in out and "old table" not in out
    assert "decodes at 128 tok/s at batch 1" in out and "2.8× HF `generate`" in out
    assert "bit-identical" in out and "all 415 test positions" in out and "margin ≤ 0.019" in out
    assert "decode is 1.40× bf16 at batch 1 and 1.40× at batch 32" in out
    assert "(plain PyTorch) gives 0.80× and 0.80×: it moves more bytes" in out and "roadmap" not in out
    assert "fits 25 sequences" in out and "| Paged KV cache (30M model) | same 64 MiB, 25 sequences | 450 | 1.50× |" in out
    assert R.render_readme(out, data) == out                              # idempotent


def test_replace_block_rejects_bad_markers():
    with pytest.raises(ValueError):
        R.replace_block("no markers", "x", "body")
    with pytest.raises(ValueError):
        R.replace_block("<!-- bench:x -->\n<!-- /bench:x -->\n<!-- bench:x -->", "x", "body")
    with pytest.raises(ValueError):
        R.replace_block("<!-- /bench:x -->\n<!-- bench:x -->", "x", "body")


def test_missing_file_names_the_rerun(tmp_path):
    run = fake_run(tmp_path)
    (run / R.BY_NAME["tl-serving"].out).unlink()
    with pytest.raises(R.MissingResult, match=r"tinyllama/serving\.json.*--only tl-serving"):
        R.load_run(run)
    with pytest.raises(R.MissingResult, match="16 result file"):
        R.load_run(tmp_path / "empty")


def test_heat_warnings():
    assert R.heat(None) is None and R.heat_text(None) == "not recorded"
    assert R.heat(prov())["warn"] == []
    assert R.heat(prov(reasons=("hw_thermal_slowdown",)))["warn"] == ["hw_thermal_slowdown"]
    assert R.heat(prov(clock=1200.0))["warn"] == []          # idle-ish GPU clocking down: not throttling
    low = R.heat(prov(clock=1200.0, reasons=("sw_thermal_slowdown",)))
    assert low["warn"] and R.heat_text(low).startswith("⚠ ")


def test_claim_output(tmp_path):
    p = tmp_path / "a" / "b.json"
    assert claim_output(p) == p and p.parent.is_dir()
    p.write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        claim_output(p)
    assert claim_output(p, overwrite=True) == p


def test_default_output_names_differ_per_mode():
    import bench_generate
    import bench_serving
    names = {bench_serving.default_out(t, g, p) for t in (False, True) for g in (False, True) for p in (False, True)}
    assert len(names) == 8 and Path("runs/bench/tinyllama/serving_graph.json") in names
    d = bench_generate.DEFAULT_PROMPT_LENS
    assert bench_generate.default_out("kv-graph", "bf16", True, "1,8,32", d, "1,8,32") == "runs/bench/tinyllama/kv-graph.json"
    assert bench_generate.default_out("kv-graph", "bf16", True, "32", d, "1,8,32") == "runs/bench/tinyllama/kv-graph_b32.json"
    assert bench_generate.default_out("kv", "bf16", False, "1,8,32", "128", "1,8,32") == "runs/bench/kv_bf16_p128.json"


def test_bench_outputs_are_unique():
    outs = [b.out for b in R.BENCHES]
    assert len(set(outs)) == len(outs) and len(R.BY_NAME) == len(R.BENCHES)


def test_only_naive_benches_change_the_allocator():
    # Growing full-forward buffers fragment the caching allocator (6.8 crash); fixed-size caches and
    # CUDA graph capture run with the default allocator.
    changed = {b.name for b in R.BENCHES if b.env}
    assert changed == {"30m-naive", "tl-naive"}
    assert all(dict(R.BY_NAME[n].env) == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} for n in changed)
