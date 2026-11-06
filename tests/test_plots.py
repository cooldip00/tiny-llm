"""Step 6.2 (D-030): the plots (tinyllm.plots), the bench_all render step and the docs/data extractors.

CPU only, on the made-up run from test_results; checks plumbing and determinism, not how the plots look.
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from test_results import README, fake_run  # noqa: E402

from tinyllm import plots as P  # noqa: E402
from tinyllm import results as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

CURVE = {"source": "checkpoints/main/train.log", "params": 29_893_120, "tokens_per_step": 131_072,
         "train": [[s, 9.0 - s / 100] for s in range(0, 600, 10)], "val": [[0, 9.0], [200, 2.4], [400, 1.6], [599, 1.3]]}

def _sweep(shed: float, ttft: float) -> dict:
    """One 4.8a *-sweep.json, as load_test.py writes it (per-request timings left out)."""
    rows = [{"summary": {"rate": r, "achieved_rps": r * 0.9, "tok_s": r * 115.0, "ttft_s": {"p50": 0.05, "p95": ttft * r},
                         "shed_rate": shed * r / 16}, "requests": []} for r in (1.0, 4.0, 16.0)]
    return {"workload": {"rates": [1, 4, 16], "duration": 60.0, "max_tokens": 128}, "slo": {"ttft_p95_s": 1.0},
            "model": "/home/someone/models/x.gguf", "rated_load_rps": 4.0, "rows": rows,
            "provenance": {"git": {"commit": "ccdc5210000"}, "time_utc": "2026-11-06T08:00:00+00:00"}}


SWEEPS = {"ours-8slots": _sweep(0.6, 0.4), "vllm-8seqs": _sweep(0.0, 3.0), "llamacpp-bf16": _sweep(0.0, 12.0)}
ALL = {"decode_b1.svg", "batch_scaling.svg", "decode_profile.svg", "serving.svg", "train_loss.svg", "load_sweep.svg"}


@pytest.fixture
def load(tmp_path, monkeypatch):
    """docs/data/load_sweep.json made by scripts/extract_load_sweep.py from fake sweeps."""
    import extract_load_sweep as X
    src, out = tmp_path / "4.8a", tmp_path / "load_sweep.json"
    src.mkdir()
    for stem, d in SWEEPS.items():
        (src / f"{stem}-sweep.json").write_text(json.dumps(d))
    monkeypatch.setattr(sys, "argv", ["extract_load_sweep.py", "--dir", str(src), "--out", str(out)])
    X.main()
    return out


def test_render_plots_is_deterministic(tmp_path, load):
    data = R.load_run(fake_run(tmp_path))
    sweep = json.loads(load.read_text())
    a, b = P.render_plots(data, CURVE, sweep), P.render_plots(data, CURVE, sweep)
    assert set(a) == ALL and a == b                       # byte-identical: re-rendering leaves git clean
    for svg in a.values():
        assert svg.lstrip().startswith("<?xml") and "<dc:date>" not in svg
    assert set(P.render_plots(data, None)) == ALL - {"train_loss.svg", "load_sweep.svg"}


def test_plot_missing_row_names_the_bench(tmp_path):
    data = R.load_run(fake_run(tmp_path))
    data["tl-serving"]["rows"] = [r for r in data["tl-serving"]["rows"] if r["engine"] != "cont+graph"]
    with pytest.raises(R.MissingResult, match="tl-serving"):
        P.render_plots(data, None)


def test_bench_all_render_writes_plots_once(tmp_path, capsys, load):
    import bench_all
    run = fake_run(tmp_path / "run")
    readme, results, img, curve = tmp_path / "README.md", tmp_path / "RESULTS.md", tmp_path / "img", tmp_path / "c.json"
    readme.write_text(README)
    curve.write_text(json.dumps(CURVE))
    capsys.readouterr()
    assert bench_all.render(run, results, readme, img, curve, load) == 0
    assert {p.name for p in img.iterdir()} == ALL and "6 of 6 changed" in capsys.readouterr().out
    assert bench_all.render(run, results, readme, img, curve, load) == 0
    assert "0 of 6 changed" in capsys.readouterr().out
    out = readme.read_text()
    assert "![Bar chart, TinyLlama 1.1B decode at batch 1: HF Transformers generate 45 tok/s" in out
    assert "(docs/img/decode_b1.svg)" in out and "(docs/img/batch_scaling.svg)" in out
    page = results.read_text()
    assert all(f"(img/{n})" in page for n in ALL - {"train_loss.svg", "load_sweep.svg"})   # README-only plots


def test_extract_train_curve_keeps_the_last_logged_step():
    import extract_train_curve as X
    log = "\n".join(["params 29,893,120 | 131,072 tokens/step", "step     0 | loss 9.0100 | lr 0",
                     "eval     0 | val 9.0000", "step    10 | loss 5.0000", "step    10 | loss 4.5000",  # resumed run
                     "eval    10 | val 4.6000", "something else"])
    c = X.parse(log)
    assert (c["params"], c["tokens_per_step"]) == (29_893_120, 131_072)
    assert c["train"] == [[0, 9.01], [10, 4.5]] and c["val"] == [[0, 9.0], [10, 4.6]]
    with pytest.raises(SystemExit, match="tinyllm.train log"):
        X.parse("nothing here")


def test_extract_load_sweep_keeps_summaries_not_paths(load):
    d = json.loads(load.read_text())
    assert list(d["engines"]) == ["Ours", "vLLM", "llama.cpp"] and d["slo_ttft_p95_s"] == 1.0
    ours = d["engines"]["Ours"]
    assert ours["commit"] == "ccdc521" and ours["points"][-1] == [16.0, 14.4, 1840.0, 6.4, 0.6]
    assert len(ours["sha256"]) == 64
    assert "/home/" not in load.read_text()              # the sweep's model path is personal; never committed
