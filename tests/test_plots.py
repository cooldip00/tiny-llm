"""Step 6.2 (D-030): the plots (tinyllm.plots), the bench_all render step and the training-curve extractor.

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
ALL = {"decode_b1.svg", "batch_scaling.svg", "decode_profile.svg", "serving.svg", "train_loss.svg"}


def test_render_plots_is_deterministic(tmp_path):
    data = R.load_run(fake_run(tmp_path))
    a, b = P.render_plots(data, CURVE), P.render_plots(data, CURVE)
    assert set(a) == ALL and a == b                       # byte-identical: re-rendering leaves git clean
    for svg in a.values():
        assert svg.lstrip().startswith("<?xml") and "<dc:date>" not in svg
    assert set(P.render_plots(data, None)) == ALL - {"train_loss.svg"}


def test_plot_missing_row_names_the_bench(tmp_path):
    data = R.load_run(fake_run(tmp_path))
    data["tl-serving"]["rows"] = [r for r in data["tl-serving"]["rows"] if r["engine"] != "cont+graph"]
    with pytest.raises(R.MissingResult, match="tl-serving"):
        P.render_plots(data, None)


def test_bench_all_render_writes_plots_once(tmp_path, capsys):
    import bench_all
    run = fake_run(tmp_path / "run")
    readme, results, img, curve = tmp_path / "README.md", tmp_path / "RESULTS.md", tmp_path / "img", tmp_path / "c.json"
    readme.write_text(README)
    curve.write_text(json.dumps(CURVE))
    assert bench_all.render(run, results, readme, img, curve) == 0
    assert {p.name for p in img.iterdir()} == ALL and "5 of 5 changed" in capsys.readouterr().out
    assert bench_all.render(run, results, readme, img, curve) == 0
    assert "0 of 5 changed" in capsys.readouterr().out
    out = readme.read_text()
    assert "![Bar chart, TinyLlama 1.1B decode at batch 1: HF Transformers generate 45 tok/s" in out
    assert "(docs/img/decode_b1.svg)" in out and "(docs/img/batch_scaling.svg)" in out
    page = results.read_text()
    assert all(f"(img/{n})" in page for n in ALL - {"train_loss.svg"})


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
