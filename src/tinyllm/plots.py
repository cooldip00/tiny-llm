"""Step 6.2 (D-030): the README and docs/RESULTS.md plots, drawn from the same run as the tables.

render_plots(data, curve) returns {file name: SVG text}; scripts/bench_all.py writes them to docs/img/.
`data` is results.load_run(); `curve` is docs/data/train_curve.json (scripts/extract_train_curve.py).

  - One transparent SVG per plot, readable on GitHub's light and dark themes: mid-gray text, and a
    blue / orange pair checked for colour-vision deficiency on both backgrounds. Blue = ours, orange =
    HF Transformers, gray = context. Every bar and line end carries its value, so color is never the
    only key.
  - Byte-identical between renders (fixed SVG hash salt, no date, text drawn as paths with the font
    matplotlib ships), so re-rendering leaves git clean. matplotlib's version is pinned by uv.lock.
  - Pyplot is not used: no global figure state, no GUI backend.
"""

from __future__ import annotations

import io

import matplotlib
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from tinyllm.results import PAGED_ROW, n, need

OURS, HF, CONTEXT = "#3987e5", "#e0602d", "#8b949e"
INK = "#6e7681"         # ≥ 4:1 on both white and GitHub's dark #0d1117
RC = {
    "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path", "svg.hashsalt": "tinyllm",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": CONTEXT, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "none", "figure.facecolor": "none", "savefig.transparent": True,
    "xtick.major.size": 3, "ytick.major.size": 0, "legend.frameon": False,
}
GRID = {"color": CONTEXT, "alpha": 0.3, "linewidth": 0.6}


def _svg(fig: Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.12)
    return buf.getvalue()


def _head(ax, title: str, subtitle: str, lift: float = 0) -> None:
    """Left-aligned title and subtitle above the plot area, placed in points so they never overlap it.
    A subtitle with line breaks pushes the title up one line per break."""
    lift_title = lift + 12 * subtitle.count("\n")
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 30 + lift_title), textcoords="offset points",
                fontsize=12.5, fontweight="bold", va="bottom")
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 12 + lift), textcoords="offset points",
                fontsize=9.5, va="bottom")


def _foot(ax, text: str, below: float = 34) -> None:
    ax.annotate(text, (0, 0), xycoords="axes fraction", xytext=(0, -below), textcoords="offset points",
                fontsize=8, va="top")


def _source(data: dict, name: str) -> str:
    commit = (((data[name].get("provenance") or {}).get("git") or {}).get("commit") or "?")[:7]
    return f"tiny-llm · {data.get('_run_dir', 'runs/bench/v1b')} · commit {commit} · RTX 3070 Ti Laptop GPU"


def decode_b1(data: dict) -> Figure:
    hf = data["tl-hf"]
    ceiling = hf["bandwidth_bytes_s"] / hf["weight_bytes"]
    engines = [("tl-kv-graph", "Ours: KV cache + CUDA graphs", OURS), ("tl-kv", "Ours: KV cache", CONTEXT),
               ("tl-hf", "HF Transformers generate", HF), ("tl-naive", "Ours: naive, no cache", CONTEXT)]
    rows = [(label, need(data[nm]["rows"], nm, batch=1, prompt_len=128), c) for nm, label, c in engines]
    rows.sort(key=lambda r: -r[1]["decode_tok_s"])
    fig = Figure(figsize=(7.6, 2.5))
    ax = fig.add_subplot()
    y = range(len(rows))
    ax.barh(y, [r["decode_tok_s"] for _, r, _ in rows], height=0.62, color=[c for *_, c in rows])
    ax.set_yticks(list(y), [label for label, *_ in rows])
    ax.invert_yaxis()
    for i, (_, r, _) in enumerate(rows):
        ax.text(r["decode_tok_s"] + ceiling * 0.015, i, f"{n(r['decode_tok_s'])} tok/s · {r['pct_of_ceiling']:.0f}%",
                va="center", fontsize=9.5)
    ax.axvline(ceiling, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(ceiling * 1.015, len(rows) - 1, f"bandwidth\nceiling\n{n(ceiling)} tok/s", va="center", fontsize=9)
    ax.set_xlim(0, ceiling * 1.28)
    ax.xaxis.set_major_locator(FixedLocator([0, 50, 100, 150, 200]))
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("decode tokens / s at batch 1")
    _head(ax, "TinyLlama 1.1B decode speed, batch 1",
          f"128-token prompt, 256 new tokens. % = share of the {n(ceiling)} tok/s ceiling "
          "(bandwidth ÷ weight bytes).")
    _foot(ax, _source(data, "tl-kv-graph"), below=40)
    return fig


def batch_scaling(data: dict) -> Figure:
    hf = data["tl-hf"]
    ceiling = hf["bandwidth_bytes_s"] / hf["weight_bytes"]
    batches = sorted({r["batch"] for r in data["tl-kv-graph"]["rows"]} & {r["batch"] for r in hf["rows"]})
    tok = {nm: [need(data[nm]["rows"], nm, batch=b, prompt_len=128)["decode_tok_s"] for b in batches]
           for nm in ("tl-kv-graph", "tl-hf")}
    fig = Figure(figsize=(7.6, 3.6))
    ax = fig.add_subplot()
    ax.plot(batches, [ceiling * b for b in batches], color=INK, linewidth=1.2, linestyle=(0, (4, 3)),
            label=f"bandwidth ceiling ({n(ceiling)} tok/s × batch)")
    ax.plot(batches, tok["tl-kv-graph"], color=OURS, linewidth=2, marker="o", markersize=7,
            label="Ours: KV cache + CUDA graphs")
    ax.plot(batches, tok["tl-hf"], color=HF, linewidth=2, marker="s", markersize=6.5,
            label="HF Transformers generate")
    # Every line rises to the right: just left of our markers (incoming line below, ceiling well above) and
    # below-right of HF's are clear.
    for nm, off, ha, va in (("tl-kv-graph", (-8, 2), "right", "bottom"), ("tl-hf", (6, -7), "left", "top")):
        for b, v in zip(batches, tok[nm]):
            ax.annotate(n(v), (b, v), xytext=off, textcoords="offset points", fontsize=9, ha=ha, va=va)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(batches))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ticks = [t for t in (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)
             if min(tok["tl-hf"][0], tok["tl-kv-graph"][0]) / 2 <= t <= ceiling * batches[-1] * 1.5]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: n(v)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(batches[0] / 1.7, batches[-1] * 1.9)
    ax.set_ylim(min(tok["tl-hf"][0], tok["tl-kv-graph"][0]) / 1.8, ceiling * batches[-1] * 1.4)
    ax.grid(axis="y", **GRID)
    ax.set_xlabel("batch size (sequences decoded together)")
    ax.set_ylabel("decode tokens / s (log scale)")
    ax.legend(loc="upper left", fontsize=9)
    wmib = hf["weight_bytes"] / 2**20
    _head(ax, "Decode throughput vs batch size, TinyLlama 1.1B",
          f"Each decode step reads all {n(wmib)} MiB of weights once; a batch shares that read. 128-token prompt.")
    _foot(ax, _source(data, "tl-kv-graph"), below=40)
    return fig


def train_loss(curve: dict) -> Figure:
    tps = curve["tokens_per_step"]
    tr, va = curve["train"], curve["val"]
    top = 3.2
    fig = Figure(figsize=(7.6, 3.2))
    ax = fig.add_subplot()
    ax.plot([s * tps / 1e6 for s, _ in tr], [v for _, v in tr], color=CONTEXT, linewidth=0.9, label="train loss")
    ax.plot([s * tps / 1e6 for s, _ in va], [v for _, v in va], color=OURS, linewidth=2, marker="o",
            markersize=3.5, label="validation loss")
    last_s, last_v = va[-1]
    ax.annotate(f"val {last_v:.4f}", (last_s * tps / 1e6, last_v), xytext=(-4, 14), textcoords="offset points",
                ha="right", fontsize=9.5)
    ax.annotate(f"← loss starts at {va[0][1]:.1f}, above the chart", (0.06, 0.97), xycoords="axes fraction",
                fontsize=8.5, va="top")
    ax.set_ylim(1.0, top)
    ax.set_xlim(0, tr[-1][0] * tps / 1e6 * 1.01)
    ax.grid(axis="y", **GRID)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(loc="upper right", fontsize=9)
    every = va[1][0] - va[0][0] if len(va) > 1 else 0
    _head(ax, "Training our 30M-parameter model from scratch on TinyStories",
          f"{curve['params'] / 1e6:.1f}M parameters, {tr[-1][0] * tps / 1e6:,.0f}M tokens"
          + (f"; validation every {every} steps ({n(every * tps / 1e6)}M tokens)" if every else "") + ".")
    _foot(ax, f"tiny-llm · docs/data/train_curve.json (from {curve.get('source', 'the training log')})", below=40)
    return fig


def decode_profile(data: dict) -> Figure:
    d = data["tl-profile"]
    # Two-line labels and a two-line subtitle keep the SVG narrow enough to show near full size in a blog column.
    modes = [("eager", "Eager:\none launch per kernel"), ("graph", "CUDA graph:\none launch per step")]
    fig = Figure(figsize=(6.0, 2.1))
    ax = fig.add_subplot()
    for i, (mode, _) in enumerate(modes):
        m = d["modes"][mode]
        busy = 1 - m["gpu_idle_frac"]
        ax.barh(i, busy, height=0.58, color=OURS, label="GPU busy" if i == 0 else None)
        ax.barh(i, m["gpu_idle_frac"], left=busy, height=0.58, color=CONTEXT, alpha=0.55,
                label="GPU idle, waiting for launches" if i == 0 else None)
        ops = m.get("gpu_ops_per_step")
        ax.text(1.02, i, f"{m['ms_per_step']:.1f} ms / step, idle {m['gpu_idle_frac']:.0%}"
                + (f"\n{n(ops)} kernels" if ops else ""), va="center", fontsize=9)
    ax.set_yticks(range(len(modes)), [label for _, label in modes])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of one decode step")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=9, borderaxespad=0.2)
    speed = d["modes"]["eager"]["ms_per_step"] / d["modes"]["graph"]["ms_per_step"]
    _head(ax, "Why CUDA graphs help: TinyLlama 1.1B decode step, batch 1",
          f"In eager mode Python launches every kernel and the GPU waits;\na graph replays the whole step "
          f"at once ({speed:.2f}× faster).", lift=16)                           # above the legend
    _foot(ax, _source(data, "tl-profile"), below=40)
    return fig


def serving(data: dict) -> Figure:
    tls, ms = data["tl-serving"]["rows"], data["30m-serving"]["rows"]
    st = need(tls, "tl-serving", engine="static", size=8)
    left = [("Static batching", st), ("Continuous batching", need(tls, "tl-serving", engine="continuous", size=8)),
            ("Continuous + CUDA graphs", need(tls, "tl-serving", engine="cont+graph", size=8))]
    c8 = need(ms, "30m-serving", engine="continuous", size=8)
    pg = need(ms, "30m-serving", engine=PAGED_ROW[0], size=PAGED_ROW[1])
    right = [(f"Contiguous KV: {c8['peak_active']} sequences", c8), (f"Paged KV: {pg['peak_active']} sequences", pg)]
    fig = Figure(figsize=(7.6, 2.4))
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 1.05, "width_ratios": [1, 1]})
    for ax, rows, title in ((axes[0], left, "TinyLlama 1.1B, 8 slots"),
                            (axes[1], right, f"30M model, same {c8['kv_reserved_mib']:.0f} MiB of KV")):
        base = rows[0][1]["tok_s"]
        vals = [r["tok_s"] for _, r in rows]
        ax.barh(range(len(rows)), vals, height=0.6, color=OURS)
        ax.set_yticks(range(len(rows)), [label for label, _ in rows])
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.03, i, n(v) + (f" ({v / base:.2f}×)" if i else ""), va="center", fontsize=9)
        ax.set_xlim(0, max(vals) * 1.45)
        ax.xaxis.set_major_locator(FixedLocator([]))
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    _head(axes[0], f"Serving throughput, {data['tl-serving']['workload']['requests']} mixed-length requests",
          "Tokens / s; × = vs the first row. Continuous batching refills a slot as soon as its request ends;\n"
          "the paged cache fits more sequences in the same memory.", lift=18)   # two lines: one line made the SVG
                                                                                # 1.6× wider than the plot
    _foot(axes[0], _source(data, "tl-serving"), below=12)
    return fig


AQUA = "#1aa775"        # third engine; blue / orange / aqua checked for colour-vision deficiency on both themes
LOAD_STYLE = {"Ours": (OURS, "o", 7), "vLLM": (HF, "s", 6.5), "llama.cpp": (AQUA, "D", 6)}


def load_sweep(load: dict) -> Figure:
    """docs/data/load_sweep.json (scripts/extract_load_sweep.py): the 4.8a equal-seat sweep, one row per rate."""
    col = {c: i for i, c in enumerate(load["columns"])}
    eng = {name: [[p[col[c]] for p in e["points"]] for c in ("rate_rps", "achieved_rps", "ttft_p95_s", "shed_rate")]
           for name, e in load["engines"].items()}
    top = max(r[-1] for r, *_ in eng.values())
    fig = Figure(figsize=(7.6, 3.3))
    left, right = fig.subplots(1, 2, gridspec_kw={"wspace": 0.7})
    peak = max(max(a) for _, a, *_ in eng.values())
    left.plot([0, peak * 1.12], [0, peak * 1.12], color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    left.annotate("every request\nserved", (peak, peak), xytext=(-8, 0), textcoords="offset points", ha="right",
                  va="center", fontsize=8.5)
    slo = load["slo_ttft_p95_s"]
    right.axhline(slo, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    right.annotate(f"SLO {n(slo)} s", (top, slo), xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                   fontsize=8.5)
    for name, (rate, done, ttft, shed) in eng.items():
        c, m, ms = LOAD_STYLE[name]
        for ax, y in ((left, done), (right, ttft)):
            ax.plot(rate, y, color=c, linewidth=2, marker=m, markersize=ms)
        left.annotate(f"{name} {done[-1]:.1f}", (rate[-1], done[-1]), xytext=(7, 0), textcoords="offset points",
                      va="center", fontsize=9, annotation_clip=False)
        if shed[-1] > 0:        # the gap between our flat line and the diagonal is rejected, not queued
            i = len(rate) - 2
            left.annotate(f"{shed[i] * 100:.0f}–{shed[-1]:.0%} rejected\nwith a 429", (rate[-1], done[-1]), xytext=(0, 9),
                          textcoords="offset points", ha="right", va="bottom", fontsize=8.5)
        end = f"{name} {ttft[-1]:.0f} s" if ttft[-1] >= 10 else f"{name} {ttft[-1]:.1f} s"
        if shed[-1] > 0:        # flat because the bounded queue rejects, not because the engine is faster
            right.annotate(end + ": capped;\na full queue answers\nnew requests with a 429", (rate[-1], ttft[-1]),
                           xytext=(7, 6), textcoords="offset points", va="top", fontsize=9, annotation_clip=False)
        else:
            right.annotate(end, (rate[-1], ttft[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                           fontsize=9, annotation_clip=False)
    for ax in (left, right):
        ax.set_xlim(0, top * 1.04)
        ax.xaxis.set_major_locator(FixedLocator([0, 4, 8, 12, 16]))
        ax.grid(axis="y", **GRID)
        ax.set_xlabel("offered requests / s")
    left.set_ylim(0, peak * 1.12)
    left.set_title("Completed requests / s", loc="left", fontsize=10, color=INK)
    right.set_yscale("log")
    right.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    right.yaxis.set_minor_locator(NullLocator())
    right.set_ylim(0.02, max(max(t) for _, _, t, _ in eng.values()) * 2)
    right.set_title("Time to first token p95, served requests", loc="left", fontsize=10, color=INK)
    w = load["workload"]
    _head(left, "TinyLlama 1.1B under load: 8 sequences per engine",
          f"Poisson arrivals, {w['max_tokens']} new tokens per request, {w['duration']:.0f} s per rate. Past capacity "
          "we reject the excess; vLLM and llama.cpp queue it.", lift=18)       # above the panel titles
    commit = next(iter(load["engines"].values()))["commit"]
    _foot(left, f"tiny-llm · docs/data/load_sweep.json ({load['source']}) · commit {commit} · "
                "RTX 3070 Ti Laptop GPU, thermally throttled", below=40)
    return fig


DRAW = {"decode_b1.svg": decode_b1, "batch_scaling.svg": batch_scaling,
        "decode_profile.svg": decode_profile, "serving.svg": serving}


def render_plots(data: dict, curve: dict | None, load: dict | None = None) -> dict[str, str]:
    """{file name: SVG text}; train_loss.svg and load_sweep.svg only when their docs/data files exist."""
    with matplotlib.rc_context(RC):
        out = {name: _svg(draw(data)) for name, draw in DRAW.items()}
        if curve is not None:
            out["train_loss.svg"] = _svg(train_loss(curve))
        if load is not None:
            out["load_sweep.svg"] = _svg(load_sweep(load))
    return out
