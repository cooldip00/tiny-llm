"""Step 6.8 (D-025, D-029): one command regenerates every results table.

    uv run python scripts/bench_all.py                                   # render only: no GPU, seconds
    uv run python scripts/bench_all.py --run --power-mode performance   # every benchmark (~1.5-2 h), then render
    uv run python scripts/bench_all.py --only tl-serving --power-mode performance   # rerun one, then render
    uv run python scripts/bench_all.py --list                            # benchmarks, commands, outputs

Render: reads runs/bench/v1/ (--dir) and rewrites docs/RESULTS.md (generated in full) and the README's
<!-- bench:NAME --> blocks (the rest of the README stays byte-identical). A missing result file is an
error naming the --only command that makes it. It also draws the plots into docs/img/ (6.2, D-030;
the training and load-test plots from docs/data/train_curve.json and load_sweep.json), byte-identical
when the data hasn't changed.

Run (--run, or --only NAME[,NAME]):
  - --run refuses a run directory that already holds results: one run = one session, never mixed.
    --only reruns into the same directory, with --overwrite passed to the benchmark script.
  - refuses a dirty work tree unless --allow-dirty: every number should come from one commit.
  - one benchmark at a time, as a subprocess, with HF offline; all output is also appended to
    <dir>/run.log (the 11-04 serving numbers survived only because sweep.log existed)
  - before each benchmark, waits until the GPU is at or below --cool-to °C (default 60), at most --cool-max-s
  - after each, warns on hardware throttling or a low busy clock (tinyllm.results.heat)
  - stops at the first failure and says which --only to rerun
  - <dir>/manifest.json records the power mode, times, commit, and each benchmark's status,
    duration, cool-down wait and start temperature
The NitroSense power mode can't be read from WSL, so --power-mode is required for any GPU run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tinyllm import results as R
from tinyllm.provenance import git_info, gpu_sample

REPO = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_temp() -> float | None:
    s = gpu_sample()
    return None if s is None else s.get("temperature.gpu")


def cool_down(target: float, max_s: float) -> tuple[float | None, float]:
    t0, temp = time.monotonic(), gpu_temp()
    if temp is not None and temp > target:
        print(f"cool-down: GPU {temp:.0f} °C, waiting for <= {target:.0f} °C (max {max_s:.0f} s) ...", flush=True)
    while temp is not None and temp > target and time.monotonic() - t0 < max_s:
        time.sleep(10)
        temp = gpu_temp()
    waited = time.monotonic() - t0
    if temp is not None and temp > target:
        print(f"cool-down: still {temp:.0f} °C after {waited:.0f} s; starting anyway (recorded)", flush=True)
    return temp, waited


def command(b: R.Bench, run_dir: Path, overwrite: bool) -> list[str]:
    out = run_dir / b.out
    cmd = [sys.executable, str(REPO / "scripts" / b.script), *b.args]
    cmd += ["--out-dir", str(out.parent)] if b.out_dir_flag else ["--out", str(out)]
    return cmd + (["--overwrite"] if overwrite else [])


def run_logged(cmd: list[str], log_path: Path, extra_env: dict[str, str]) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **extra_env}
    env.setdefault("HF_HUB_OFFLINE", "1")         # weights and data are cached; never wait on the network mid-run
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    with log_path.open("a") as log:
        prefix = "".join(f"{k}={v} " for k, v in extra_env.items())
        log.write(f"\n[{now()}] $ {prefix}{' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO, env=env)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return p.wait()


def save(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def run(args, benches: list[R.Bench], run_dir: Path, rerun: bool) -> int:
    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        print("work tree has uncommitted changes: commit first (every number should come from one commit), "
              "or pass --allow-dirty")
        return 2
    if not rerun:
        have = [b.name for b in R.BENCHES if (run_dir / b.out).exists()]
        if have:
            print(f"{run_dir} already has results ({', '.join(have)}): one run = one session. "
                  "Use a new --dir, or --only NAME to rerun one benchmark.")
            return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    s = gpu_sample() or {}
    if (s.get("memory.used") or 0) > 1024:
        print(f"warning: {s['memory.used']:.0f} MiB of GPU memory already in use: is something else on the GPU?")
    mpath = run_dir / R.MANIFEST
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    if not rerun:
        manifest = {"power_mode": args.power_mode, "started_utc": now(), "git": git, "cool_to_c": args.cool_to,
                    "benches": {}}
    manifest.setdefault("benches", {})
    log = run_dir / "run.log"
    total = sum(b.minutes for b in benches)
    print(f"{len(benches)} benchmark(s), ~{total} min plus cool-downs -> {run_dir} (log: {log})", flush=True)

    for i, b in enumerate(benches, 1):
        temp, waited = cool_down(args.cool_to, args.cool_max_s)
        print(f"\n=== [{i}/{len(benches)}] {b.name} (~{b.minutes} min) at {temp} °C ===", flush=True)
        t0 = time.monotonic()
        code = run_logged(command(b, run_dir, overwrite=rerun), log, dict(b.env))
        rec = {"status": "ok" if code == 0 else f"failed (exit {code})", "seconds": round(time.monotonic() - t0),
               "cooldown_s": round(waited), "cool_to_c": args.cool_to, "start_temp_c": temp, "finished_utc": now(),
               "power_mode": args.power_mode, "git": git, "rerun": rerun, "env": dict(b.env)}
        manifest["benches"][b.name] = rec
        if code != 0:
            save(mpath, manifest)
            print(f"\n{b.name} failed (exit {code}); see {log}. Fix it, then: "
                  f"scripts/bench_all.py --only {b.name} --power-mode {args.power_mode}")
            return 1
        h = R.heat(json.loads((run_dir / b.out).read_text()).get("provenance"))
        print(f"{b.name}: done in {rec['seconds']} s | heat {R.heat_text(h)}", flush=True)
        if h and h["warn"]:
            print(f"⚠ {b.name} throttled ({', '.join(h['warn'])}): consider --only {b.name} once cooler")
        save(mpath, manifest)
    if not rerun:
        manifest["finished_utc"] = now()
        save(mpath, manifest)
    return 0


def render(run_dir: Path, results_md: Path, readme: Path, img_dir: Path = Path("docs/img"),
           train_curve: Path = Path("docs/data/train_curve.json"),
           load_sweep: Path = Path("docs/data/load_sweep.json")) -> int:
    try:
        data = R.load_run(run_dir)
        page = R.render_results(data, run_dir.as_posix())
        old = readme.read_text()
        new = R.render_readme(old, data)
    except R.MissingResult as e:
        print(e)
        return 1
    try:
        from tinyllm.plots import render_plots     # matplotlib is a dev dependency: `uv sync` installs it
    except ImportError as e:
        print(f"{e}: run `uv sync` (matplotlib is in the dev group) to draw the plots")
        return 1
    curve = json.loads(train_curve.read_text()) if train_curve.exists() else None
    load = json.loads(load_sweep.read_text()) if load_sweep.exists() else None
    try:
        svgs = render_plots(data, curve, load)
    except R.MissingResult as e:
        print(e)
        return 1
    results_md.write_text(page)
    readme.write_text(new)
    print(f"wrote {results_md} ({page.count(chr(10))} lines); README blocks "
          f"{', '.join(R.block_names(new))} {'updated' if new != old else 'unchanged'}")
    img_dir.mkdir(parents=True, exist_ok=True)
    changed = [name for name, svg in svgs.items()
               if not (img_dir / name).exists() or (img_dir / name).read_text() != svg]
    for name in changed:
        (img_dir / name).write_text(svgs[name])
    print(f"plots in {img_dir}: {len(changed)} of {len(svgs)} changed"
          + ("" if curve else f" ({train_curve} not found: no training plot; see scripts/extract_train_curve.py)")
          + ("" if load else f" ({load_sweep} not found: no load plot; see scripts/extract_load_sweep.py)"))
    for w in R.warnings(data):
        print("⚠", w)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, default=Path("runs/bench/v1"), help="the run directory")
    p.add_argument("--run", action="store_true", help="run every benchmark into a fresh --dir, then render")
    p.add_argument("--only", default=None, help="comma-separated benchmark names to rerun into --dir")
    p.add_argument("--power-mode", default=None, help="NitroSense mode, recorded (required to run)")
    p.add_argument("--cool-to", type=float, default=60.0, help="°C to wait for before each benchmark")
    p.add_argument("--cool-max-s", type=float, default=900.0, help="longest cool-down wait per benchmark")
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (recorded)")
    p.add_argument("--list", action="store_true", help="print the benchmarks and exit")
    p.add_argument("--results", type=Path, default=Path("docs/RESULTS.md"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    args = p.parse_args()
    os.chdir(REPO)

    if args.list:
        for b in R.BENCHES:
            env = "".join(f"{k}={v} " for k, v in b.env)
            print(f"{b.name:<13} ~{b.minutes:>2} min  {env}{' '.join(command(b, args.dir, False)[1:])}")
        print(f"total ~{sum(b.minutes for b in R.BENCHES)} min plus cool-downs")
        return 0
    if args.run and args.only:
        p.error("--run and --only are separate: --run starts a fresh run, --only reruns into an existing one")
    if args.run or args.only:
        if not args.power_mode:
            p.error("--power-mode is required to run (it can't be read from WSL); e.g. --power-mode performance")
        names = [x.strip() for x in args.only.split(",")] if args.only else [b.name for b in R.BENCHES]
        unknown = [x for x in names if x not in R.BY_NAME]
        if unknown:
            p.error(f"unknown benchmark(s) {unknown}; see --list")
        code = run(args, [R.BY_NAME[x] for x in names], args.dir, rerun=bool(args.only))
        if code != 0:
            return code
    return render(args.dir, args.results, args.readme)


if __name__ == "__main__":
    sys.exit(main())
