"""Step 4.12 (D-025): record where every benchmark number came from.

    prov = Provenance(weights="fp32 + bf16 autocast").start()   # after model load, before timing
    ... timed runs ...
    json_out["provenance"] = prov.finish()

Recorded:
  static   UTC time, git commit and whether tracked files are modified, python / torch / CUDA /
           cuDNN versions, GPU name and driver, plus any fields passed in (e.g. the weight dtype)
  before   GPU sample, WSL load average, Windows CPU load %
  during   GPU samples from one `nvidia-smi -lms` process running next to the benchmark:
           min / median / max of SM clock, memory clock, temperature, power, utilisation; every
           throttle reason seen; and the WSL load average
  after    same as before

Notes:
  - WSL's /proc/loadavg only sees the Linux VM. Load from Windows programs (the usual cause of
    the 4.5 outliers) shows up only in the Windows CPU %, read via powershell.exe.
  - That read costs ~2 s of CPU, so it runs only before and after, never during the timed region.
  - The GPU sampler is a separate process, so it costs the benchmark's Python process nothing.
    Every probe is optional: missing tools give None, never an error.
"""

from __future__ import annotations

import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

GPU_FIELDS = ["pstate", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
              "utilization.gpu", "memory.used", "clocks_throttle_reasons.active"]
NUMERIC = ["clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu", "memory.used"]
BUSY_UTIL = 10  # % GPU utilisation at or above which a sample counts as "busy"

# NVML clocksThrottleReasons bits (nvml.h)
THROTTLE_BITS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}
REPO = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def claim_output(path: str | Path, overwrite: bool = False) -> Path:
    """Call before a benchmark starts: refuse an existing result file unless overwrite (6.8).

    The 11-04 TinyLlama serving JSON was lost because a later run with other flags wrote the same
    default name. Checking at startup costs nothing; finding out after an hour's run costs the hour.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists: pass a new --out, or --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def decode_throttle(mask: int | None) -> list[str]:
    if mask is None:
        return []
    names = [name for bit, name in THROTTLE_BITS.items() if mask & bit]
    rest = mask & ~sum(THROTTLE_BITS)
    return names + ([f"unknown_0x{rest:x}"] if rest else [])


def parse_gpu_line(line: str) -> dict | None:
    """One `nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits` line -> dict."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != len(GPU_FIELDS):
        return None
    out: dict = {}
    for k, v in zip(GPU_FIELDS, parts):
        if v in ("[N/A]", "N/A", "[Not Supported]", ""):
            out[k] = None
        elif k == "pstate":
            out[k] = v
        elif k == "clocks_throttle_reasons.active":
            out[k] = int(v, 16)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
    return out


def _smi() -> str | None:
    return shutil.which("nvidia-smi")


def gpu_sample() -> dict | None:
    smi = _smi()
    if smi is None:
        return None
    out = _run([smi, f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    return parse_gpu_line(out.splitlines()[0]) if out else None


def wsl_loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def windows_cpu_percent() -> float | None:
    ps = shutil.which("powershell.exe")
    if ps is None:
        return None
    out = _run([ps, "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
               timeout=20)
    try:
        return float(out) if out else None
    except ValueError:
        return None


def host_sample() -> dict:
    return {"gpu": gpu_sample(), "wsl_loadavg_1m": wsl_loadavg(), "windows_cpu_percent": windows_cpu_percent()}


def summarize(samples: list[dict], loads: list[float]) -> dict:
    """min / median / max per numeric GPU field, the throttle reasons seen, and the WSL load."""
    out: dict = {"n_samples": len(samples)}
    for k in NUMERIC:
        vals = [s[k] for s in samples if s.get(k) is not None]
        out[k] = ({"min": min(vals), "median": statistics.median(vals), "max": max(vals)} if vals else None)
    # Idle gaps (startup, CPU-only work between runs) drag the minimum down; this is the clock while working.
    busy = [s["clocks.sm"] for s in samples
            if s.get("clocks.sm") is not None and (s.get("utilization.gpu") or 0) >= BUSY_UTIL]
    out["clocks.sm_busy"] = ({"min": min(busy), "median": statistics.median(busy), "max": max(busy),
                              "n": len(busy)} if busy else None)
    mask = 0
    for s in samples:
        mask |= s.get("clocks_throttle_reasons.active") or 0
    out["throttle_reasons_seen"] = decode_throttle(mask)
    out["pstates_seen"] = sorted({s["pstate"] for s in samples if s.get("pstate")})
    out["wsl_loadavg_1m"] = ({"min": min(loads), "max": max(loads)} if loads else None)
    return out


class GpuMonitor:
    """Streams `nvidia-smi -lms <interval>` in a background thread (one process for the whole run)."""

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self.loads: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> GpuMonitor:
        smi = _smi()
        if smi is None:
            return self
        try:
            self._proc = subprocess.Popen(
                [smi, f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
                 f"-lms={self.interval_ms}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            s = parse_gpu_line(line)
            if s is not None:
                self.samples.append(s)
                load = wsl_loadavg()
                if load is not None:
                    self.loads.append(load)

    def stop(self) -> dict:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return summarize(self.samples, self.loads)


class Provenance:
    def __init__(self, interval_ms: int = 500, **extra):
        self.record: dict = {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_info(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "driver": _run([_smi(), "--query-gpu=driver_version", "--format=csv,noheader"]) if _smi() else None,
            "cpu_count": os.cpu_count(),
            **extra,
        }
        self._monitor = GpuMonitor(interval_ms)
        self._t0 = 0.0

    def start(self) -> Provenance:
        self.record["before"] = host_sample()
        self._monitor.start()
        self._t0 = time.perf_counter()
        return self

    def finish(self) -> dict:
        self.record["monitored_s"] = round(time.perf_counter() - self._t0, 2)
        self.record["during"] = self._monitor.stop()
        self.record["after"] = host_sample()
        return self.record
