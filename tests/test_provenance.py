"""Step 4.12: benchmark provenance. Pure parsing and summaries, plus graceful fallbacks without tools."""

import re

from tinyllm import provenance as pv

LINE = "P2, 1785, 7000, 61, 72.40, 51, 812, 0x0000000000000004"


def test_parse_gpu_line():
    s = pv.parse_gpu_line(LINE)
    assert s == {"pstate": "P2", "clocks.sm": 1785.0, "clocks.mem": 7000.0, "temperature.gpu": 61.0,
                 "power.draw": 72.4, "utilization.gpu": 51.0, "memory.used": 812.0,
                 "clocks_throttle_reasons.active": 4}


def test_parse_handles_not_available_and_bad_lines():
    s = pv.parse_gpu_line("P8, 210, 405, 43, [N/A], 0, 0, 0x0000000000000001")
    assert s["power.draw"] is None and s["clocks.sm"] == 210.0
    assert pv.parse_gpu_line("garbage") is None


def test_decode_throttle():
    assert pv.decode_throttle(0x1 | 0x4 | 0x40) == ["gpu_idle", "sw_power_cap", "hw_thermal_slowdown"]
    assert pv.decode_throttle(0) == [] and pv.decode_throttle(None) == []
    assert pv.decode_throttle(0x1000) == ["unknown_0x1000"]


def test_summarize():
    a = pv.parse_gpu_line(LINE)
    b = dict(a, **{"clocks.sm": 1500.0, "clocks_throttle_reasons.active": 0x20, "power.draw": None})
    c = dict(a, **{"clocks.sm": 1650.0, "pstate": "P0"})
    out = pv.summarize([a, b, c], loads=[0.5, 1.5])
    assert out["n_samples"] == 3
    assert out["clocks.sm"] == {"min": 1500.0, "median": 1650.0, "max": 1785.0}
    assert out["power.draw"] == {"min": 72.4, "median": 72.4, "max": 72.4}   # None values skipped
    assert out["throttle_reasons_seen"] == ["sw_power_cap", "sw_thermal_slowdown"]
    assert out["pstates_seen"] == ["P0", "P2"]
    assert out["wsl_loadavg_1m"] == {"min": 0.5, "max": 1.5}
    idle = dict(a, **{"clocks.sm": 210.0, "utilization.gpu": 0.0})
    busy = pv.summarize([a, idle], [])["clocks.sm_busy"]
    assert busy == {"min": 1785.0, "median": 1785.0, "max": 1785.0, "n": 1}      # the idle sample is ignored
    assert pv.summarize([], [])["clocks.sm"] is None and pv.summarize([], [])["clocks.sm_busy"] is None


def test_git_info_in_repo():
    g = pv.git_info()
    assert re.fullmatch(r"[0-9a-f]{40}", g["commit"]) and isinstance(g["dirty"], bool)


def test_provenance_without_any_tools(monkeypatch):
    # No nvidia-smi and no powershell.exe (e.g. CI): every probe gives None, nothing raises.
    monkeypatch.setattr(pv.shutil, "which", lambda name: None)
    rec = pv.Provenance(weights="bf16").start().finish()
    assert rec["weights"] == "bf16" and rec["driver"] is None
    assert rec["before"] == {"gpu": None, "wsl_loadavg_1m": rec["before"]["wsl_loadavg_1m"],
                             "windows_cpu_percent": None}
    assert rec["during"]["n_samples"] == 0 and rec["during"]["throttle_reasons_seen"] == []
    assert rec["git"]["commit"] is not None
