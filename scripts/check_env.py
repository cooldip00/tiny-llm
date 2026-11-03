"""Step 0.5: environment check and GPU benchmark.

Prints the software stack, the GPU, bf16 support, measured matmul TFLOPS,
memory bandwidth, and a sustained-load run (to see laptop thermal throttling).

Run:  uv run python scripts/check_env.py            # ~1 minute
      uv run python scripts/check_env.py --sustain 0  # skip the sustained run
"""

import argparse
import platform
import subprocess
import time

import torch

# RTX 3070 Ti Laptop: 256-bit GDDR6 at 14 Gbps -> 448 GB/s (spec sheet).
RATED_BW_GBS = 448.0
# Ampere GeForce (sm_86) tensor cores: 512 dense bf16 FLOPs/clock/SM with fp32 accumulate.
BF16_FLOPS_PER_CLK_PER_SM = 512


def smi(fields: str) -> list[str] | None:
    """Query nvidia-smi (comes from the Windows driver inside WSL)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return [x.strip() for x in out.splitlines()[0].split(",")]
    except Exception:
        return None


def time_cuda(fn, iters: int) -> float:
    """Average seconds per call, measured with CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def matmul_tflops(n: int, dtype: torch.dtype, iters: int = 20) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    c = torch.empty(n, n, device="cuda", dtype=dtype)
    for _ in range(3):  # warmup: cuBLAS picks its kernel
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    t = time_cuda(lambda: torch.matmul(a, b, out=c), iters)
    return 2 * n**3 / t / 1e12


def bandwidth_gbs(gib: float = 1.0, iters: int = 20) -> tuple[float, float]:
    """(copy GB/s, read-only GB/s). Copy = read + write; read-only ~ what LLM decode does."""
    n = int(gib * 2**30 / 4)
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
        x.sum()
    torch.cuda.synchronize()
    t_copy = time_cuda(lambda: y.copy_(x), iters)
    t_read = time_cuda(lambda: x.sum(), iters)
    nbytes = x.numel() * 4
    return 2 * nbytes / t_copy / 1e9, nbytes / t_read / 1e9


def sustained(seconds: float, n: int = 8192, report_every: float = 5.0) -> list[float]:
    """Run bf16 matmuls nonstop; print TFLOPS, temperature, power, clock per window."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    flops = 2 * n**3
    results = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        count, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < report_every:
            for _ in range(10):
                torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            count += 10
        tflops = count * flops / (time.perf_counter() - t0) / 1e12
        results.append(tflops)
        s = smi("temperature.gpu,power.draw,clocks.sm") or ["?", "?", "?"]
        print(f"  t={len(results) * report_every:4.0f}s  {tflops:5.1f} TFLOPS  "
              f"{s[0]} C  {s[1]} W  {s[2]} MHz")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sustain", type=float, default=30, help="seconds of sustained load (0 = skip)")
    args = p.parse_args()

    print("== Software ==")
    print(f"python        {platform.python_version()}")
    print(f"torch         {torch.__version__} (CUDA {torch.version.cuda})")
    try:
        import triton
        print(f"triton        {triton.__version__}")
    except ImportError:
        print("triton        not installed")
    drv = smi("driver_version")
    print(f"driver        {drv[0] if drv else '?'}")

    assert torch.cuda.is_available(), "CUDA not available"
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print("\n== GPU ==")
    print(f"name          {props.name}")
    print(f"capability    sm_{cap[0]}{cap[1]}")
    print(f"SMs           {props.multi_processor_count}")
    print(f"memory        {props.total_memory / 2**30:.2f} GiB")
    print(f"bf16          {torch.cuda.is_bf16_supported()}")
    lim = smi("power.limit,clocks.max.sm,temperature.gpu")
    if lim:
        print(f"power limit   {lim[0]} W   max SM clock {lim[1]} MHz   idle temp {lim[2]} C")

    peak = None
    if lim and cap == (8, 6):
        peak = props.multi_processor_count * float(lim[1]) * 1e6 * BF16_FLOPS_PER_CLK_PER_SM / 1e12
        print(f"bf16 peak     {peak:.1f} TFLOPS (theoretical, at max clock)")

    print("\n== Matmul (square n x n) ==")
    best = 0.0
    for n in (1024, 2048, 4096, 8192):
        tf = matmul_tflops(n, torch.bfloat16)
        best = max(best, tf)
        pct = f"  ({tf / peak:.0%} of peak)" if peak else ""
        print(f"bf16  n={n:<5} {tf:6.1f} TFLOPS{pct}")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"tf32  n=4096  {matmul_tflops(4096, torch.float32):6.1f} TFLOPS  (fp32 inputs, for reference)")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"fp32  n=4096  {matmul_tflops(4096, torch.float32):6.1f} TFLOPS  (no tensor cores)")

    print("\n== Memory bandwidth (1 GiB buffers) ==")
    bw_copy, bw_read = bandwidth_gbs()
    print(f"copy          {bw_copy:6.1f} GB/s  ({bw_copy / RATED_BW_GBS:.0%} of rated {RATED_BW_GBS:.0f})")
    print(f"read          {bw_read:6.1f} GB/s  ({bw_read / RATED_BW_GBS:.0%} of rated {RATED_BW_GBS:.0f})")

    sus = []
    if args.sustain > 0:
        print(f"\n== Sustained bf16 n=8192 for {args.sustain:.0f}s ==")
        sus = sustained(args.sustain)

    print("\n== Summary (copy into docs/HARDWARE.md) ==")
    print(f"bf16 matmul best      {best:.1f} TFLOPS")
    if sus:
        print(f"bf16 sustained        {sum(sus[1:] or sus) / len(sus[1:] or sus):.1f} TFLOPS "
              f"(min {min(sus):.1f}, max {max(sus):.1f})")
    print(f"bandwidth copy/read   {bw_copy:.0f} / {bw_read:.0f} GB/s")


if __name__ == "__main__":
    main()
