# Hardware & Budget

## Specs (measured 2026-11-03 on the Windows host)
| Item | Value |
|---|---|
| Laptop | Acer Nitro V 15 |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU (Ampere) |
| VRAM | 8192 MiB |
| Driver | 610.91 |
| System RAM | 16 GB DDR4 |
| OS | Windows 11 Home 10.0.26300 |
| Host Python | 3.12.6 |
| Host nvcc | 12.4 (not used by the PyTorch wheels) |
| bf16 | Supported (Ampere) · FP8: **not** supported |
| GPU power cap | 140 W base in `nvidia-smi`; draws up to ~150 W under load (Dynamic Boost) |
| SMs / max SM clock | 46 SMs / 1785 MHz |
| CUDA version the driver supports | 13.3 |

## WSL runtime (measured 2026-11-03)
| Item | Value |
|---|---|
| WSL | 2.6.1.0, kernel 6.6.87.2-1 |
| Distro | Ubuntu 24.04.5 LTS, user `kuladip` |
| Distro disk | `D:\WSL\Ubuntu-24.04` (on D:, which has 167 GB free) |
| RAM visible to WSL | 7.6 GiB (default 50%; raise with `.wslconfig` if needed) |

## Measured performance (2026-11-03, `scripts/check_env.py`, plugged in, performance mode)
Theoretical bf16 peak = 46 SMs × 1785 MHz × 512 FLOPs/clk/SM = **42.0 TFLOPS**. Rated bandwidth = **448 GB/s**.

| Metric | Value |
|---|---|
| bf16 matmul, n=8192 | **39.7 TFLOPS** (94% of peak) |
| bf16 matmul, n=4096 / 2048 / 1024 | 30.0 / 29.3 / 20.3 TFLOPS (smaller GEMMs reach less of peak) |
| bf16 sustained, 30 s at n=8192 | **38.3 TFLOPS** avg (39.3 → 37.9); 70 → 81 °C; 1635–1755 MHz; ~126–150 W |
| tf32 / plain fp32, n=4096 | 19.3 / 9.9 TFLOPS (this is why we train in bf16) |
| Memory bandwidth, device copy | **410 GB/s** (91% of rated) |
| Memory bandwidth, read-only | **423 GB/s** (94% of rated). Use this as the decode ceiling |
| Sustained training MFU (from Phase 3) | _TBD_ |

Takeaways:
- Small GEMMs reach much less of peak. Our 30M model's matmuls are small, so expect a real training MFU of ~30–50% of 38 TFLOPS.
- Decode ceiling = 423 GB/s ÷ weight bytes. TinyLlama bf16 (~2.2 GB) → **~190 tok/s** at batch 1. Our 30M model (60 MB) → ~7,000 tok/s on paper, so in practice it will be limited by kernel-launch overhead, not bandwidth.
- Temperature was still rising after 30 s, so watch it during long runs.

## Compute budget
Training FLOPs ≈ **6 × N (params) × D (tokens)**.

Planning assumption was ~12–19 TFLOPS effective. **Measured in Step 3.5** (PaLM FLOPs convention, vs the 42.0 TFLOPS peak): smoke 13.1 TFLOPS (**31% MFU**, ~135K tok/s), main 15.0 TFLOPS (**36% MFU**, ~73K tok/s). Those are 1-minute numbers. **Sustained over the 35-min smoke run (Step 3.6), throughput is ~8% lower** as the GPU heats up: ~124K tok/s, 28.6% MFU. Eager main would sustain ~67K tok/s (1B tokens ≈ 4.2 h). **With `torch.compile` (D-020), main reaches 114K tok/s, 55.5% MFU** over a 2-minute test. Allowing ~8% for heat, I expected ~105K sustained. **Actual main run (Step 3.7): 115.3K tok/s mean (111.6–128.3K), 56.1% MFU, 23.6 TFLOPS sustained over 2 h 27 min for 996M tokens.** Heat cost only ~3% when compiled.

| Model | Tokens | FLOPs | Est. time |
|---|---|---|---|
| ~14M (smoke) | 300M | 2.5e16 | ~0.4–0.6 h |
| ~30M (main) | 1B | 1.8e17 | ~2.6–4 h |
| 124M (parked) | 2.5B | 1.9e18 | ~28–44 h |

## Training memory (rule of thumb)
With AdamW and mixed precision (fp32 master weights), per parameter:
- weights 4 B + grads 4 B + Adam m and v 8 B = **~16 B/param**
- 30M params comes to about 0.5 GB. **Activations dominate**, so control them with micro-batch size and gradient accumulation.

## Inference memory
- Weights: params × bytes (bf16 = 2, int8 = 1, int4 = 0.5)
- KV cache per sequence = `2 × n_layers × n_kv_heads × head_dim × seq_len × bytes`
- Decode speed is limited by memory bandwidth: each token reads all the weights plus the KV cache.

## Laptop operating notes
- Stay plugged in, use the Nitro performance mode, raise the rear for airflow.
- Watch temperature with `nvidia-smi -l 5`. Sustained throttling hurts throughput, so record it if it happens.
- Inside WSL, CUDA comes from the Windows driver. Don't install a Linux NVIDIA driver.
- WSL default RAM limit is 50% of host RAM. Adjust in `%UserProfile%\.wslconfig` if data processing needs more.
