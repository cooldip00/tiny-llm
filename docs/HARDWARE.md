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
| GPU power cap | 140 W (from `nvidia-smi`) |
| CUDA version the driver supports | 13.3 |

## WSL runtime (measured 2026-11-03)
| Item | Value |
|---|---|
| WSL | 2.6.1.0, kernel 6.6.87.2-1 |
| Distro | Ubuntu 24.04.5 LTS, user `kuladip` |
| Distro disk | `D:\WSL\Ubuntu-24.04` (on D:, which has 167 GB free) |
| RAM visible to WSL | 7.6 GiB (default 50%; raise with `.wslconfig` if needed) |

## Measured performance (fill in during Phase 0.5)
| Metric | Value |
|---|---|
| bf16 matmul TFLOPS (large square GEMM) | _TBD_ |
| Memory bandwidth (GB/s, device copy) | _TBD_ |
| Sustained training MFU (from Phase 3) | _TBD_ |

## Compute budget
Training FLOPs ≈ **6 × N (params) × D (tokens)**.

Planning assumption is ~10–15 TFLOPS effective. Replace it with the measured number.

| Model | Tokens | FLOPs | Est. time |
|---|---|---|---|
| ~14M (smoke) | 300M | 2.5e16 | ~0.5–1 h |
| ~30M (main) | 1B | 1.8e17 | ~4–5 h |
| 124M (parked) | 2.5B | 1.9e18 | ~40–50 h |

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
