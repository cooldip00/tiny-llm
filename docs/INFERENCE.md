# Inference Engine & Serving at Scale

## Core mental model
| Stage | Work | Bottleneck |
|---|---|---|
| **Prefill** | Process the whole prompt in one forward pass | Compute (FLOPs) |
| **Decode** | One token per forward pass | **Memory bandwidth** (read all weights + KV cache per token) |

Nearly every serving optimization targets decode. They either read fewer bytes per token or share each read across more tokens (batching).

## Technique map
| Technique | Problem it solves | Our step |
|---|---|---|
| KV cache | Recomputing attention over past tokens | 4.3 |
| Static, then continuous batching | Idle GPU and requests of different lengths | 4.4 |
| PagedAttention (vLLM) | KV memory fragmentation, wasted reserved space | 4.5 |
| Prefix caching | Re-prefilling shared system prompts | 4.5 (stretch) |
| Weight quantization (INT8/INT4) | Bytes read per token | 4.6 |
| KV-cache quantization | KV bytes at long context | parked |
| Speculative decoding | Only one token per big-model pass | 4.7 |
| FlashAttention / FlashDecoding | Attention memory traffic | via SDPA |
| CUDA graphs | Kernel launch overhead at small batch | 4.x stretch |
| Tensor, pipeline and expert parallelism | Model bigger than one GPU | concept only |
| Disaggregated prefill/decode | Prefill interfering with decode latency | concept only |

## Steps (PLAN Phase 4)
1. **4.1 Naive:** a full forward pass over the whole sequence for every new token. Cost per step is O(n²).
2. **4.2 Sampling:** temperature, top-k, top-p (nucleus).
3. **4.3 KV cache:** preallocate `[B, n_kv_heads, max_len, head_dim]` per layer; prefill once, then decode one token at a time.
4. **4.4 Batching:** static batch with padding, then continuous batching (a scheduler that adds and removes sequences each step).
5. **4.5 Paged KV:** fixed-size blocks (e.g. 16 tokens), a free list and a per-sequence block table.
6. **4.6 INT8 weight-only:** per-channel absmax scales; measure tok/s and the val-loss delta.
7. **4.7 Speculative:** the `smoke` model drafts k tokens and `main` verifies them in one pass. Measure acceptance rate and speedup across k.
8. **4.8 Baselines:** export to HF Llama, then serve with vLLM and convert to GGUF for llama.cpp (Q8_0, Q4_K_M).
9. **4.9 TinyLlama 1.1B:** add GQA, load the pretrained weights, match the HF reference logits, rerun the benchmarks against the bandwidth ceiling.
10. **4.10 Triton kernels (timeboxed):** fused RMSNorm + residual, then INT8 weight-only GEMV.

## Bandwidth ceiling method (D-015)
Batch-1 decode reads every weight once per token, so:
**max decode tok/s ≈ measured bandwidth (GB/s) ÷ weight size (GB)**

| Model | Weights | Ceiling @ 448 GB/s (rated spec, replace with the Phase 0.5 measurement) |
|---|---|---|
| TinyLlama bf16 | ~2.2 GB | ~200 tok/s |
| TinyLlama INT8 | ~1.1 GB | ~400 tok/s |

Report every TinyLlama decode result as a **% of the ceiling**. Batching raises tokens per second because each weight read serves several sequences.

TinyLlama KV cache: `2 × 22 layers × 4 kv_heads × 64 head_dim × 2 B` = **~22 KiB/token** → 2048-token context ≈ 44 MiB per sequence.

## Reference-answer rule
Every optimized path must match the slow path before it replaces it:
- Our engine: naive generation (4.1) is the reference. Greedy outputs must be identical, or logits within tolerance.
- TinyLlama: HF Transformers logits are the reference (tests only, D-015).
- Triton kernels: the PyTorch implementation is the reference (`torch.testing.assert_close`).

## Benchmark protocol
- Fixed prompts: lengths 32 / 128 / 512 tokens; generate 256 tokens.
- Batch sizes: 1, 8, 32 (where they fit).
- Warmup: 3 runs, then `torch.cuda.synchronize()` around timed regions; median of 5.
- Report: prefill tok/s, decode tok/s, time to first token, peak VRAM (`torch.cuda.max_memory_allocated`).
- Same model checkpoint for every row.

## KV cache size, main config
`2 × 8 layers × 8 kv_heads × 64 head_dim × 512 seq × 2 B` = **8 MiB per sequence** (bf16).
With GQA and 2 KV heads it's 2 MiB. Hence GQA/MLA matter for long context and large batches.

## Results (fill in)
| Step | Batch | Prompt | Decode tok/s | TTFT (ms) | Peak VRAM | Notes |
|---|---|---|---|---|---|---|
| 4.1 naive | 1 | 128 | | | | |
| 4.3 KV cache | 1 | 128 | | | | |
| 4.4 static batch | 8 | 128 | | | | |
| 4.4 continuous | mixed | mixed | | | | |
| 4.6 INT8 | 1 | 128 | | | | val loss Δ: |
| 4.7 speculative (k=4) | 1 | 128 | | | | accept rate: |
| vLLM | 1 / 32 | 128 | | | | |
| llama.cpp Q8_0 | 1 | 128 | | | | |

## Results: TinyLlama 1.1B (fill in)
| Step | Batch | Prompt | Decode tok/s | % of ceiling | TTFT (ms) | Peak VRAM | Notes |
|---|---|---|---|---|---|---|---|
| HF Transformers `generate` | 1 | 128 | | | | | baseline |
| ours naive | 1 | 128 | | | | | |
| ours KV cache | 1 | 128 | | | | | |
| ours continuous batching | 16 | mixed | | n/a | | | |
| ours INT8 | 1 | 128 | | | | | quality Δ: |
| vLLM | 1 / 16 | 128 | | | | | |
| llama.cpp Q8_0 | 1 | 128 | | | | | |

## Results: Triton kernels (fill in)
| Kernel | Shape | PyTorch time | Triton time | Achieved GB/s | % of peak | End-to-end tok/s Δ |
|---|---|---|---|---|---|---|
| Fused RMSNorm + residual | | | | | | |
| INT8 GEMV (decode) | | | | | | |
