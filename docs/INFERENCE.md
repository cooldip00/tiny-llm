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
- Provenance (4.12, D-025): every `runs/bench/*.json` written from 4.12 on has a `"provenance"` key (`src/tinyllm/provenance.py`):
  - static: git commit + dirty flag, python / torch / CUDA / cuDNN, GPU + driver, weight dtype, script
  - `before` / `after`: one GPU sample, WSL load average, Windows CPU % (via `powershell.exe`, ~2 s, so never inside the timed region)
  - `during`: one `nvidia-smi -lms 500` process beside the run: min / median / max SM clock, memory clock, temperature, power, utilisation; `clocks.sm_busy` (samples at ≥ 10% utilisation only); every NVML throttle reason seen
  - `bench_serving.py` also stores `gpu_during` per row, so an outlier row (as in 4.5) can be matched to clocks or throttling
  - WSL's `/proc/loadavg` sees only the Linux VM; load from Windows programs shows up only in the Windows CPU %
  - First reading (2026-11-04 smoke runs): during B=1 KV decode the GPU sits at **~860 MHz SM, ~27 W**, with only `gpu_idle` set (boost is ~1,875 MHz; a heavy matmul holds ~1,350 MHz at the 135 W power cap). The decode step is so launch-bound that the GPU downclocks between kernels. More evidence for 4.11.

## KV cache size, main config
`2 × 8 layers × 8 kv_heads × 64 head_dim × 512 seq × 2 B` = **8 MiB per sequence** (bf16).
With GQA and 2 KV heads it's 2 MiB. Hence GQA/MLA matter for long context and large batches.

## Results (fill in)
_This page is a dated history: each table is a snapshot from its step. The current numbers, all from one run and one session, are in [RESULTS.md](RESULTS.md), generated by `scripts/bench_all.py` (D-029)._

| Step | Batch | Prompt | Decode tok/s | TTFT (ms) | Peak VRAM | Notes |
|---|---|---|---|---|---|---|
| 4.1 naive | 1 | 128 | 104 | 9.1 | 199 MiB | ~9.6 ms/step, flat across prompt lengths: overhead-bound (see below) |
| 4.3 KV cache | 1 | 128 | 97 | 10.7 | 195 MiB | 0.93× vs 4.1 at B=1 (overhead-bound); **3.1×** at B=32 P=128 (see below) |
| 4.4 static batch | 8 | mixed¹ | 391 | 9,870 (mean, incl. queueing) | 357 MiB | 64-request mixed workload (see below); 32 rows: 1,281 tok/s |
| 4.4 continuous | 8 slots | mixed | 525 | 6,090 (mean, incl. queueing) | 319 MiB | **1.34×** static at 8, 1.12× at 32 (1,434 tok/s); p95 latency −33% / −27% |
| 4.5 paged KV (block 16) | 32 slots | mixed | 924 (vs 633 for 8 slots, same run) | 2,580 (mean, incl. queueing) | 458 MiB | **same 64 MiB of KV** as 8 slots: **1.46×**, 25 sequences at once instead of 8; 97% of claimed KV holds real tokens (vs 52%) |
| 4.6 INT8 | 1 | 128 | 82 (vs 106 fp32, 135 bf16 weights; same run) | 12.4 | weights: matmul 28.7 MiB (vs 57 bf16) | val loss Δ: **+0.0001** (1.2825 → 1.2826). **0.78×** speed at B=1: overhead-bound, and three PyTorch ops per Linear (see 4.6 below) |
| 4.7 speculative (k=4) | 1 | 128 | | | | accept rate: |
| vLLM | 1 / 32 | 128 | | | | |
| llama.cpp Q8_0 | 1 | 128 | | | | |

¹ The row was planned as "8 × 128". It was measured on the mixed 64-request workload instead, because equal prompt and output lengths are exactly the case where static and continuous batching do the same work.

### 4.1 naive: full grid (2026-11-04)
`scripts/bench_generate.py`, `checkpoints/main/best.pt` (29.9M params, val 1.2825), bf16 autocast, greedy, 256 new tokens, median of 5 after 3 warmups. Raw numbers: `runs/bench/4.1_naive.json`. The timed loop is checked against `sample(temperature=0)` before timing.

| Batch | Prompt | TTFT (ms) | Prefill tok/s | Decode tok/s | ms/step | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|
| 1 | 32 | 9.75 | 3,281 | 101 | 9.92 | 194 |
| 1 | 128 | 9.06 | 14,134 | 104 | 9.58 | 199 |
| 1 | 512* | 10.36 | 49,403 | 97 | 10.33 | 203 |
| 8 | 32 | 11.02 | 23,223 | 800 | 10.00 | 270 |
| 8 | 128 | 10.92 | 93,734 | 742 | 10.79 | 298 |
| 8 | 512* | 18.80 | 217,839 | 446 | 17.95 | 331 |
| 32 | 32 | 10.29 | 99,484 | 1,273 | 25.13 | 519 |
| 32 | 128 | 19.44 | 210,668 | 891 | 35.90 | 628 |
| 32 | 512* | 70.93 | 230,976 | 453 | 70.68 | 777 |

\* prompt + 256 > max_seq_len 512, so the window slides: every step is a full 512-token forward.

Reading (not yet profiled):
- **Batch 1 is overhead-bound, not bandwidth-bound.** ~10 ms/step whether the context is 32 or 512 tokens. The bf16 weights are ~60 MB, so the bandwidth ceiling at the measured 423 GB/s is ~7,000 tok/s; 104 tok/s is ~1.5% of it. The time is likely Python, kernel launches and autocast re-casting the fp32 weights on every forward.
- **Consequence for 4.3:** at batch 1 on this small model, a KV cache alone may give little speedup, because the per-step cost is not the attention work. The gain should show at long context and large batch, where the step time does grow (B=32 P=512: 70.7 ms/step). TinyLlama (4.9) is where the bandwidth ceiling becomes the real limit.
- **Batching helps until compute catches up:** decode tok/s rises ~8× from B=1 to B=8 at P=32 (same ms/step), then ms/step grows with B×context.
- One transient allocator OOM warning (`free: 0`) appeared during the B=32 P=128 row; nvidia-smi showed 0 MiB in use before and after. A clean rerun gave 873 tok/s decode (vs 891), so the row stands.
- Ceiling note: the table above still uses the 448 GB/s rated spec; the Phase 0.5 measurement is 423 GB/s, which gives TinyLlama ceilings of ~190 (bf16) and ~380 (INT8) tok/s.

### 4.3 KV cache: full grid (2026-11-04)
`scripts/bench_generate.py --engine kv`, same checkpoint and protocol as 4.1. Raw numbers: `runs/bench/kv.json`. Code: `src/tinyllm/kv_cache.py`, `generate.generate()`.

| Batch | Prompt | TTFT (ms) | Prefill tok/s | Decode tok/s | ms/step | Peak VRAM (MiB) | vs 4.1 decode |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 10.42 | 3,072 | 97 | 10.27 | 193 | 0.96× |
| 1 | 128 | 10.65 | 12,024 | 97 | 10.26 | 195 | 0.93× |
| 8 | 32 | 9.52 | 26,900 | 808 | 9.91 | 253 | 1.01× |
| 8 | 128 | 10.65 | 96,156 | 831 | 9.62 | 274 | 1.12× |
| 32 | 32 | 10.22 | 100,212 | 2,976 | 10.75 | 466 | **2.3×** |
| 32 | 128 | 21.88 | 187,176 | 2,767 | 11.56 | 526 | **3.1×** |
| any | 512 | — | — | — | — | — | n/a: 512 + 256 > max_seq_len, and a cache can't slide the window (RoPE positions are baked into cached keys) |

Reading:
- **As predicted by 4.1: no gain at batch 1.** The step time stays ~10 ms, so the work the cache removes was never the bottleneck there. The cache pays off where naive's step time grew with B × context: 3.1× at B=32 P=128.
- **Now every row is overhead-bound:** ~10–11.5 ms/step from B=1 to B=32. The next B=1 win is cutting per-step overhead (CUDA graphs / `torch.compile`, bf16 weights instead of autocast re-casting; PARKING_LOT), not more attention work.
- **Correctness:**
  - fp32 (CPU tests): greedy output **identical** to naive `sample()`, including batches, chunked prefill and seeded sampling with top-k/top-p (`tests/test_kv_cache.py`).
  - bf16 (GPU, 8×128 prompts, 256 tokens): 5/8 rows identical; 3 rows diverge at tokens 51, 78 and 239 (81% of tokens equal overall). Teacher-forced on the same tokens, max |logit diff| is 0.125 at a logit scale of 24.1 (about one bf16 step at that size), and argmax agrees on 99.76% of positions. So the divergences are near-ties flipped by bf16 rounding, then compounded by greedy feedback. This meets the reference-answer rule ("identical, or logits within tolerance").
- **Bug found by the benchmark:** the first version had a reference cycle (cache ↔ per-layer objects), so each finished cache's GPU memory stayed alive until Python's cycle collector ran. Peak VRAM at B=32 P=32 was 2,588 MiB, about 8 caches' worth (3 warmups + 5 runs). After removing the cycle it's 466 MiB, below naive's 519. A test now checks that a cache is freed by refcount alone.

### 4.4 Static vs continuous batching (2026-11-04)
`scripts/bench_serving.py`, same checkpoint. Raw numbers: `runs/bench/serving.json`. Code: `src/tinyllm/engine.py` (`run_static`, `run_continuous`), `kv_cache.SlotKVCache`.

Workload: 64 requests (seed 0), all queued at t=0 (offline: no arrival process yet). Prompts are validation windows of 33–252 tokens, `max_new_tokens` uniform 16–256 (9,004 tokens in total), greedy, no EOS. Latency and TTFT are measured from engine start, so they include queueing.

| Engine | Size | Useful tok/s | Wall (s) | Latency mean / p95 (s) | TTFT mean / p95 (s) | Decode steps | Row util | KV reserved (MiB) | KV used mean / peak | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| static | 8 | 391 | 23.01 | 11.55 / 22.08 | 9.87 / 20.00 | 1,892 | 59% | 62.6 | 31% / 49% | 357 |
| continuous | 8 | **525** | 17.14 | **7.94 / 14.79** | 6.09 / 13.05 | 1,295 | **86%** | 64.0 | 41% / 59% | 319 |
| static | 32 | 1,281 | 7.03 | 3.83 / 6.86 | 1.96 / 3.86 | 510 | 55% | 254.0 | 27% / 39% | 851 |
| continuous | 32 | **1,434** | 6.28 | **2.79 / 4.99** | 0.85 / 2.62 | 434 | 64% | 256.0 | 31% / 52% | 716 |

How the two engines work:
- **Static:** take requests in order, B at a time. Left-pad the prompts to the longest and run one prefill. A per-row key mask hides the pad keys. Pad queries may see only themselves, so no softmax row is empty (NaN). Then decode until the batch's *longest* request is done. No per-row RoPE positions are needed: padding shifts every real token of a row equally, and RoPE only sees distances.
- **Continuous:** S slots in a `SlotKVCache`, each with its own length. Every step, queued requests are admitted into free slots (one batched, padded prefill for all newcomers, copied into their slots). Then all slots decode one token together, each at its own position (per-row RoPE, scatter write, per-slot key mask). A finished request frees its slot at once.

Reading:
- **Row utilisation explains the gain.** Static computes rows for finished requests until the slowest one in the batch ends: 59% of decode rows were useful at B=8. Continuous refills them: 86%, and 1,295 instead of 1,892 decode steps (−32%).
- **Throughput gains less than the step count:** 1.34×, not 1.46×. A continuous step costs ~13.2 ms vs ~12.2 ms, because admission prefills run between decode steps and the per-slot mask and scatter add work. Admission is a blocking prefill, so running requests stall while newcomers prefill. Chunked prefill would fix this; it is out of scope here.
- **At 32 slots the gain shrinks to 1.12×.** 64 requests are only 2 waves of 32, so the run ends in a long drain tail where slots sit empty and nothing is queued (row util 64%). A longer queue or an arrival process would show more. Latency still improves (p95 −27%, mean TTFT 1.96 → 0.85 s).
- **KV memory is the next waste (→ 4.5).** Every slot reserves `max_seq_len` = 512 positions (8 MiB in bf16) whatever its request needs. On average only 41% (8 slots) and 31% (32 slots) of the reserved KV memory holds real tokens. Static is no better: it reserves `B × (longest prompt + longest output)` per batch. Paged KV allocates fixed-size blocks on demand, so reserved ≈ used, and that memory can hold more concurrent sequences.
- **Correctness:**
  - fp32 (CPU tests): both engines give **identical** greedy tokens to each request's solo `generate()`. This covers batch sizes 1/3/8, slot counts 1/3/10 (including more slots than requests), EOS freeing a slot mid-run, and sequences joining a running slot batch at different times (per-step logits checked against the full forward). See `tests/test_engine.py`, 12 tests.
  - bf16 (GPU): 39–43 of 64 requests are token-identical to solo, depending on the engine. At every divergence, the fp32 top-1/top-2 logit gap of the solo prefix is ≤ 0.076, while the median gap over all generated positions is 1.84 and only 3.9% of positions are below 0.125. So every divergence is a near-tie flipped by bf16 rounding (batched GEMMs pick different kernels than B=1), the same effect as in 4.3. With ~140 tokens per request, a flip somewhere in about a third of requests is expected.
- **Rerun note (added in 4.5):** the 4.5 benchmark reran these four rows with 5 repeats each (median run reported, spread < 5%). Absolute tok/s came out 15–20% higher than the first run, on the same code, because machine state differs between sessions. The ratios are what transfer: continuous/static was **1.37×** at 8 (633 vs 461 tok/s), but only **1.05×** at 32 (1,785 vs 1,702) instead of 1.12×. So the 32-slot gain is close to session noise; the latency gains (p95 −16%, mean TTFT −46%) are the more robust result there.

### 4.5 Paged KV cache (2026-11-04)
`scripts/bench_serving.py --paged --repeats 5`: the same 64-request workload, with the median of 5 runs. Raw numbers: `runs/bench/paged.json`. Code: `src/tinyllm/paged_kv.py` (`BlockAllocator`, `PagedKVCache`), `engine.run_continuous(kv="paged")`.

Design (vLLM-style; the analogy is virtual memory):
- **Pool and blocks.** One pool per layer, `[n_blocks, kv_heads, block_size, head_dim]`. A **block table** per sequence maps logical block j to a physical block (the page table). Token t lives at `(table[t // bs], t % bs)`, and all layers share one table.
- **Allocation.** Blocks come from a free list as a sequence grows: one more block each time it crosses a block boundary. Every block has the same size, so a freed block fits any sequence. There is no external fragmentation; at most `bs − 1` positions per sequence are wasted, in its last, partly filled block.
- **Padding is not stored.** Only real tokens go into blocks. The prefill's left-pad count becomes a RoPE offset, so positions (and outputs) stay identical.
- **Admission and preemption.** A request is admitted if the free blocks cover its context plus one token. When a decode step needs more blocks than are free, the **newest** sequence is preempted: its blocks are freed, and it goes back to the front of the queue as prompt + tokens so far, to be recomputed (vLLM's default recompute policy). Greedy recompute reproduces the same tokens (tested).
- **Reads.** Attention gathers each sequence's blocks into a contiguous view, then calls SDPA. vLLM's PagedAttention kernel reads blocks in place; the gather is this toy's extra copy.

Workload (all rows): the same 64 requests, 9,004 tokens to generate. The **KV budget for the paged rows = the 8-slot cache's memory**: 4,096 token positions, 64 MiB in bf16.

| Engine | Slots | Block | KV (MiB) | Useful tok/s | Wall median (s) [5 runs] | Latency mean / p95 (s) | Peak active | Preemptions | Pool used mean / peak | Claimed KV holding tokens | Row util |
|---|---|---|---|---|---|---|---|---|---|---|---|
| slots (4.4) | 8 | — | 64 | 633 | 14.23 [14.1–14.4] | 6.61 / 12.24 | 8 | 0 | 41% / 59% | 52% | 86% |
| paged | 8 | 16 | 64 | 592 | 15.20 [15.0–15.4] | 7.07 / 13.07 | 8 | 0 | 41% / 59% | **97%** | 86% |
| paged | 32 | 16 | 64 | **924** | 9.75 [9.4–10.5] | 4.73 / 8.54 | **25** | 42 | **76% / 98%** | 97% | 40% |
| paged | 32 | 8 | 64 | **975** | 9.23 [9.1–9.9] | 4.26 / 7.98 | 26 | 48 | 78% / 99% | 99% | 41% |
| paged | 32 | 32 | 64 | 803 | 11.22 [8.8–14.6]² | 4.95 / 9.66 | 25 | 40 | 74% / 96% | 94% | 38% |
| paged | 64 | 16 | 64 | 745 | 12.08 [10.6–20.9]² | 6.11 / 10.78 | 25 | 42 | 76% / 98% | 97% | 20% |
| slots (4.4) | 32 | — | 256 | 1,785 | 5.04 [5.0–5.5] | 2.41 / 4.33 | 32 | 0 | 31% / 52% | 57% | 64% |

² Large spread, not explained yet. An isolated rerun of block 32 gave 9.7–10.5 s at a steady 1,875 MHz SM clock, so the slow runs look like interference from the host. These decode steps are CPU-bound (GPU utilisation 28–51%), so they are sensitive to other load on the machine. Treat those two medians as rough.

Reading:
- **Same memory, 1.46× the throughput.** In the 64 MiB that holds 8 contiguous slots, paging fits **25** sequences at once (block 16). Most requests need far fewer than 512 positions, and a paged sequence holds only the blocks it has filled. The pool goes from 41% to 76% used on average (98% at peak), and claimed KV holding real tokens goes from 52% to 97%. With block 8 the gain is 1.54×.
- **The gather costs ~7%.** At equal concurrency (8 slots, no memory pressure), paged runs at 592 vs 633 tok/s: 11.7 vs 11.0 ms per decode step. That copy is what a PagedAttention kernel removes (PARKING_LOT).
- **Block size.** Smaller blocks waste less of the last block per sequence: 99% / 97% / 94% of claimed KV holds tokens for blocks 8 / 16 / 32. They fit slightly more sequences (26 vs 25) and ran a little faster here. The cost is more blocks to gather and a larger table. vLLM's default is 16. Here 8 and 16 are within noise of each other, and 32 is worse.
- **The budget, not the slot count, is now the limit.** 64 slots run no more sequences than 32 (still 25 active); the extra slots only add empty decode rows (row util 20%), so the run is slower. In the 32-slot rows, empty rows also hold row util at ~40%. Decoding only active rows would recover this (PARKING_LOT).
- **Preemption is the price of admitting optimistically.** 42 recomputes per 64 requests (block 16). Sequences are admitted while they are short, then grow until the pool is full, and the newest are evicted and re-prefilled later. vLLM uses the same recompute policy with a free-block watermark; a watermark or reserve-on-admit would trade some concurrency for fewer recomputes (PARKING_LOT).
- **Versus 4× the memory.** 32 contiguous slots with 256 MiB run at 1,785 tok/s. Paged with a quarter of that memory reaches 52–55% of it, held back by the preemptions and empty rows above, not by memory.
- **Bug found by the benchmark:** a preempted request returns with a longer context (prompt + tokens so far). Batched into one padded prefill with a fresh request that still had many tokens to go, the padded length + remaining tokens passed the 512-position RoPE table. Admission now stops adding requests to a prefill batch once `max(context) + max(remaining) > max_len` (regression test `test_requests_that_only_fit_alone_are_not_batched`).
- **Correctness:**
  - fp32 (CPU tests, `tests/test_paged_kv.py`, 12 tests): allocator alloc/free/reuse, and double free raises. Paged decode logits equal the full forward with a padded joint prefill, a late joiner and block boundaries crossed. `run_continuous(kv="paged")` is identical to solo `generate()` for slots 1/3/10 and blocks 4/8/16, including a tiny budget that forces preemption. No blocks are leaked at the end.
  - bf16 (GPU): 38–49 of 64 requests are identical to solo. On the block-16, 32-slot run (42 preemptions), every divergence is a near-tie (fp32 top-2 gap ≤ 0.076, against a median of 1.84), as in 4.3 and 4.4.

### 4.6 INT8 weight-only quantization (2026-11-04)
`scripts/bench_quant.py`, `checkpoints/main/best.pt`. Raw numbers: `runs/bench/int8.json`. Code: `src/tinyllm/quant.py`.

Method:
- **Scales:** per output channel, symmetric absmax: `s_j = max|W_j| / 127`, `q_j = round(W_j / s_j)` clipped to ±127. Activations stay floating point (W8A16).
- **Forward:** `F.linear(x, q.to(bf16)) * s`. The scale factors out of the matmul exactly. The int8 → bf16 cast is exact (|q| ≤ 127).
- **What is quantized:** all 57 Linears (q/k/v/o and gate/up/down × 8 layers, plus `lm_head`). With tied embeddings, `lm_head` gets its own int8 copy; the embedding table stays float, because a lookup reads only B rows per step.

Three variants of the same weights, all under bf16 autocast:
- **fp32:** fp32-stored weights, which autocast recasts to bf16 at every matmul (the engine so far).
- **bf16:** weights stored in bf16.
- **int8:** as above; embedding and norms stay fp32.

Quality:

| Variant | Val loss (20 × 32 × 512 tokens) | Argmax agreement with plain fp32 | Mean KL vs plain fp32 (nats) | Largest top-1 margin at a flip | Greedy rows identical to fp32+autocast |
|---|---|---|---|---|---|
| fp32 + autocast | 1.2825 | 99.34% | 1.1e-4 | 0.066 | (reference) |
| bf16 weights | 1.2826 | 99.34% | 1.3e-4 | 0.097 | 4 / 32 |
| int8 | 1.2826 (**Δ +0.0001**) | 99.08% | 2.3e-4 | 0.102 | 0 / 32 |

The agreement and KL columns are teacher-forced on 32 sequences (128 val tokens + 256 greedy tokens), against the same model in plain fp32 with no autocast. The median top-1 margin is 1.70 nats.

Speed: KV engine, P=128, 256 new tokens, median of 5 per round, 3 interleaved rounds (median round shown).

| Batch | fp32 + autocast tok/s (ms/step) | bf16 weights | int8 | int8 / fp32 | Rounds fp32 / bf16 / int8 |
|---|---|---|---|---|---|
| 1 | 106 (9.43) | 135 (7.39) | 82 (12.12) | **0.78×** | 103–111 / 126–144 / 82–87 |
| 8 | 865 (9.25) | 1,042 (7.68) | 735 (10.89) | 0.85× | 847–876 / 994–1,060 / 706–735 |
| 32 | 3,275 (9.77) | 4,287 (7.46) | 3,020 (10.60) | 0.92× | 3,169–3,658 / 3,898–4,419 / 2,988–3,040 |

Stored weights: fp32 114 MiB, bf16 57 MiB. INT8 is 28.7 MiB of matmul weights (int8 + fp32 scales) plus the untied 16 MiB fp32 embedding, 44.7 MiB in all (36.7 if the embedding were bf16).

Reading:
- **Quality is free at this size.** The val loss change is +0.0001, the same as merely storing in bf16. INT8 flips about twice as many argmaxes as bf16 (0.92% vs 0.66% of positions), and every flip is a near-tie (margin ≤ 0.10 nats, median 1.70). Per-channel scales are enough; nothing needed per-group scales or outlier handling.
- **Speed: INT8 is slower here, as predicted.**
  - A step takes ~9.5 ms at every batch size. Reading 57 MiB of bf16 weights at 423 GB/s takes ~0.14 ms, about 1.5% of that. So even a perfect INT8 kernel could save at most ~0.07 ms per step on this model.
  - What INT8 adds is ops. An `Int8Linear` is three PyTorch ops (cast, matmul, scale) plus Python dispatch, against two (autocast cast + matmul) for fp32 and one for bf16. That's +2.7 ms per step over 57 layers, likely mostly Python and launch overhead (not profiled).
- **Even when bandwidth-bound, cast-then-matmul loses to bf16.** Per weight per step:
  - fp32 + autocast moves ~8 bytes (read 4 + write 2 for the cast, read 2 for the matmul).
  - int8 + cast moves ~5 bytes (read 1, write 2, read 2).
  - bf16 moves 2 bytes.
  - So INT8 only beats bf16 when the matmul reads int8 directly and dequantizes in registers. That is the 4.10b fused GEMV; the 4.9d rerun on TinyLlama (2.2 GB bf16, truly bandwidth-bound) is the real test.
- **Side finding: storing weights in bf16 is worth 1.20–1.31× today,** at no quality cost. The engine has paid an fp32 → bf16 cast of every weight on every step since 4.1. This was measured for the PARKING_LOT B=1 item. Making it the default changes the engine, so it needs a DECISIONS entry first. The 4.1–4.5 numbers above are all fp32 + autocast.
- **Correctness** (`tests/test_quant.py`, 16 tests, fp32 CPU):
  - Rounding error is ≤ half a step per element, and an all-zero row gives zeros, not NaN.
  - `Int8Linear` equals `F.linear` on the dequantized weight, and every Linear is replaced, for tied and untied embeddings.
  - Matmul bytes are exactly 1 per weight + 4 per row.
  - KV-cached `generate()` equals naive `sample()`, and `run_continuous` (slots and paged) equals solo `generate()`, all on the quantized model.
  - The GPU timed loop equals `generate(temperature=0)` on the int8 model.

### 4.11 CUDA graphs on the decode step (2026-11-04)
- **What:** `src/tinyllm/cuda_graph.py`. Captured by hand with `torch.cuda.CUDAGraph`, with no `torch.compile`. Use it with `generate(..., cuda_graph=True)`, or pass a `GraphDecoder` to reuse its graphs. It is off by default, and the eager path is unchanged.
  - **`StaticKVCache`** implements the same cache protocol, so the model code doesn't change. The position lives in a device tensor. K/V are written with `index_copy_`, and RoPE is gathered at that position. Every step attends over a fixed window of W keys, and a bool mask hides keys beyond the position and left padding.
  - **Buckets:** W = 128, 256, …, `max_seq_len`. Each bucket gets one graph, captured on first use. All graphs share one memory pool and the same static input and cache buffers.
  - **Eager parts:** prefill and sampling. Autocast runs with `cache_enabled=False`, because its weight-cast cache doesn't survive capture.
  - **Scope:** B = 1 and static batches, with or without left padding. The continuous and paged engines are not covered.
- **Correctness:**
  - **Replay equals the same static step run eagerly**, bit for bit (`torch.equal`), across every bucket boundary, on the tiny model and on TinyLlama.
  - **Against today's eager `generate()` it is only bf16-close.** Eager decode attends over exactly pos + 1 keys with no mask, so SDPA picks flash attention. The graph's masked window gets the mem-efficient kernel instead.
    - 30M model, teacher-forced, 8 × 256 tokens: max |Δlogit| 0.19 (logit scale 24) and argmax agreement 99.41%. All 12 flips are near-ties: eager's top-2 margin is ≤ 0.125 (one bf16 ulp at |logit| 16–32), against a median of 1.75.
    - Free-running greedy is identical on 1/8 rows. Four of the divergences fall at exactly the tokens where eager kv already left naive in 4.3 (60, 12, 74, 125). Graph decode matches naive on 3/8 rows, the same as kv did.
    - TinyLlama, 3 chat prompts × 64 tokens: one flip in total ("haiku", first token, eager margin 0.125 against a median of 1.19). The other two prompts are token-identical.
    - So the done-when ("outputs equal eager") holds exactly against the static eager step, but only up to bf16 near-ties against the variable-shape eager path, the same standard 4.3 met against naive.
  - `tests/test_cuda_graph.py` (CUDA only, 9 tests):
    - replay == static eager step
    - greedy == eager `generate` across all buckets, on a tiny model with well-separated logits
    - left-padded batch == `prefill_padded` + eager decode
    - EOS, decoder reuse, sampling with the same generator
    - errors for a full cache, wrong batch and a bucket set too small
    - TinyLlama: flips only at near-ties (≤ 2 ulps), and free-running greedy diverges only at a flip
- **Speed** (`scripts/bench_generate.py --engine kv | kv-graph --dtype bf16`): 30M model, bf16 weights (D-023), 256 new tokens, median of 5 runs. Prompt 512 is skipped because 512 + 256 > `max_seq_len`.

| B | Prompt | eager kv ms/step (tok/s) | kv-graph ms/step (tok/s) | Speedup |
|---|---|---|---|---|
| 1 | 32 | 8.00 (125) | 0.99 (1,014) | 8.1× |
| 1 | 128 | 8.08 (124) | 1.08 (928) | 7.5× |
| 8 | 32 | 8.30 (964) | 1.19 (6,744) | 7.0× |
| 8 | 128 | 8.35 (958) | 1.25 (6,403) | 6.7× |
| 32 | 32 | 8.53 (3,751) | 1.56 (20,477) | 5.5× |
| 32 | 128 | 8.64 (3,702) | 1.75 (18,297) | 4.9× |

  - TTFT is unchanged, since prefill is the same eager code (8–19 ms either way).
  - Peak VRAM is higher with graphs (237–599 MiB against 76–395 MiB). That includes the graph pool, and in this script the decoders kept alive for the other batch sizes, so it is not a clean per-row figure.
- **Profile** (`scripts/profile_decode.py`, bf16, context 128; Chrome traces in `runs/profile/`): one decode step, eager against graph. Wall time is synced. "Host issue" is the time Python takes to issue one step starting from an empty GPU queue. GPU busy and idle come from 20 profiled steps. The SM clock is the nvidia-smi median over 2 s of decoding.

| Model, B | Mode | ms/step | Host issue (ms) | GPU busy (ms) | GPU ops | GPU idle | SM clock | Power |
|---|---|---|---|---|---|---|---|---|
| 30M, 1 | eager | 8.64 | 8.20 | 3.97 | 372 | 75% | 1,410 MHz | 35 W |
| 30M, 1 | graph | 1.25 | 0.35 | 1.09 | 412 | 10% | 1,875 MHz | 69 W |
| 30M, 32 | eager | 9.28 | 8.80 | 6.95 | 404 | 58% | 1,410 MHz | 40 W |
| 30M, 32 | graph | 1.83 | 0.63 | 1.88 | 452 | 11% | 1,875 MHz | 105 W |
| TinyLlama, 1 | eager | 26.0 | 25.9 | 8.03 | 1,068 | 83% | 1,380 MHz | 45 W |
| TinyLlama, 1 | graph | 8.06 | 1.36 | 7.90 | 1,164 | 3% | 1,875 MHz | 125 W |

- **Reading:**
  - **Eager decode was host-bound at every size we run.** Issuing a step costs the CPU 8–9 ms on the 30M model (26 ms for TinyLlama's 22 layers), whatever the batch. The GPU sits idle 58–92% of the time (across runs) and downclocks to ~1,400 MHz (~800 MHz in one run).
    - With the GPU slower, its own kernels stretch too: eager GPU-busy time fluctuates between runs (1.2–4.0 ms at B=1, 2.2–9.5 ms at B=32), against a steady 1.1–1.9 ms for graph replay at 1,875 MHz.
    - So the earlier "~9.5 ms at every batch size" (4.1, 4.6) was launch overhead, as suspected. The 4.12 low-clock reading has the same cause.
  - **Graphs make the 30M model GPU-bound** (idle ~10%). A graph launch still costs ~0.35–1.4 ms of host time on WSL, growing with node count. The next limit is GPU time across ~410–450 small kernels, about 2.5–4 µs each. Only fusion lowers that: `torch.compile` or Triton (4.10).
  - **TinyLlama B=1: 38–48 → 124–133 tok/s over two runs, about 65–69% of the 192 tok/s bandwidth ceiling** (20–25% before). The GPU is now busy 97% of the time. The rest of the gap to the ceiling is kernel efficiency, not overhead, and 4.9d will measure it properly.
  - The graph step runs a few more ops than eager (mask, `index_select`, `index_copy_`), and its attention kernel differs. Both are small next to the launches they remove.
- Not done: graphs for the continuous and paged engines (bucketed by slot count), and `torch.compile(mode="reduce-overhead")` as a comparison. Both are in PARKING_LOT.

## Results: TinyLlama 1.1B (fill in)
| Step | Batch | Prompt | Decode tok/s | % of ceiling | TTFT (ms) | Peak VRAM | Notes |
|---|---|---|---|---|---|---|---|
| HF Transformers `generate` | 1 | 128 | 45 | 23.3% | 24.6 | 2,118 MiB | baseline (bf16, SDPA). 22.3 ms/step at every batch and prompt: host-bound |
| ours naive | 1 | 128 | 33 | 17.4% | 24.6 | 2,160 MiB | full forward per token |
| ours KV cache | 1 | 128 | 45 | 23.4% | 22.8 | 2,162 MiB | eager, host-bound like HF |
| ours KV cache + CUDA graphs (4.11) | 1 | 128 | **128** | **66.8%** | 24.9 | 2,685 MiB | 2.8× HF |
| ours KV cache + CUDA graphs (4.11) | 32 | 128 | 2,083 | 33.9% | 333 | 3,886 MiB | 1.44× HF (1,445). % counts weight reads only |
| ours continuous batching | 16 | mixed | 475 | n/a | 5,290 (mean) | 3,151 MiB | 1.32× static (360 tok/s); 64 requests |
| ours INT8 | 1 | 128 | 37 | 10.1% of 362 | 30.0 | 3,455 MiB | quality Δ: val loss +0.0010 (1.6283 → 1.6293), argmax 98.49% vs bf16. 0.80× bf16 speed. VRAM includes the bf16 copy |
| vLLM | 1 / 16 | 128 | | | | | |
| llama.cpp Q8_0 | 1 | 128 | | | | | |

### 4.9b Loading TinyLlama (2026-11-04)
- Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea` (D-026), a single bf16 `model.safetensors` (2,098 MiB). It stays in the HF cache, never in the repo.
- `tinyllm.hf_llama.load_hf_llama()`:
  - Maps the config (and raises on rope_scaling, biases, a non-SiLU activation, `pretraining_tp`, or a mismatched `head_dim`).
  - Builds the model on the meta device, so no fp32 copy is ever allocated.
  - Copies each weight once, from the memory-mapped file to the GPU (our `safetensors_io`).
  - Runs a strict name and shape check.
- Weight names need no remapping beyond dropping the `model.` prefix, and q/k need no permutation (same `rotate_half` RoPE as HF).
- Measured (RTX 4060 Laptop, WSL2), from `scripts/tinyllama_smoke.py`:
  - Load: 3.8 s from a warm page cache.
  - 1,100,048,384 parameters, which equals `estimate_params`. All bf16.
  - VRAM: 2.05 GiB allocated, 2.05 GiB peak.
  - _Correction (4.9d): the GPU is the RTX 3070 Ti Laptop (8 GB), as in Phase 0, not an RTX 4060. The numbers are unaffected._
- Greedy chat answers are coherent:
  - "What is the capital of France?" → "The capital of France is Paris." followed by EOS.
  - Poem and explanation prompts give fluent text up to the 96-token limit.
- Speed is unoptimised eager decode, about 40 tok/s at B=1. Not a benchmark; 4.11 and 4.9d do that.
- Logits vs HF Transformers: 4.9c.

### 4.9c Logits match HF Transformers (2026-11-04)
- Reference: HF `LlamaForCausalLM` (transformers 5.22.1, dev/test dependency only, D-015). fp32 + eager attention is the ground truth.
- Method: each of 5 prompts (4 chat, 1 plain text) is extended by 64 fp32-greedy tokens, and that fixed sequence is fed to every model. "Decode" rows go one token at a time through each model's KV cache, so they test the cached path, not a full forward. From `scripts/tinyllama_vs_hf.py`, saved as `runs/bench/tinyllama_vs_hf.json` (with provenance).

| Phase | Model vs fp32 | Positions | max abs Δ | mean abs Δ | KL mean (nats) | top-1 |
|---|---|---|---|---|---|---|
| prefill | **ours bf16** | 151 | 0.764 | 0.0315 | 3.3e-4 | 100% |
| prefill | HF bf16 (SDPA) | 151 | 0.764 | 0.0315 | 3.3e-4 | 100% |
| decode (cached) | **ours bf16** | 264 | 0.560 | 0.0207 | 2.6e-4 | 98.86% |
| decode (cached) | HF bf16 (SDPA) | 264 | 0.560 | 0.0207 | 2.6e-4 | 98.86% |

- **Ours in bf16 is bit-identical to HF in bf16** (`torch.equal`) at all 415 positions, prefill and decode. We do the same ops in the same order in the same dtype: RMSNorm in fp32 then cast, `rotate_half` RoPE, SDPA with `enable_gqa`, SwiGLU.
- The 3 top-1 flips against fp32 are the same 3 positions for both. They are near-ties: fp32's top-2 margin there is 0.002 / 0.007 / 0.019 logits, vs a median margin of 1.7. That's why "top-1 ≥ 99%" was the wrong done-when (HF bf16 fails it too). The test now checks "no more flips than HF + 1, each at a margin < 0.1".
- Free-running greedy text: ours == HF bf16 on all 5 prompts. Both leave the fp32 path at the same token (24, 32 and 43 on three prompts; never on the other two).
- Tiny random Llamas on CPU in fp32 (GQA 4/2, θ = 500, tied and untied), against HF eager, in both conversion directions (HF → ours, and ours → HF via `save_as_hf`): max diff ~2e-5 on logits of ~12 (rounding). For comparison, a wrong `rope_theta` gives a max diff of ~14. KV-cached decode matches too, and greedy `generate` is token-identical.
- **Bug found:** transformers 5 writes `rope_parameters: {rope_theta, rope_type}` instead of a top-level `rope_theta`. `config_from_hf` would have silently used θ = 10000. It now reads both and rejects scaled `rope_type`s. TinyLlama's own config is the old layout, so 4.9b was unaffected.
- **Tokenizer:** we encode with TinyLlama's shipped `tokenizer.json` as-is. transformers 5's `AutoTokenizer` also applies the `"legacy": false` flag from `tokenizer_config.json`, which differs in two places:
  - It adds no `▁` after a special token. Ours gives `</s>` `▁` `\n` (the normalizer prepends `▁` to every segment).
  - It absorbs one leading space of the input.

  Its `apply_chat_template` also adds no BOS. The rendered chat string is identical, and so is ordinary text.

  We keep the file's behaviour. The 4.35-era fast tokenizer, used when the model was fine-tuned, most likely read `tokenizer.json` unchanged (`legacy` only affected the slow tokenizer). Not verified against the original training run. The test asserts exactly these differences.

### 4.9d TinyLlama benchmarks against the bandwidth ceiling (2026-11-04)
- Setup:
  - TinyLlama 1.1B, bf16 weights, bf16 autocast, RTX 3070 Ti Laptop (8 GB), WSL2.
  - Protocol as in 4.1: 256 new tokens, 3 warmups, median of 5.
  - Prompts are our validation stories re-tokenized with TinyLlama's tokenizer, joined by `</s>` (`data/validation_tinyllama.bin`, git-ignored, built on first use).
  - Scripts: `bench_generate.py` / `bench_serving.py` / `bench_quant.py --model tinyllama` → `runs/bench/tinyllama/*.json` (with provenance). `--engine hf` times HF `generate` as the baseline (D-015).
- **Ceiling:** 423 GB/s ÷ 2,098 MiB of stored weights = **192 tok/s per sequence**. The % column is B × that.
  - It counts weight reads only, not KV reads, so it understates efficiency at large batch and long context.
  - A step reads only one row of the 125 MiB input embedding table. Excluding the table gives **~204 tok/s**, so the B=1 graph row is ~63% of that stricter ceiling.
  - INT8 rows use their own ceiling: 1,113 MiB stored, so 362 tok/s.

Decode tok/s (% of ceiling):

| Engine | B | P=32 | P=128 | P=512 | ms/step at P=128 |
|---|---|---|---|---|---|
| HF `generate` | 1 | 45 (23.2%) | 45 (23.3%) | 44 (23.1%) | 22.3 |
| ours naive | 1 | 44 (23.0%) | 33 (17.4%) | 16 (8.1%) | 29.9 |
| ours KV, eager | 1 | 45 (23.2%) | 45 (23.4%) | 49 (25.3%) | 22.3 |
| **ours KV + graphs** | 1 | **134 (69.5%)** | **128 (66.8%)** | **115 (59.8%)** | 7.8 |
| HF `generate` | 8 | 361 | 358 | 358 | 22.3 |
| ours KV, eager | 8 | 395 | 391 | 393 | 20.5 |
| **ours KV + graphs** | 8 | **907 (59.0%)** | **851 (55.4%)** | **662 (43.0%)** | 9.4 |
| HF `generate` | 32 | 1,448 | 1,445 | 1,439 | 22.1 |
| ours KV, eager | 32 | 1,350–1,577 | 1,089–1,586 | 721–1,335 | 20.2–29.4 |
| **ours KV + graphs** | 32 | **2,485 (40.4%)** | **2,083 (33.9%)** | **1,254 (20.4%)** | 15.4 |

- TTFT at B=1, P=128: HF 24.6 ms, ours 22.8–24.9 ms. Prefill is the same eager code on every engine of ours. At B=32, P=512: 1.2–1.3 s for all of ours and HF.
- Reading:
  - **Eager decode is host-bound at ~20–22 ms/step**, for us and for HF alike, whatever the batch or prompt. That gives ~23% of the ceiling at B=1. At B=8 our eager loop is 9% faster than HF's.
  - **CUDA graphs reach 60–70% of the ceiling at B=1** (2.8× HF). The rest of the gap is kernel efficiency, not overhead (4.11 profile: GPU idle 3%).
  - **At larger batch and context the graph step slows**, from 7.5 ms (B=1, P=32) to 25.5 ms (B=32, P=512). The extra bytes alone don't explain it: B=32 reads at most ~0.7 GB of KV per step on top of 2.2 GB of weights. The likely rest is attention over the fixed masked window, which uses the mem-efficient kernel (4.11). Not profiled; that's 4.10 territory.
  - Naive drops 44 → 16 tok/s from P=32 to P=512, because it re-runs the whole context every step. With 2,048 positions of context, TinyLlama never slides at these lengths.
- **Eager B=32 rows vary from run to run** (P=128: 1,586, 1,298, 1,130 and 1,089 tok/s on four runs; P=512: 721, 945, 952 and 1,335). The table gives the range, from the first sweep and the B=32 rerun (`kv_b32.json`).
  - At B=32 the eager `KVCache` grows the caching allocator's reserved memory to 4.6–5.7 GiB, against 3.8–4.6 GiB allocated, on a card where Windows already holds some VRAM.
  - **WSL finding:** when reserved memory passes physical VRAM, the WSL driver spills to host memory instead of raising OOM. The first sweep's B=32 P=512 eager row took 10.6 s to its first token and decoded at 721 tok/s for that reason.
  - `bench()` now calls `torch.cuda.empty_cache()` before each row's warmup. After that the same row gives 1.2 s TTFT and 952 tok/s.
  - With nothing left to spill, a real OOM can surface instead. The first kv-graph B=32 rerun crashed with one. Each `GraphDecoder` holds a 2,048-position static cache (1.4 GiB at B=32), and the decoders from the B=2 and B=8 correctness checks were still alive, ~0.45 GiB, the most likely cause. The benchmark now keeps one live decoder at a time. The rerun then completed (the allocator logged one OOM and recovered) and matched the first sweep within 2% (2,485 / 2,083 / 1,254 vs 2,433 / 2,071 / 1,259).
- **Continuous batching** (`bench_serving`, the same 64-request workload shape as 4.4, 9,004 tokens; median of 3):

  | Slots | static tok/s | continuous tok/s | × | latency mean / p95 (s), cont. | row util static → cont. | KV reserved (cont.) | KV holding tokens |
  |---|---|---|---|---|---|---|---|
  | 8 | 198 | 272 | 1.37× | 15.4 / 28.5 | 59% → 86% | 352 MiB | 10% |
  | 16 | 360 | 475 | 1.32× | 9.0 / 16.5 | 55% → 78% | 704 MiB | 9% |
  | 32 | 669 | 738 | 1.10× | 6.1 / 10.5 | 55% → 64% | 1,408 MiB | 8% |

  _Source (added 2026-11-06, 6.8): this table's JSON was later overwritten by a rerun with another mode. The printed output survives in `runs/bench/tinyllama/sweep.log`._

  - Same pattern as on the 30M model (4.4): the gain is largest at few slots and shrinks at 32, where 64 requests are only 2 waves.
  - These run eager (no graphs), so they are host-bound like the eager rows above.
  - Each slot reserves `max_seq_len` = 2,048 positions, so only 8–10% of the reserved KV holds tokens. That is the paged cache's case (4.5), now 4× larger than on the 30M model.
  - Identical to solo `generate()`: 21–32 of 64 requests per row. These are bf16 near-ties, as in 4.4.
- **INT8** (`bench_quant`, P=128; median of 3 rounds):
  - Matmul weights 1,973 → 988 MiB (−50%). Embedding (125 MiB) stays bf16.
  - Quality: val loss 1.6283 → 1.6293 (**Δ +0.0010**), on 80 × 8 × 512 re-tokenized validation tokens. Teacher-forced against bf16 on 8 × 256 tokens: argmax agreement 98.49%, KL 9.7e-4 nats. The largest bf16 margin at a flip is 0.31 (median margin 2.0). Greedy rows identical to bf16: 0/8.
  - Speed, decode tok/s:

    | B | bf16 | int8 | int8 ÷ bf16 |
    |---|---|---|---|
    | 1 | 46 | 37 | 0.80× |
    | 8 | 369 | 275 | 0.74× |
    | 32 | 1,323 | 717 | 0.54× |

  - **INT8 is slower, and more so at larger batch.** `Int8Linear` casts the whole weight to bf16 every call, so each step reads 1 byte, writes 2 and reads 2 again per weight, instead of reading 2. Overhead hides this at B=1; it shows once the step gets closer to bandwidth-bound. Peak VRAM 6.9 GiB at B=32 (both variants resident, plus the cast temporaries). The fix is a fused dequant GEMV (4.10b).
- Correctness checks (not asserted; bf16 near-ties, per the reference-answer rule):
  - Every engine's timed loop equals its `generate()` / `sample()` greedy output (asserted).
  - KV vs naive: 1/8 rows identical. Teacher-forced max |logit diff| 1.53 on a logit scale of 33, argmax agreement 99.17%. That fits 4.9c, where each bf16 path is up to 0.76 off fp32.
  - Graphs vs eager KV: teacher-forced max diff 0.58, argmax 98.97%, 21 flips, all at an eager margin ≤ 0.125 (median 2.0).
  - HF `generate` vs our KV: 2/8 greedy rows identical. Teacher-forced argmax agreement 99.41%, 6 flips, all at our margin ≤ 0.125 (1 bf16 ulp at that scale; median 1.6). 4.9c was bit-identical because both models ran plain bf16 there; here ours runs under autocast.
- Not done: the vLLM / llama.cpp rows (4.8a), CUDA graphs for the serving engines (PARKING_LOT), and the fused INT8 kernel (4.10b). _(INT8 kernel done 2026-11-07, see 4.10b below.)_

### 4.11b CUDA graphs in the continuous engine (2026-11-06)
- **What:** `SlotGraphDecoder` in `src/tinyllm/cuda_graph.py`, enabled with `run_continuous(..., cuda_graph=True)`. You can also pass a decoder to reuse its graphs. Slot cache only; paged KV is not done.
  - The slot step already had fixed shapes except one: the attention window ran to the longest slot (`max(lens) + 1`). `SlotKVCache.window` now fixes it to a bucket (128, 256, … `max_seq_len`), so the step stays the same while slots join and leave.
  - One graph per bucket, all in a shared memory pool, as in 4.11. Admit / release update the device-side lengths and slot state in place, so replays see them without a recapture.
  - A replay runs only the device half of `end_step`, so the decoder then calls `advance_host()` to advance the CPU-side lengths. Prefill and sampling stay eager.
- **Correctness:** **graphs equal eager exactly**, unlike 4.11's `generate()`. The eager slot step already attends with a mask, so both paths use the same SDPA kernel. Tested teacher-forced (tiny model and TinyLlama, 0 flips in 240 TinyLlama slot-steps) and end to end through `run_continuous` (1 / 3 / 8 slots, with and without EOS, joins / leaves and bucket switches mid-run).
- **Speed** (`bench_serving --model tinyllama --cuda-graph`, same 64 requests as above, 9,004 tokens, median of 3, NitroSense Performance mode):

  | Slots | static tok/s | continuous (eager) | continuous + graphs | graphs vs eager | latency mean / p95 (s), graphs | capture |
  |---|---|---|---|---|---|---|
  | 8 | 181 | 245 | **581** | 2.37× | 7.4 / 13.5 | 0.27 s |
  | 32 | 631 | 679 | **956** | 1.41× | 4.8 / 8.2 | 0.25 s |

  - The gain is largest at 8 slots, where eager is the most host-bound. With graphs, GPU utilisation goes from 44–65% to 93–97%.
  - The graph rows throttled on heat (84 °C, SM 1575–1725 MHz vs 1785–1875 for eager), so the speedup is measured at a lower clock.
  - The eager rows here are ~9% below the table above (245 vs 272 at 8 slots, measured 2026-11-04). Eager is host-bound, so CPU speed and load move it. Compare within one run.
  - Same tokens as eager at both sizes: 22/64 and 32/64 requests identical to solo `generate()`, with the same first divergences.

### 4.10b Triton INT8 kernel (2026-11-07)
- **What:** `src/tinyllm/kernels/int8_gemv.py`, one Triton kernel computing `y = scale * (x @ q.T)` straight from the int8 weight. Each program owns a block of output columns and loops over K. It loads int8 tiles, converts them to bf16 in registers (exact, since |q| ≤ 127), runs `tl.dot` on tensor cores with fp32 accumulation, and applies the per-row scale once at the end. The rows (M = batch at decode) are padded to 16 for `tl.dot`; the wasted flops cost nothing next to the weight reads. Block sizes are autotuned per (N, K, M bucket), on warm-up, outside CUDA-graph capture.
  - `Int8Linear(backend="triton")` uses it for M ≤ 64 on CUDA. Prefill (M > 64), CPU and fp32 stay on the PyTorch path, which casts the weight to bf16 on every call (three kernels: cast, matmul, scale).
  - Tests: error against an fp64 reference is at most the PyTorch path's, on every TinyLlama shape, M 1–64; CUDA-graph replay; the fallbacks.
- **Stop rule (PLAN):** int8 must beat bf16 at batch 1 with CUDA graphs on. **Passed** in the first session, 1.45×.
- **Found on the way:** batch-32 int8 was a benchmark artifact, not the kernel. With every variant resident, int8 at batch 32 reserved ~8.7 GB on the 8 GB card and WSL spilled to host memory (TTFT 5.4–6.8 s vs 0.34 s bf16). `bench_quant` now keeps only the timed variant's weights on the GPU (`timed_alone` in the JSON). Batch 32 timed alone: bf16 2,106 / int8 1,080 / int8-triton 2,222 tok/s. The old `tl-int8` v1 number (int8 0.36× bf16 at batch 32) had the same spill.
- **Final session** (`bench_all --only tl-int8-gemv,tl-int8,tl-int8-graph`, commit `2fdd1dc`, NitroSense performance, fans Max, each job started ≤ 50 °C; all three tables in [RESULTS.md](RESULTS.md)):
  - **The matmuls alone** (`bench_int8_gemv`, all 155 Linears of one decode step, CUDA-graph replays over weight copies larger than L2):

    | Rows | bf16 ms | PyTorch int8 ms | Triton ms | Triton vs bf16 |
    |---|---|---|---|---|
    | 1 | 5.67 | 16.66 | 3.04 | 1.87× |
    | 32 | 6.74 | 17.75 | 3.75 | 1.80× |
    | 64 | 6.15 | 17.62 | 6.25 | 0.99× |

    At 1 row, bf16 reads 1,975 MiB at 86% of the measured 423 GB/s; the Triton kernel reads 991 MiB at 81%. The large layers (gate/up, down, lm_head) reach 89–96%. At 64 rows the tile becomes compute-bound and the gain is gone, which is why prefill stays on cuBLAS.
  - **End to end, CUDA graphs** (`bench_quant --engine kv-graph --triton`, decode tok/s, median of 3 interleaved rounds):

    | Batch | bf16 | int8 (PyTorch) | int8-triton | triton vs bf16 |
    |---|---|---|---|---|
    | 1 | 128 | 50 | 186 | **1.45×** |
    | 8 | 839 | 355 | 1,063 | **1.27×** |
    | 32 | 2,053 | 1,064 | 2,175 | **1.06×** |

    Quality is the same for both int8 paths: val loss +0.0010 vs bf16, argmax agreement 98.49%. Heat: ⚠ hardware slowdown flags seen, busy clock median 1,545 MHz, 85 °C. The numbers match the first session (B=1 186, B=8 1,080) and the cooler batch-32-alone run (1.05×) within 3%, so they were kept.
  - **Eager** (`tl-int8`, no graphs, PyTorch path only): int8 is 0.74–0.75× bf16 at every batch size; at batch 32, 1,085 tok/s, up from 533 before the spill fix.
- **Known gap:** int8 TTFT is 10–130 ms above bf16 (batch 1 to 32), because prefill uses the PyTorch cast path. A Triton GEMM for prefill is out of scope (Triton is capped at one day, D-016).
- **Why it gains less at batch 32:** the matmuls are 5.7 of bf16's 7.8 ms step at batch 1 (73%) but 6.7 of 15.6 ms at batch 32 (43%); the rest is attention over the KV cache and the other ops. At batch 1 the step saves 2.4 ms, close to the 2.6 ms the matmul bench predicts. At batch 32 it saves 0.9 ms of a predicted 3.0 ms; that gap is not explained yet.

## Results: Triton kernels (fill in)
| Kernel | Shape | PyTorch time | Triton time | Achieved GB/s | % of peak | End-to-end tok/s Δ |
|---|---|---|---|---|---|---|
| Fused RMSNorm + residual | | | | | | |
| INT8 GEMV (decode) | 155 Linears of one TinyLlama decode step, 1 row | 16.66 ms (cast + matmul); bf16 5.67 ms | 3.04 ms | 342 | 81% of 423 GB/s | 1.45× bf16 at batch 1 (186 vs 128 tok/s, graphs); 3.7× the PyTorch int8 path |
