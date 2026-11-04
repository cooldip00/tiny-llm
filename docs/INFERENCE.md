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
