# Progress

## Current
- **Phase:** 4 — Inference engine (Phase 3 complete)
- **Step:** 6.7 CI (4.1–4.6, 4.12, 4.9a–d and 4.11 done; order per D-024/D-025: 4.9d → 6.7 CI → first README → public = v1-preview around 2026-11-11)
- **Next action:** GitHub Actions workflow running the CPU test suite (GPU tests skip without CUDA) on every push, plus a status badge for the README (D-025).
- _Previous step 4.9d (TinyLlama benchmarks against the bandwidth ceiling): done 2026-11-04, see the log._
- **Later (4.6):** per-channel INT8 weights for the Linear layers. Expect little speed gain on the 30M model (overhead-bound); the real test is TinyLlama in 4.9d. _Done 2026-11-04: as expected, quality Δ ≈ 0 and no speed gain here (0.78× at B=1); rerun on TinyLlama in 4.9d._
- **Blockers:** none
- **Open decisions:** none
- **Order (D-022):** 4.4 + 4.5 moved before 4.6 (4.1–4.3 finished a day early). Dates unchanged. Checkpoint 2026-11-09: if 4.9 hasn't started, cut scope instead of moving v1-preview. **D-024:** 4.9a → 4.9b → 4.9c → 4.11 (CUDA graphs) → 4.9d → README + public.
- **Scope (D-021):** **v1-preview** by about 2026-11-11 (4.1 → 4.2 → 4.3 → 4.6 → 4.9a–d, first README, public repo); **v1** by about 2026-11-18 (4.4 + 4.5 → 4.8, rest of Phase 6). 4.7 and 4.10 only if ahead. Phase 5 and Phase 7 deferred.
- **Daily:** ~30 min explaining the day's work out loud (feeds 6.6).
- **Housekeeping:** push the repo to a public GitHub repo (no remote is set up yet; 20 commits so far). _Done 2026-11-04: pushed to github.com/cooldip00/tiny-llm, private for now; make it public at v1-preview (6.5a)._
- **Housekeeping:** delete the old doc copies in `D:\ResumeProject\tiny-llm` (keep `MOVED.md`). My automatic delete was blocked.
- **Reminder:** the W&B/Forge trial ends around 2026-12-02. Downgrade to Free then.

---

## Log
Newest first. One entry per work block: what was done, numbers, next.

### 2026-11-04 — Step 4.9d TinyLlama benchmarks against the bandwidth ceiling
- `bench_generate.py`, `bench_serving.py`, `bench_quant.py` take `--model tinyllama`, which writes to `runs/bench/tinyllama/*.json`.
  - Prompts: validation stories re-tokenized with TinyLlama's tokenizer (`data/validation_tinyllama.bin`, git-ignored, built on first use).
  - Each row reports the ceiling, the % of it, and peak reserved memory.
  - `--engine hf` times HF `generate` (bf16, SDPA) as the baseline, with a greedy and teacher-forced comparison.
  - `bench_quant` on TinyLlama compares bf16 (the baseline) with int8. An fp32 copy doesn't fit in 8 GB.
- Ceiling 192 tok/s per sequence (423 GB/s ÷ 2,098 MiB). Excluding the 125 MiB input embedding gives ~204.
- Decode at B=1, P=128: HF 45 (23.3%), naive 33, eager KV 45 (23.4%), **KV + CUDA graphs 128 tok/s (66.8%)**. At P=32 graphs reach 134 (69.5%).
  - Eager, ours and HF, is host-bound at ~20–22 ms/step whatever B or P.
  - Graphs: B=8 851 (55%), B=32 2,083 (34%; the % counts weight reads only). They slow at long context: B=32 P=512 is 1,254 tok/s.
- Continuous vs static: 1.37× / 1.32× / 1.10× at 8 / 16 / 32 slots. Slots reserve 2,048 positions each, so only 8–10% of the reserved KV holds tokens.
- INT8: val loss Δ +0.0010, argmax 98.49% vs bf16. Speed 0.80× / 0.74× / 0.54× bf16 at B=1 / 8 / 32: the cast materialises a bf16 weight every call. A fused kernel is 4.10b.
- Near-ties only:
  - HF vs ours: 6 teacher-forced flips, all at margin ≤ 0.125.
  - Graphs vs eager: 21 flips, ≤ 0.125.
  - KV vs naive: max diff 1.53, consistent with 4.9c.
- **WSL finding:**
  - When the caching allocator's reserved memory passes physical VRAM, WSL spills to host memory instead of raising OOM. Eager B=32 P=512 then took 10.6 s to its first token (it should take 1.2 s).
  - `bench()` now calls `empty_cache()` before each row. The kv-graph bench keeps one live `GraphDecoder`; stale ones caused a real OOM.
  - Eager B=32 still varies ±20% run to run (memory-marginal on 8 GB).
- Docs: TinyLlama results table filled (vLLM / llama.cpp rows wait for 4.8a), plus the INFERENCE.md 4.9d section. A correction note in 4.9b: the GPU is the 3070 Ti Laptop, not a 4060.
- Next: 6.7 CI.

### 2026-11-04 — Step 4.11 CUDA graphs on the decode step
- `src/tinyllm/cuda_graph.py`:
  - `StaticKVCache` uses the same cache protocol, so the model code doesn't change.
    - The position is a device tensor. K/V are written with `index_copy_`, and RoPE is read with `index_select` at that position.
    - Every step attends over a fixed window of W keys with a mask, covering left padding too.
  - `GraphDecoder` keeps prefill and sampling eager and captures one graph per length bucket (128, 256, …, max), all in a shared memory pool.
  - Turned on with `generate(cuda_graph=True | decoder)`. Off by default.
- Correctness:
  - Replay is bit-identical to the same static step run eagerly.
  - Against variable-shape eager it is bf16-close. On the 30M model: argmax agreement 99.41%, and all 12 flips at eager margin ≤ 0.125 (median 1.75). On TinyLlama: 1 flip in 3 × 64 tokens, at margin 0.125.
  - The cause is SDPA choosing flash for the no-mask eager step and mem-efficient for the masked window.
  - `tests/test_cuda_graph.py`: 9 tests. Suite: **221 passed**.
- Speed (bench_generate, 30M, bf16, median of 5): **8.00 → 0.99 ms/step at B=1 (8.1×)**, 8.35 → 1.25 at B=8, 8.64 → 1.75 at B=32 (4.9–8.1× overall). TTFT is unchanged.
- Profile (`scripts/profile_decode.py`, traces in `runs/profile/`):
  - Eager was host-bound. Issuing a step took 8–9 ms of CPU whatever the batch. The GPU was idle 58–92% of the time and downclocked to 800–1,410 MHz.
  - Graph replay keeps 1,875 MHz with the GPU idle ~10%. A launch still costs 0.35–1.4 ms of host time on WSL.
  - The next limit is ~430 small kernels per step, which needs fusion (4.10 / torch.compile).
- TinyLlama B=1: 38–48 → **124–133 tok/s, 65–69% of the 192 tok/s ceiling** (GPU idle 3%).
- `bench_generate.py`: `--engine kv-graph` and `--dtype bf16`. Wrote runs/bench/kv_bf16.json and kv-graph_bf16.json.
- Next: 4.9d.

### 2026-11-04 — Step 4.9c logits match HF Transformers
- `transformers` 5.22.1 added to the `dev` group only. `src/tinyllm` never imports it (D-015).
- `src/tinyllm/logit_compare.py`:
  - `ours_stepwise` / `hf_stepwise`: prefill, then one token per step through each model's own KV cache.
  - `logit_stats`: max/mean abs, KL, top-1. `first_divergence`. Reusable for 4.8b.
- `scripts/tinyllama_vs_hf.py` → `runs/bench/tinyllama_vs_hf.json`. 5 prompts × 64 teacher-forced decode tokens, against HF fp32 eager.
  - **Ours bf16 is bit-identical to HF bf16** at all 415 positions (prefill + cached decode).
  - Error vs fp32, the same for both: KL 3.3e-4 (prefill) / 2.6e-4 (decode); max abs 0.76 / 0.56.
  - Top-1 100% / 98.86%. The 3 flips are near-ties (fp32 margin 0.002–0.019 vs a median of 1.7), at the same positions for HF.
  - Greedy text: ours == HF bf16 on 5/5 prompts.
- Done-when adjusted from the plan: "top-1 ≥ 99%" fails for HF's own bf16 too. It's replaced by "no more flips than HF + 1, each at a near-tie (margin < 0.1)", plus KL and max-abs ≤ 1.5× HF's.
- **Bug fixed:** transformers 5 configs put θ in `rope_parameters`, and `config_from_hf` would have silently used 10000. It now reads both layouts and rejects scaled `rope_type`s. TinyLlama's own config is the old layout, so 4.9b's results stand.
- Tokenizer: we follow the shipped `tokenizer.json`. transformers 5 `AutoTokenizer` applies `legacy: false` on top. Two documented differences (an extra `▁` after special tokens, one leading space), plus no BOS in its chat template. The chat string itself is identical. Details in INFERENCE.md 4.9c.
- Tests: `tests/test_hf_reference.py`, 8 tests:
  - 6 CPU tests (random tiny Llamas in fp32 vs HF eager, both conversion directions, cached decode, greedy generate). These run in CI.
  - 2 GPU TinyLlama tests: tokenizer differences, and the bf16-vs-fp32 done-when.
  - `test_hf_llama.py` +2 (`rope_parameters` layout and scaled rope_type).
- Next: 4.11 decode-step overhead.

### 2026-11-04 — Step 4.9b load TinyLlama
- D-026: `TinyLlama-1.1B-Chat-v1.0`, pinned at `fe8a4ea`.
  - Single bf16 file (2,098 MiB), in the HF cache only.
  - Our own safetensors reader/writer; no new dependency.
- `src/tinyllm/safetensors_io.py` (~90 lines):
  - `read_header`; `load_file` memory-maps the file, checks byte counts, and copies each tensor once to its device/dtype.
  - `save_file` sorts names and pads the header to 8 bytes.
- `src/tinyllm/hf_llama.py`:
  - `config_from_hf` raises on features we don't implement.
  - Name mapping is just the `model.` prefix.
  - `load_from_dir` builds on meta, then a strict name/shape check, then `assign=True`, then rebuilds RoPE on the real device. Tied and untied heads both work.
  - `save_as_hf` (for tests and 4.8b), `HFTokenizer` (`tokenizers` + Zephyr chat prompt).
- `scripts/tinyllama_smoke.py`:
  - Load 3.8 s; **1,100,048,384 params = `estimate_params`**; all bf16; **2.05 GiB** VRAM.
  - Greedy chat is coherent ("The capital of France is Paris." + EOS).
  - About 40 tok/s eager at B=1 (not a benchmark).
- Tokenizer chat template checked by hand against our `chat_prompt` (same string under HF's `trim_blocks`). 4.9c checks it token by token.
- Tests: `tests/test_hf_llama.py`, 16 tests:
  - safetensors round trip over all dtypes; a hand-built file; bad byte count
  - TinyLlama config → 1.1B exactly; 6 unsupported-config cases
  - HF-dir round trip (tied and untied) gives identical logits and generate output
  - missing/extra weights raise; bf16 load rebuilds RoPE
  - TinyLlama header == our state dict; tokenizer ids (the last two need the cached download)
- Suite: **202 passed**.
- `DECISIONS.md` had D-023 twice: an accidental duplicate pasted after D-025.
  - The two copies differed only in how they cite the training setup ("D-019's 'Other' line" vs "the training 'Other' choices above"). The first is the precise reference.
  - With the user's OK, I merged them into the first copy and removed the second. This is a one-off exception to the D-021 additions-only rule, and no content was lost.
- Next: 4.9c logits match HF.

### 2026-11-04 — Step 4.9a `gqa` attention
- `src/tinyllm/attention/gqa.py`, registered as `"gqa"`. `k_proj` / `v_proj` map d → n_kv_heads·hd, so every cache stores only the KV heads (the caches and engines already sized by `n_kv_heads`).
  - Head order matches HF's `repeat_kv`: query head i reads KV head i // g. That's what TinyLlama's weights expect.
  - With `n_kv_heads == n_heads` it is exactly `mha` (same names and shapes; an `mha` state dict loads into it).
- `config.py`: default `layer_types` is `"gqa"` when `n_kv_heads < n_heads`, and `get_config(..., n_kv_heads=…)` re-derives it. Existing configs and checkpoints are unchanged.
- **Kernel choice, measured** (torch 2.14, TinyLlama heads 32 / 4, hd 64, bf16): SDPA `enable_gqa=True` takes **0.07–0.74×** the time of `repeat_kv` + SDPA, masked cases included, with the same outputs. Copying K/V every decode step is the expensive part (B=32, 512 positions: 1.09 ms vs 0.08 ms). I had expected masks to force the slow path; they don't. `repeat_kv` stays as the test reference.
- Tests: `tests/test_gqa.py`, 14 tests:
  - matches a per-head reference loop; equals `mha` exactly at equal heads; HF head order
  - causal; chunked KV cache == full pass; bf16 CUDA vs reference
  - cache bytes (1/4 of mha at 2 KV heads); params == `estimate_params`; a training step backprops
  - model level: `generate` == `sample`; slots and paged engines == solo; `quantize_model` works
- Suite: **186 passed**.
- Next: 4.9b load TinyLlama.

### 2026-11-04 — Step 4.12 benchmark provenance
- `src/tinyllm/provenance.py`: `Provenance(**extra).start()` / `.finish()` returns a dict of static info (git commit + dirty, python / torch / CUDA / cuDNN, GPU + driver, the weight dtype), plus `before` / `during` / `after` samples.
  - `during` comes from one `nvidia-smi -lms 500` process beside the run (min / median / max SM + memory clock, temperature, power, utilisation; busy-only SM clock; NVML throttle reasons seen; WSL load).
  - Windows CPU % (via `powershell.exe`, ~2 s) is read only before and after. Missing tools give None, never an error.
- Wired into `bench_generate.py`, `bench_serving.py` (also `gpu_during` per row) and `bench_quant.py`, as a `"provenance"` key in each JSON. All three smoke-tested.
- **Finding:** during B=1 KV decode the GPU runs at **~860 MHz SM / ~27 W** with only `gpu_idle` set, versus ~1,875 MHz boost and ~1,350 MHz under a heavy matmul at the 135 W cap. The step is so launch-bound that the GPU downclocks between kernels. More evidence for 4.11.
- Suite: **172 passed**. `tests/test_provenance.py`: 6 tests (parsing, N/A handling, throttle decoding, summaries incl. busy-only clock, git info, no-tools fallback).
- Next: 4.9a `gqa`.

### 2026-11-04 — Step 4.6 INT8 weight-only quantization
- `src/tinyllm/quant.py`: per-output-channel symmetric absmax INT8 (`quantize_per_channel`, `Int8Linear`, `quantize_model`, `weight_bytes`).
  - Forward: `F.linear(x, q.to(bf16)) * scale`. The scale factors out of the matmul exactly.
  - All 57 Linears are quantized, including `lm_head`. With tied embeddings, `lm_head` gets its own int8 copy and the embedding stays float.
- `scripts/bench_quant.py` compares three variants of `best.pt`: fp32 + autocast (the engine so far), bf16-stored weights, and int8.
  - **Quality:** val loss 1.2825 → 1.2826 (**Δ +0.0001**, same as bf16). Argmax agreement with plain fp32: 99.08%, vs 99.34% for bf16. Every flip is a near-tie (margin ≤ 0.10 nats; median 1.70).
  - **Memory:** matmul weights 57 → 28.7 MiB (−50%).
  - **Speed:** INT8 is **slower**: 0.78× / 0.85× / 0.92× fp32 at B=1 / 8 / 32 (82 vs 106 tok/s at B=1). The cast + matmul + scale is three ops per Linear in PyTorch, and decode here is overhead-bound (weight reads are ~2% of a step). As expected in the plan; the fused GEMV (4.10b) and TinyLlama (4.9d) are where INT8 can pay off.
  - **Side finding:** bf16-stored weights are **1.20–1.31×** fp32 + autocast at the same quality, measured for the PARKING_LOT B=1 item. Not adopted yet; needs a DECISIONS entry.
- Suite: **166 passed** (16 new in `tests/test_quant.py`).
- Also fixed the INFERENCE results table: the 4.4 footnote and the 4.5 row had split it in two.
- Next: 4.9a `gqa`.
- D-023 accepted (after the 4.6 commit): from 4.9 on, inference stores weights in bf16; training is unchanged. Earlier numbers stay labelled fp32 + autocast.
- D-024 accepted:
  - new step 4.11 (CUDA graphs on the decode step, profiled), placed between 4.9c and 4.9d;
  - 4.7 becomes prompt-lookup decoding;
  - 4.8 compares on TinyLlama first, with the 30M export as 4.8b.
  - Dates unchanged. New order: 4.9a → 4.9b → 4.9c → 4.11 → 4.9d → README + public.
- D-025 accepted:
  - 4.12 benchmark provenance (first, before 4.9);
  - 4.13 OpenAI-compatible server + load generator (v1, before 4.8a);
  - 6.7 CI (before going public);
  - 6.8 one-command results tables (v1).
  - Revised timeline table in PLAN.md; dates unchanged.
- Fixed the parking-lot test `test_initial_loss_near_uniform`: seeded, and its target corrected to ln V + σ²/2 = 9.088 (it was ln V; measured 9.088 ± 0.026 over 20 seeds). Suite: 166 passed.

### 2026-11-04 — Step 4.5 paged KV cache
- `src/tinyllm/paged_kv.py`:
  - `BlockAllocator`: free list; double free raises.
  - `PagedKVCache`: per-layer block pools and a shared block table. Blocks are allocated on demand; pad tokens are not stored (the pad count becomes a RoPE offset). Reads gather the blocks into a contiguous view, then call SDPA.
  - Preemption of the newest sequence when blocks run out, with recompute (as in vLLM).
- `run_continuous(kv="paged", block_size, kv_blocks)`. `SlotKVCache` gets the same `can_admit` / `reserve_step` / `allocated_tokens` hooks.
- Benchmark (5 repeats, median), with the same 64 MiB of KV as 8 slots:
  - paged runs **25** sequences at once: **1.46×** the throughput at block 16 (924 vs 633 tok/s) and 1.54× at block 8.
  - Pool used 41% → 76%; claimed KV holding tokens 52% → 97%.
  - Gather cost at equal concurrency: −7%.
  - 42 preemptions; 64 slots don't help (budget-bound).
- Bug found by the benchmark: a preempted request (longer context), padded together with a long-remaining fresh one, overflowed the 512-position RoPE table. Admission now checks the padded batch; regression test added.
- Benchmark variance: absolute tok/s moved 15–20% between sessions, and two paged rows had a large spread (host interference, CPU-bound steps). Now `--repeats` with a median. The 4.4 rerun gives 1.37× at 8 slots and only 1.05× at 32, noted in INFERENCE.md.
- Suite: **150 passed**. PARKING_LOT: paged-attention kernel, active-row compaction, admission watermark, prefix caching.
- Next: 4.6 INT8.

### 2026-11-04 — Step 4.4 static and continuous batching
- D-022 accepted: 4.4 + 4.5 before 4.6, dates unchanged.
- Cache protocol refactor: `begin_step` / `layer(i)` / `end_step`, with the mask logic in `kv_cache.py`. `KVCache(pad_lens=)` adds left padding with a padded causal mask. `SlotKVCache` adds per-slot lengths, `RotaryEmbedding.at(positions)` and scatter writes, with no per-step GPU sync for bookkeeping.
- `src/tinyllm/engine.py`: `run_static` (left-pad, decode to the longest in the batch) and `run_continuous` (admit into free slots with a batched padded prefill, then one decode step over all slots; release on length or EOS). Both report throughput, latency, TTFT, row utilisation and KV reserved vs used.
- `scripts/bench_serving.py`, 64 mixed requests: continuous **1.34×** static at 8 (525 vs 391 tok/s, p95 latency 22.1 → 14.8 s, row util 59% → 86%) and **1.12×** at 32 (1,434 vs 1,281 tok/s; 2 waves only, so the drain tail dominates). Only 31–41% of reserved KV holds real tokens, which is 4.5's target.
- Correctness:
  - fp32: both engines are identical to solo `generate()` (`tests/test_engine.py`, 12 tests).
  - bf16: 39–43 of 64 requests are identical. Every divergence is a near-tie: the top-2 gap is ≤ 0.076, against a median of 1.84.
- Suite: **136 passed**.
- Next: 4.5 paged KV.

### 2026-11-04 — Step 4.3 KV cache
- Added `src/tinyllm/kv_cache.py` (preallocated `[B, n_kv_heads, max_len, head_dim]` per layer, allocated on first use with the layer's dtype), `Transformer.forward(..., cache=)` (RoPE offset = `cache.pos`), MHA cache path with correct masks for prefill / single-token decode / multi-token on a non-empty cache, and `generate.generate()` (prefill once, then one token per step, EOS per row). `bench_generate.py --engine kv`.
- Speedup vs 4.1 naive: B=1 P=128 **0.93×** (97 tok/s); B=8 P=128 1.12×; B=32 P=32 **2.3×** (2,976 tok/s); B=32 P=128 **3.1×** (2,767 tok/s). P=512 rows n/a (a cache can't slide past max_seq_len).
- Every row is now ~10–11.5 ms/step: overhead-bound, as 4.1 predicted.
- Correctness: fp32 greedy identical to naive (tests). bf16: 5/8 rows identical, 3 diverge at tokens 51/78/239; teacher-forced max |logit diff| 0.125 on scale 24.1, argmax agreement 99.76% → bf16 near-ties, within the reference-answer rule.
- Bug found and fixed: a cache ↔ layer reference cycle kept finished caches alive until gc (B=32 peak 2,588 MiB → 466 MiB after the fix). Test added.
- Replaced the placeholder `test_cache_not_implemented_yet` with `test_cached_chunks_match_full_pass`. Suite: **124 passed**.
- Next: 4.6 INT8.

### 2026-11-04 — Step 4.2 sampling (temperature, top-k, top-p)
- Moved the token choice into a pure `generate.sample_next(logits, temperature, top_k, top_p, generator)` (B, V) → (B, 1), so the KV-cache and batching engines (4.3+) reuse it. `sample()` gained `top_p` and calls it; greedy behaviour is unchanged (4.1 gate still OK).
- Order follows HF: temperature → top-k → top-p. Top-p keeps the smallest set of most likely tokens with mass ≥ p, always including the top token. Invalid `top_k < 1` or `top_p` outside (0, 1] raise.
- Tests: new `tests/test_generate.py`, 13 tests (known-distribution nucleus sets and renormalised frequency, threshold crossing, top-k then top-p, temperature before top-p, per-row filtering, tiny top-p = greedy end to end, invalid args). Suite: **112 passed**.
- Found: `test_initial_loss_near_uniform` is unseeded and order-dependent (fails alone). The new test uses `fork_rng` so it doesn't shift the global RNG. Logged in PARKING_LOT, not fixed here.
- Next: 4.3 KV cache.

### 2026-11-04 — Step 4.1 naive generation baseline
- Added `scripts/bench_generate.py`: naive greedy generation (full forward per token, sliding window at 512), INFERENCE.md protocol (prompts 32/128/512, 256 new tokens, batch 1/8/32, 3 warmups, median of 5). Prompts are fixed validation windows. Correctness gate: the timed loop equals `sample(temperature=0)` (OK).
- Key numbers (`checkpoints/main/best.pt`, full grid in docs/INFERENCE.md):
  - B=1 P=128: **104 tok/s decode**, TTFT 9.1 ms, peak 199 MiB.
  - B=8 P=32: 800 tok/s. B=32 P=32: 1,273 tok/s. B=32 P=512: 453 tok/s (70.7 ms/step).
- Finding: batch 1 is overhead-bound (~10 ms/step at any context length), ~1.5% of the ~7,000 tok/s bandwidth ceiling for 60 MB of bf16 weights at 423 GB/s. Expect 4.3's KV cache to help mostly at long context and large batch, not at B=1.
- One transient allocator OOM warning in the B=32 P=128 row; a clean rerun matched (873 vs 891 tok/s).
- Pushed the repo to GitHub (private) before starting.
- Next: 4.2, add top-p to `sample`, with tests.

### 2026-11-04 — Scope decision for v1 (D-021)
- No code changes. Planning only.
- Decided: two milestones following the step dependencies. **v1-preview** (~2026-11-11): 4.1–4.3, 4.6, 4.9a–d, first README, public repo. **v1** (~2026-11-18): 4.4 + 4.5, 4.8, rest of Phase 6. A single one-week deadline for everything was judged too tight.
- 4.4 + 4.5 are done together, and 4.5 (paged KV) is must-do because it links to prior allocator experience. 4.8 (vLLM / llama.cpp comparison) is must-do for credibility.
- Deferred, not dropped: Phase 5 ablation, and new Phase 7 future research (Weight-Space Programming replication, experiments from the research report).
- Docs updated (additions only): DECISIONS D-021; PLAN (`[>]` status key, Phase 4 priority, Phase 5 deferral note, 6.5 publish, 6.6 self-check, Phase 7, revised timeline); PARKING_LOT (D-021 section); RESEARCH ("Future directions"); PRIOR_ART (423 GB/s update, positioning update, PA-2); CLAUDE.md (goal 3 deferral note).
- Next: 4.1. Read docs/INFERENCE.md, then benchmark naive generation with `checkpoints/main/best.pt`.

### 2026-11-03 — Step 3.7 main run · Phase 3 complete
- Run: `--preset main --max_steps 7600 --run_name main-3.7`, D-019 defaults, compiled (D-020). Logged online: [run `ugd124lz`](https://forge.coreweave.com/wandb/cooldip00-personal/tinyllm/runs/ugd124lz). Log in `checkpoints/main/train.log`.
- 996M tokens (~2.1 passes over the 465M-token train set) in **2 h 27 min** (14:40 → 17:07 UTC), including 39 evals and samples.
- Val loss:

  | Step | 0 | 200 | 400 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 7599 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Val | 9.009 | 2.431 | 1.917 | 1.617 | 1.484 | 1.417 | 1.369 | 1.333 | 1.304 | 1.287 | **1.2825** |

  - **Plateaued:** −0.0026 over the last 400 steps, −0.0006 over the last 200, as the LR reached its floor.
  - Final train loss 1.280 ≈ val 1.2825, so there's no overfitting despite ~2 epochs.
  - Main vs smoke at the same step 1999: 1.484 vs 1.542. The bigger model is ahead from early on, and its full run ends 0.26 lower than smoke.
  - This loss isn't directly comparable to llama2.c's numbers, because loss per token depends on the tokenizer (ours is BPE 8192, theirs Llama 32K).
- Throughput: **115.3K tok/s mean** (111.6–128.3K), **56.1% MFU**, ~23.6 TFLOPS over the whole run. Heat cost only ~3% (vs ~8% eager).
- Samples ("Once upon a time"):
  - Step 0: random tokens.
  - By step 1000: coherent multi-paragraph stories with dialogue.
  - Final, step 7599: "…a little girl named Lily. She was very curious… she found a shiny rock that looked like it was magic… she saw a butterfly that was also very shiny… found a beautiful park with a big fountain…" The story is fluent and on-topic, with a few loose details ("rock" later becomes "crystal").
- Checkpoints: `ckpt.pt` 359 MB, `best.pt` 120 MB (git-ignored). `best.pt` is the model for Phase 4.
- **Phase 3 "Done when" met:** val plateaued, coherent stories, final val loss / tok/s / MFU recorded.
- Next: Phase 4.1 naive generation baseline.

### 2026-11-03 — `torch.compile` before the main run (D-020)
- `TrainConfig.compile` (default on, `--no-compile` to disable) compiles only the training step. Eval, samples and checkpoints use the eager model, so there are no recompiles and no `_orig_mod.` keys.
- First attempt failed: there was no C compiler in WSL, and Triton needs one. You installed `build-essential` (gcc 13.3, SETUP Step 16).
- Benchmark (main preset, 120 steps, eager → compiled → eager): **72.5K → 114.1K → 70.3K tok/s**. Compiled is **+57%**, MFU 35% → 55.5%. Startup costs ~20 s more.
- Loss check: the two eager runs with the same seed differ by 0.53 at step 119 (3.82 vs 4.35), because GPU runs aren't bit-reproducible and they pass the early plateau at different times. The compiled run (4.10) falls between them.
- Tests 99/99 pass, including a new CUDA test (compiled ≈ eager losses, plain checkpoint keys).
- Main-run estimate: ~2.7 h for 1B tokens, down from ~4.2 h.
- Next: 3.7 main run.

### 2026-11-03 — Step 3.6 smoke run
- W&B login done by the user. Host `api.forge.coreweave.com`, user `cooldip00` (SETUP Step 15).
- Run: `--preset smoke --run_name smoke-3.6`, default D-019 hyperparameters, 2,000 steps = 262M tokens. Online to W&B: [run `cel67ssj`](https://forge.coreweave.com/wandb/cooldip00-personal/tinyllm/runs/cel67ssj). Log in `checkpoints/smoke/train.log`.
- Wall clock **35 min** (13:40 → 14:15 UTC), including 11 evals and samples.
- Val loss:

  | Step | 0 | 200 | 400 | 800 | 1200 | 1600 | 1999 |
  |---|---|---|---|---|---|---|---|
  | Val | 9.031 | 2.646 | 2.032 | 1.753 | 1.641 | 1.573 | **1.542** |

  It was still falling slowly at the end (−0.012 over the last 200 steps), so the bigger model and more tokens in 3.7 should help.
- Throughput: steady at **124K tok/s** (122.6–136.3K), 28.6% MFU. That's ~8% below the 1-minute 3.5 numbers because of thermals. Main estimate becomes ~67K tok/s sustained.
- Samples ("Once upon a time"):
  - Step 0: random tokens.
  - By step 800: coherent short stories.
  - Final, step 1999: "…she found a shiny coin on the ground… she accidentally dropped the coin in a puddle… a kind lady saw what had happened and helped her… She learned that it's okay to cry, but it's important to be honest and kind." The grammar holds and the story has an arc, with occasional logic slips.
- Checkpoints: `ckpt.pt` 165 MB, `best.pt` 55 MB (git-ignored).
- Next: 3.7 main run.

### 2026-11-03 — Step 3.5 throughput and MFU
- `ModelConfig.flops_per_token(seq_len)` = 6·N + 12·L·d·T, the PaLM convention.
  - N is the matmul weights only (layers + LM head); embedding lookup and norms are excluded.
  - The attention term counts the full T×T matrix, so the numbers compare with published MFU.
  - Values at T = 512: main 204.5M FLOPs/token, smoke 96.7M.
- `train.py` logs `train/tflops` and `train/mfu` (vs the 42.0 TFLOPS bf16 peak, `--peak_tflops`) next to tok/s, in the terminal and to W&B.
- GPU check (60 steps per preset, plugged in; step 0 is slow from warmup and is ignored):

  | Preset | tok/s | TFLOPS | MFU |
  |---|---|---|---|
  | smoke | 134–137K | 13.0–13.3 | **31%** |
  | main | 72–75K | 14.7–15.4 | **35–37%** |

  - Throughput drifts down ~4% over a minute as the GPU heats up.
  - The earlier "~32%" for main used 6·N only. With the attention term it's ~36%.
  - Main at ~72K tok/s: 1B tokens ≈ 3.9 h.
- Tests (98/98 pass): hand-computed FLOPs/token for both presets; the attention term scales with T; GQA lowers it; logged MFU = tok/s · FLOPs/token / peak.
- Parked: `torch.compile` for more throughput (PARKING_LOT).
- Next: 3.6 smoke run, which needs `wandb login` first.

### 2026-11-03 — Step 3.4 W&B logging
- `uv add wandb` → wandb 0.31.0.
- `src/tinyllm/logger.py`: one small interface (`log` / `log_sample` / `finish`) per D-013.
  - `Logger` is the no-op (`--logger none`).
  - `WandbLogger` takes `--wandb_mode online|offline|disabled`. wandb is imported lazily, so the no-op path never needs it.
- `train.py` logs:
  - `train/loss`, `train/lr`, `train/grad_norm`, `train/tok_per_s`, `train/tokens` every `log_every` steps.
  - `val/loss` and `val/best` at each eval, plus a `samples` table (step, text).
  - Only scalars and text go to W&B, never checkpoints (Free plan is 5 GB/month).
- Resume: the W&B run ID is saved in `ckpt.pt`, so `--resume` continues the same run and the charts stay continuous. The logger is closed in a `finally` block, so a crash still closes the run.
- New CLI flags: `--logger`, `--wandb_project` (default `tinyllm`), `--wandb_mode`, `--run_name`.
- GPU check (smoke, offline, 30 steps, stopped at 20 and resumed): both parts used the same run ID `evdo3w4w`, and `ckpt.pt` stores it. Val 8.58 → 5.73 → 5.56 → (resume) **5.34**.
- Tests (97/97 pass), using a recording logger:
  - Exact steps for train and val metrics, the metric key set, `train/tokens`.
  - The run ID lands in the checkpoint.
  - `finish()` runs even when training crashes.
  - W&B `disabled` mode works with no network; an unknown backend raises.
- Next: 3.5 MFU logging.

### 2026-11-03 — Step 3.3 eval, checkpoints, samples
- `train.py` additions:
  - `evaluate()`: mean val loss over 20 fixed batches (327,680 tokens). The seed is fixed, so every eval uses the same windows and losses are comparable.
  - `save_checkpoint` / `load_checkpoint`: temp file then rename (crash-safe), loaded with `weights_only=True`.
    - `ckpt.pt` holds model + optimizer + rng state + step + best val + both configs, for resuming.
    - `best.pt` holds weights + model config only, for sampling and export.
  - Every `eval_every` steps (default 200) and at the last step: val loss, then save `ckpt.pt` (and `best.pt` if val improved), then a sample from the prompt "Once upon a time".
  - `--resume` continues from `<out_dir>/ckpt.pt`. `--stop_at N` stops early while keeping the full-length LR schedule, so the ~3.7 h main run can be split across sessions. A crash loses at most `eval_every` steps (~6 min for main).
  - Default run dir: `checkpoints/<preset>` (git-ignored).
- `src/tinyllm/generate.py`: `sample()` with temperature (0 = greedy), top-k, EOS stop and sliding context, no KV cache. It's only for training samples; the real engine comes in Phase 4.
- GPU check (smoke, 60 steps, stopped at 40 and resumed): val 8.64 → 4.67 (step 30) → 4.38 (step 39) → **4.13** (step 59).
  - The samples move from random tokens to TinyStories-like text ("Then, Lily was a good to, the best for…").
  - Checkpoint sizes: smoke `ckpt.pt` 165 MB, `best.pt` 55 MB. Main will be ~360 MB / ~120 MB.
- Tests (93/93 pass):
  - **Resume is bit-identical:** 20 straight steps equal 10 + stop + resume + 10 (CPU, exact equality).
  - Eval schedule; val loss drops; checkpoint contents; `best.pt` loads into a fresh model.
  - Sampling: greedy is deterministic, top-k=1 equals greedy, stops at EOS, slides past max_seq_len, restores train mode.

### 2026-11-03 — Step 3.2 training loop
- `src/tinyllm/train.py`:
  - `TrainConfig` dataclass; every field is also a CLI flag (`python -m tinyllm.train --preset main --max_steps 100`).
  - `lr_at`: linear warmup, then cosine to `min_lr`.
  - `make_optimizer`: fused AdamW, with decay on 2-D weights only.
  - `train_step`: gradient accumulation (loss ÷ n micro-batches), bf16 autocast on CUDA, clip at 1.0, and returns device tensors so there's no GPU sync per step.
  - `train`: the step loop, logging loss / lr / grad norm / tok/s.
- Hyperparameters recorded as **D-019**: 131,072 tokens/step (32×512 × 8), peak LR 1e-3, warmup 200, cosine to 1e-4, wd 0.1, betas (0.9, 0.95).
  - My first default had 4 accumulation steps, which is only 65,536 tokens/step. The test caught it, and it's fixed to 8.
- Short GPU runs (warmup 10):

| Preset | Steps | Loss | Throughput | ≈ TFLOPS (6·N·tok/s) | 1B tokens would take |
|---|---|---|---|---|---|
| smoke | 40 | 9.07 → 4.59 | ~131K tok/s | 10.8 | ~2.1 h |
| main | 30 | 9.09 → 5.60 | ~74K tok/s | 13.3 (~32% of peak) | **~3.7 h** |

- `tests/test_train.py`:
  - LR schedule values and monotonic decay; optimizer groups (norms not decayed, every param exactly once).
  - **Grad accumulation 2 × B equals one 2B batch** (SGD, params equal to 1e-6).
  - Clipping: the reported norm is pre-clip, and the step size equals the clip value.
  - The CPU loop learns a periodic corpus; seq_len above max raises; CUDA fused + bf16 path.
- 87/87 tests pass.

### 2026-11-03 — Step 3.1 data loader
- `tinyllm.data.get_batch(tokens, batch_size, seq_len, rng, device)`:
  - Returns random windows `x = tokens[s:s+T]`, `y = tokens[s+1:s+T+1]` as int64. Sampling is with replacement, and windows cross story boundaries (EOS marks where a story ends).
  - One vectorised fancy-index gather from the memmap. On CUDA the batch goes through pinned memory with `non_blocking=True`.
  - The seeded `np.random.Generator` makes runs reproducible. Saving `rng.bit_generator.state` in a checkpoint makes resume exact (tested).
- Throughput, train memmap to GPU: 16×512 at 6.0 ms (1.4M tok/s); 32×512 at 4.6 ms (**3.6M tok/s**).
  - Training is expected at ~70K tok/s, so the loader costs ~2% of step time and needs no prefetch thread.
- `tests/test_batch.py`: shapes/dtype, contiguous windows with y shifted by one, same seed gives the same batch, rng-state resume, both ends of the array reachable with no out-of-range starts, too-short input raises, CUDA device, real `train.bin`.
- 79/79 tests pass.

### 2026-11-03 — Step 2.6 one-batch overfit done, so Phase 2 is complete
- `scripts/overfit_batch.py`: one fixed batch of 8 × 256 random windows from `train.bin`, AdamW lr 1e-3 (no weight decay), bf16 autocast, 300 steps. Exits non-zero if the final loss is ≥ 0.05.

| Preset | Params | Start loss | Final loss | < 1.0 at step | < 0.05 at step | Time | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| smoke | 13.8M | 9.067 | **0.0033** | 118 | 153 | 9.7 s | 727 MiB |
| main | 29.9M | ~9.0 | **0.0034** | 184 | 209 | 12.3 s | 1213 MiB |

- Start loss ≈ ln(8192) = 9.011, as expected from the init.
- Phase 2 "Done when" met: all 72 tests pass and the overfit succeeds.

### 2026-11-03 — Step 2.5 whole-model tests
- Added to `tests/test_model.py`:
  - Shapes at T = 1, 7 and 512 (max).
  - Causality through all 6 layers: changing the token at position 20 leaves logits 0–19 identical.
  - `num_params()` == `estimate_params()` for smoke, main, untied, and 3 layers. Exact sizes: smoke 13,767,552, main 29,893,120.
  - Initial loss within 0.1 of ln(8192) = 9.01, so the init is near-uniform as intended.
  - Every parameter gets a non-zero gradient (nothing detached or unused).
  - bf16 autocast on CUDA: bf16 logits, fp32 finite loss ≈ 9.01.
- 72/72 tests pass.

### 2026-11-03 — Step 2.4 block + full model
- `src/tinyllm/model.py`:
  - `Block`: pre-norm, `x + self_attn(input_layernorm(x))`, then `x + mlp(post_attention_layernorm(x))`. The attention class comes from the registry via `cfg.layer_types`.
  - `Transformer`: `embed_tokens` → layers → `norm` → `lm_head`, with `lm_head` tied to `embed_tokens` by default (saves V·d = 4.2M params in main).
    - `forward(idx, targets=None)` returns `(logits, loss)`; the loss is cross-entropy computed in fp32.
    - `num_params()` counts the tied matrix once.
  - Init: Linear/Embedding weights N(0, 0.02); `o_proj` and `down_proj` N(0, 0.02/√(2·n_layers)) (smoke: 0.0058); RMSNorm weights 1.
- The attention import inside `Block.__init__` is lazy because `attention/mha.py` imports `apply_rope` from `model.py` (avoids a circular import).
- `tests/test_model.py`: block and model shapes, tied/untied head, HF Llama names, RoPE tables not in the state_dict, init std per weight group, too-long input rejected.
- 60/60 tests pass.

### 2026-11-03 — Step 2.3 attention interface + `mha`
- `src/tinyllm/attention/base.py`: `Attention` contract (`forward(x, rope, cache=None)`, `cache_bytes_per_token()`), `ATTENTION` registry, `@register` decorator (duplicates raise), `build_attention(cfg, layer_idx)` (unknown names raise `KeyError` listing the registered ones).
- `src/tinyllm/attention/mha.py`: `MHA` with HF names `q/k/v/o_proj`, no biases, RoPE on q and k, `F.scaled_dot_product_attention(is_causal=True)`.
  - Rejects `n_kv_heads != n_heads` (that is `gqa`, Step 4.9a).
  - `cache` raises `NotImplementedError` until the KV cache in Phase 4.
  - Cache cost: main = 2048 bytes per token per layer (bf16), 16 KB per token for 8 layers.
- `tests/test_attention.py`: matches a naive explicit-mask softmax oracle (atol 1e-5); changing tokens 10–15 leaves outputs 0–9 unchanged; registry, HF names, GQA rejection; bf16 autocast on CUDA within 3e-2 of fp32.
- 52/52 tests pass (CUDA test ran, nothing skipped).

### 2026-11-03 — Step 2.2 components
- `src/tinyllm/model.py`:
  - `RMSNorm`: statistic computed in fp32, output in the input dtype.
  - `RotaryEmbedding` + `apply_rope`: rotate_half (HF Llama) layout, precomputed non-persistent cos/sin, `offset` for KV-cached decode.
  - `SwiGLU`: HF names `gate_proj` / `up_proj` / `down_proj`, no biases.
- `tests/test_components.py`, checked against independent references:
  - RMSNorm matches `F.rms_norm`.
  - RoPE matches a complex-number reference, is identity at position 0, preserves norms, and gives the same q·k score for the same relative distance (pos 5/2 vs 105/102).
  - SwiGLU: formula, param count 3·d·ffn, HF names.
- 41/41 tests pass.

### 2026-11-03 — Step 2.1 ModelConfig
- `src/tinyllm/config.py`: frozen `ModelConfig` dataclass.
  - Fields: sizes, `n_kv_heads` (None = MHA), per-layer `layer_types` (None = all `mha`), norm eps, RoPE theta, init std, tied embeddings.
  - Validation: heads divide d_model; KV heads divide heads; even head_dim for RoPE; layer_types length matches n_layers.
  - Also: `estimate_params()`, dict round trip for checkpoints, and `get_config(name, **overrides)`.
- Exact param counts: smoke **13,767,552**, main **29,893,120**. They match the ARCHITECTURE.md estimates.
- `tests/test_config.py`: 29/29 tests pass in total.

### 2026-11-03 — Steps 1.3 + 1.4 done, so Phase 1 is complete
- `tinyllm.data.write_tokens` / `load_tokens`: stream-encode, story + EOS, uint16 written via `.tmp` then renamed; memmap loader.
- `scripts/tokenize_data.py` (wall 879 s, peak RAM 2.7 GB):

| Split | Stories | Tokens (incl. EOS) | File | Avg / median / p99 / max tokens per story | Stories > 512 ctx |
|---|---|---|---|---|---|
| train | 2,118,859 | **465,373,792** | 930.7 MB | 219 / 187 / 641 / 1576 | 3.05% |
| val | 21,981 | **4,677,682** | 9.4 MB | 212 / 185 / 622 / 1079 | 2.60% |

- Checks passed: one EOS per story, the file ends with EOS, max id < vocab, and the first, middle and last story decode exactly from the memmap.
- Tests: 17/17 pass (adds `tests/test_tokens.py`).
- Implications:
  - The 1B-token main run is about **2.15 epochs** of train.
  - Ctx 512 covers about 97% of stories whole. Training samples random 512-token windows across story boundaries (EOS marks the boundary).
- Possible speed-up (not needed now): encoding was single-core, ~15 min. Parallelize only if we ever re-tokenize often.

### 2026-11-03 — Step 1.2 tokenizer trained
- `src/tinyllm/tokenizer.py`: byte-level BPE with a GPT-2-style pre-tokenizer, `add_prefix_space=False` (exact roundtrip), `min_frequency=2`, all 256 bytes in the base alphabet, `<|endoftext|>` = id 0.
- `scripts/train_tokenizer.py` trained on 2,118,859 cleaned train stories in **443 s**, peak RAM **2.7 GB**.
- Saved to `tokenizer/tinystories-bpe8192.json` (554 KB, committed).
- Validation: **4,655,701 tokens**, **4.12 chars/token**, **212 tokens/story**, 0 roundtrip mismatches.
- Train-token estimate: 1.898B chars ÷ 4.12 ≈ **461M tokens**. The exact count comes in 1.3.
- The last merges learned are whole words (`Ġscooters`, `Ġknives`), so the vocabulary is saturated with real words, not junk.
- `tests/test_tokenizer.py`: vocab, EOS id, uint16 fit, exact roundtrip (including non-ASCII via bytes), common words are single tokens. 16/16 tests pass in total.

### 2026-11-03 — Step 1.1b text cleanup (D-018)
- `scripts/inspect_chars.py` showed that the non-ASCII text is mojibake. Of 130,115 non-ASCII train stories, 129,940 are affected. Top sequences: `â€œ` 270,583; `â€™` 195,201; bare `â€` ~270K (lost closing quote); dashes 15K.
- Added `src/tinyllm/data.py` with `clean_text` and `load_clean`, and made `tinyllm` an installed package (hatchling build backend).
- `tests/test_data.py`: 8 passed.
- `scripts/check_clean.py`: train 2,118,859 kept (860 dropped), val 21,981 kept (9 dropped), 0 non-ASCII left, 1,898,184,006 train characters.

### 2026-11-03 — Phase 1.1 done
- Added `datasets` 5.1.0 and `tokenizers` 0.23.3.
- `scripts/download_data.py` downloaded `roneneldan/TinyStories` into the HF cache (2.8 GB on disk).
- Train: 2,119,719 stories, 1.90B characters, average 896 characters, 230 empty stories.
- Val: 21,990 stories, 19.2M characters, none empty.
- 6.1% of train stories contain non-ASCII characters. Check what they are before training the tokenizer.
- Rough estimate: ~4 characters per token gives ~470M train tokens, so the planned 1B-token main run is about 2 epochs.

### 2026-11-03 — Phase 0.5 done, so Phase 0 is complete
- Added `scripts/check_env.py`. Ran it plugged in, in performance mode.
- bf16 GEMM n=8192: **39.7 TFLOPS** (94% of the 42.0 theoretical peak). Sustained 30 s: **38.3 TFLOPS**, 70 → 81 °C, no real throttling.
- Bandwidth: copy 410 GB/s, read 423 GB/s (rated 448).
- Training estimate revised to ~12–19 TFLOPS effective, so the main run takes ~2.6–4 h (was 4–5 h).
- **Next:** Phase 1.

### 2026-11-03 — Phase 0.3, 0.6, 0.7 done; move to `~/llm`
- Docs copied to `~/llm` and checked identical with `diff -r`. D-014 accepted.
- `git init -b main`, `.gitignore` added, local author set. Nothing committed yet.
- uv 0.13.7, Python 3.12.16, `.python-version` pinned to 3.12.
- `pyproject.toml`: torch 2.14.1 from the cu130 index (D-017), numpy, pytest (dev).
- 0.4 verified: `torch 2.14.1+cu130`, CUDA 13.0, `cuda.is_available()=True`, RTX 3070 Ti Laptop, compute capability (8, 6), bf16 supported. Triton 3.8.0 came with torch.

### 2026-11-03 — Phase 0.1–0.2 done
- Free space: C: 38.3 GB, D: 167 GB, so Ubuntu was installed on D:.
- WSL 2.6.1.0. Ubuntu 24.04.5 LTS, user `kuladip`, disk at `D:\WSL\Ubuntu-24.04`.
- `nvidia-smi` in WSL: RTX 3070 Ti Laptop, 8192 MiB, driver 610.91, CUDA 13.3 supported, 140 W power cap.
- RAM visible to WSL: 7.6 GiB (WSL's default of 50% of host RAM).

### 2026-11-03 — W&B account
- Checked W&B pricing on the official page (now CoreWeave Forge): Free = $0, 5 GB/month, public projects allowed.
- Created the account (`cooldip00`, org `cooldip00-personal`). D-013 accepted.

### 2026-11-03 — Plan update: Track B
- Reviewed a LinkedIn post about a C++/CUDA inference engine for TinyLlama 1.1B; adopted its method, not its scope.
- Added D-015 (TinyLlama 1.1B inference track) and D-016 (timeboxed Triton kernels).
- PLAN: new steps 4.9 and 4.10; `gqa` pulled forward into 4.9a; timeline extended.
- INFERENCE.md: bandwidth-ceiling method, reference-answer rule, TinyLlama and Triton results tables.

### 2026-11-03 — Planning
- Discussed feasibility, the inference-at-scale concepts and the research options.
- Chose the framework stack (DECISIONS.md D-001 to D-012).
- Host environment check:
  - GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB, driver 610.91
  - Python 3.12.6 (Windows), nvcc 12.4
  - WSL2: only the `docker-desktop` distro exists, so Ubuntu still needs installing
- Created the project docs.
- **Next:** Phase 0.1.
