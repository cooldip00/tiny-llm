# Parking Lot

Good ideas that are **out of scope for the current phase**. Add them here instead of acting on them.
Review at Phase 6.4.

Format: `- [date] idea — why it's interesting — which phase it might fit`

## Seeded
- [2026-11-03] GPT-2-small scale (124M) on FineWeb-Edu — real web-text model — post-Phase 6, ~2 days of compute
- [2026-11-03] Muon optimizer vs AdamW — big training-speed gains in modded-nanogpt — Phase 5 extension
- [2026-11-03] Small Mixture-of-Experts MLP — the efficiency lever behind frontier models — Phase 5 extension
- [2026-11-03] BitNet b1.58 (ternary weights) — extreme quantization trained in from the start
- [2026-11-03] Multi-token prediction — better sample efficiency, and doubles as a speculative-decoding head
- [2026-11-03] Mamba-2 / RWKV-7 as full replacements, not hybrids
- [2026-11-03] nGPT (normalized transformer), QK-norm — training stability and speed
- [2026-11-03] Mixture-of-depths / early exit — adaptive compute per token
- [2026-11-03] Supervised fine-tuning into a chat format, then DPO
- ~~[2026-11-03] Custom Triton kernel (e.g. fused RMSNorm or RoPE)~~ → moved into PLAN 4.10 (D-016). More kernels beyond those two stay parked.
- [2026-11-03] Inference engine in C++/Rust — llama.cpp-style learning

## Added during work
- ~~[2026-11-03] `torch.compile` on the training step — likely +20–40% tok/s by fusing RMSNorm/RoPE/SwiGLU element-wise ops (MFU is 31–36% today) — before the 3.7 main run if time matters, needs a DECISIONS entry~~ → done before 3.7: **+57%**, on by default (D-020).
- ~~[2026-11-04] `tests/test_model.py::test_initial_loss_near_uniform` is unseeded, so it depends on global RNG order: it fails when run alone (9.11 vs 9.01 ± 0.1) and passes in the full suite. Seed it, or loosen the tolerance with a reason. Found in 4.2.~~ → fixed 2026-11-04. It is seeded now, and the target was wrong: the initial loss is ln V + σ²/2 = 9.088 (σ = init_std·√d), not ln V. Measured over 20 seeds: 9.088 ± 0.026.
- [2026-11-04] Batch-1 decode is overhead-bound (~10 ms/step, ~1.5% of the bandwidth ceiling, from 4.1). CUDA graphs or `torch.compile(mode="reduce-overhead")` on the decode step, plus keeping bf16 weights instead of autocast re-casting each step, could be the bigger B=1 win than the KV cache. Measure after 4.3; needs a DECISIONS entry.
  - _Measured in 4.6 (`scripts/bench_quant.py`):_ bf16-stored weights alone are 1.28× / 1.20× / 1.31× fp32 + autocast at B=1 / 8 / 32, with val loss Δ +0.0001. CUDA graphs / compile are still unmeasured.
- [2026-11-04] Continuous batching admits with a blocking prefill: running slots stall while newcomers prefill (a continuous step is ~13.2 ms vs ~12.2 ms for a static one, in 4.4). Chunked prefill (split a prompt into chunks interleaved with decode steps, as in vLLM / Sarathi) would smooth this out. An arrival-process workload (Poisson arrivals) would also make the 4.4 latency numbers more realistic than "all queued at t=0".
- [2026-11-04] From 4.5 (paged KV):
  - (a) A paged-attention kernel (Triton) that reads K/V blocks in place through the block table, instead of gathering them into a contiguous copy every step. This would also fit Phase 7.
  - (b) Decode only the active rows. Today every slot runs, even empty ones, so 64 slots with ≤25 active wastes ~60% of the rows.
  - (c) An admission watermark, or a reserve-on-admit policy, to cut preemption thrash: 42 recomputes per 64 requests in a 64 MiB budget.
  - (d) Prefix caching: share the blocks of a common prompt prefix through refcounts (the 4.5 stretch goal).
    - _2026-11-06 (4.8a): vLLM and llama.cpp both have it, we don't. Two options, pros / cons and estimates in docs/SERVING.md, "Prefix caching". A, reuse within a slot (llama.cpp style, works with graphs), ~3–4 h — v1.1. B, shared blocks with refcounts and eviction (vLLM style), ~6–8 h plus graphs for paged KV — Phase 7. Either needs prefill starting at a per-row offset, output still equal to `generate()`, and a per-user cache salt (timing side channel)._

## Deferred by D-021 (2026-11-04) — future research, after v1
These map to PLAN Phase 7. Don't start any of them before v1 ships.

- [2026-11-04] **Weight-Space Programming replication** — MIT CSAIL (Takehi, Teoh, Zhang, Khattab), blog post 2026-11-02: GPT-6 Astra, given a Python kernel over Qwen3.5-0.8B's weights and a budget of forward passes, traced a reasoning-loop direction in the layer 15 and 19 MLPs and subtracted it from 131,072 weights (0.016%). No gradients. MATH-500 7.9 → 48.1 at 50 forward passes. The open question: does a non-GPT editor (e.g. Claude) manage the same? Also relevant to continual learning (tiny edits, no retraining) — PLAN 7.2. Details and caveats in PRIOR_ART PA-2.
  - Blog: https://rikiyat.github.io/blog/weight-space-programming/
  - Logs: https://github.com/RikiyaT/lm_understand-logs
  - Edited model: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
  - 8 GB budget: a full replication (512 problems × 32K thinking tokens per forward pass) is too slow here. Scale down (~100 problems, 4K-token cap) and say so in any writeup.

- [2026-11-04] **Experiments from the "machines that learn, not imitate" research report** — PLAN 7.3. Report: `D:\ResumeProject\research-learning-not-imitation\reports\Machines that learn not imitate.md`; notes in `research_notes\`. Each was sized for one 8 GB GPU. Check prior work before starting any of them: some novelty claims and arXiv links in the report were not verified.
  1. Recall probes added to the Phase 5 ablation — **known flaw:** we train at max_seq_len 512, so a 2K–8K needle test measures length extrapolation, not recall. Test within the trained length, or train longer.
  2. From-scratch RLVR vs SFT on arithmetic — **possible prior work** (unverified): "Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining" (Zhao et al., 2025). Frame as a replication if it overlaps.
  3. Grokking prediction sweep — **possible prior work** (unverified): "Predicting Grokking Long Before It Happens" (Notsawo et al., 2023).
  4. Othello world-model emergence (probe for a board-state representation) — runtime is ~2.9 h for 1.2B tokens, not the ~2 h the report says.
  5. Loss of plasticity in a tiny GPT (continual-backprop style)
  6. Mini sparse memory finetuning (continual learning without forgetting)
  7. TRM/HRM-style Sudoku: recursion vs depth
  8. BabyLM Strict-Small (learning from a child-sized ~10M-word budget)

- [2026-11-04] **CUDA graphs for the continuous / paged engines** (4.11 stretch). _Continuous engine done in 4.11b (2026-11-06); the paged engine is still open, after v1._ Bucket by slot count and by the longest slot's length. `SlotKVCache` already keeps its lengths on the device; `begin_step` would need a fixed `_end` (the bucket) instead of `max(self.lens) + 1`. Paged attention needs a fixed block-table shape too.
  - _Slot cache done in 4.11b (2026-11-06): 2.37× at 8 slots on TinyLlama. Paged KV still open._
- [2026-11-04] **`torch.compile(mode="reduce-overhead")` vs manual graphs** (4.11 comparison). After 4.11 the 30M step is GPU-bound across ~430 small kernels (2.5–4 µs each). Fusion (compile, or Triton in 4.10) is the next lever. Also check: graph replay costs 0.35–1.4 ms of host time on WSL, growing with node count. Is that WSL-specific?
- [2026-11-04] **Slot KV caches sized by `max_seq_len`** (4.9d). On TinyLlama each slot reserves 2,048 positions, so only 8–10% of the continuous engine's KV holds tokens (1.4 GiB at 32 slots). Cap slots at the workload's max prompt + new tokens, or use the paged cache by default for TinyLlama.
- [2026-11-04] **Eager `KVCache` memory at B=32 on TinyLlama** (4.9d). Reserved memory grows to 4.6–5.7 GiB against 3.8–4.6 GiB allocated, and decode varies ±20% run to run. Check fragmentation (`expandable_segments`), or allocate the cache once per batch shape.
- [2026-11-05] **README polish for v1** (6.1 / 6.5a; the first README at v1-preview only has the text, the results table and two Mermaid diagrams). Add:
  1. Plots in `docs/img/`, generated from `runs/bench/*.json` by 6.8's `bench_all.py`:
     - tok/s against the bandwidth ceiling by engine (B=1)
     - throughput vs batch size
     - the training val-loss curve
     - _Done 2026-11-06 in PLAN 6.2 (D-030), plus a serving plot and a decode-step profile in RESULTS.md._
  2. A short demo GIF of streaming generation (TinyLlama, KV + graphs).
  3. A link to the HF Space once 6.5b is up.
  4. A link to the blog post (6.5d).
  5. A GitHub social preview image (Settings → Social preview).
  6. A "What I learned / design decisions" section linking the key D-xxx entries.
  7. Plain-language explainers for non-specialists: what a KV cache is and why decode is memory-bound.
- [2026-11-06] **Animated paged-KV explainer** (from 6.2). An animation (SVG/JS or a short video) of requests arriving, taking KV blocks from the pool through their block tables, finishing and freeing them, next to contiguous slots wasting their reserved tail. It needs JavaScript or video, so it can't live in the README (D-030); it belongs in the blog post or on the Space — 6.5d / v1.1.
- [2026-11-06] **Docker Space** — run the HF Space as a Docker Space with the same image as the local server, so "it works on my machine" and "it works on the Space" are one claim. Also gives a one-line `docker run` in the README — 6.5b.
  - _Promoted to PLAN 6.5b (D-028 amendment, 2026-11-06)._
- [2026-11-06] **CI smoke deploy** — a CI job that builds the image, starts the server on the CPU with a tiny random model, and checks `/health`, `/ready` and one completion. It catches packaging breaks before the Space does — 6.5b / 6.7.
  - _Promoted to PLAN 6.5b (D-028 amendment, 2026-11-06)._
- [2026-11-06] **Failure-injection tests** — a fault-injection hook in the engine thread (slow step, exception at step N, OOM), to test that timeouts, 500s, readiness and draining behave as designed. This is a common interview question ("what happens when...") — 4.13 step 2 if cheap, else v1.1.
  - _Promoted to PLAN 4.13 step 2 (D-028 amendment, 2026-11-06)._
- [2026-11-06] **Serving architecture diagram** — one diagram in the README and SERVING.md: client → router → engine thread → slots / KV, plus where each metric is measured. Useful to point at in interviews — 6.1.
  - _Promoted to PLAN 6.1 (D-028 amendment, 2026-11-06)._
- [2026-11-06] **Offline eval gate per variant** — before any shadow / canary, a variant must pass a fixed eval: val loss (or perplexity) within a threshold, greedy agreement vs the current model on a fixed prompt set, and the benchmark. Gives 6.9's promotion pipeline its first stage as code, not prose — 6.9.
  - _Promoted to PLAN 6.9 (D-028 amendment, 2026-11-06)._
- [2026-11-06] **Written SLOs + error budget** — define 2–3 SLOs (TTFT p95, availability, 429 rate) in SERVING.md with their error budgets, and have the status page show the budget left — 6.5b.
  - _Promoted: defined in PLAN 4.13 step 3, shown in 6.5b (D-028 amendment, 2026-11-06)._
- [2026-11-06] **Queue sized from the SLO** (from the 4.13 step 3 load test). With `--max-queue 32`, admitted requests waited up to ~5.9 s past capacity, against a 1 s TTFT target. A queue of ≈ TTFT target × throughput (≈ 5 on TinyLlama at 8 slots) sheds more 429s but keeps admitted requests within the SLO. Possibly derive it automatically from the measured throughput. Rerun `load_test.py` at 6 / 8 req/s to compare — 6.5b (Space defaults) or 4.8a.
  - _Measured 2026-11-06 with `--max-queue 5`: TTFT p95 5.8 → 0.92 / 1.13 s at 6 / 8 req/s, same throughput, 429s 33 → 37 / 92 → 156 (PROGRESS). Manual sizing works; deriving it automatically (~1.5 h) stays parked — v1.1._
- [2026-11-06] **CUDA graphs for prefill** (from the 4.13 step 3 load test). ITL p99 is 30–45 ms vs p50 9 ms: when newcomers arrive, running slots wait for their prefill. The prompts are only ~15 tokens and newcomers already share one padded prefill, so the stall is the cost of one *eager* forward (about one uncaptured step), not prompt length. Chunked prefill (2026-11-04 entry) would not help this workload; it only helps long prompts. Capturing the prefill per (newcomer count, prompt-length bucket) would. ~2–3 h — v1.1.
- [2026-11-06] **Leading space dropped from served text** (found in the 4.8a API check). For "…named Max." our server returns `'Max was…'`; vLLM and llama.cpp return `' Max was…'` (SentencePiece `▁Max`). `_Detokenizer` strips the first token's leading space, as a standalone decode does. Tokens are identical, so benchmarks are unaffected, but OpenAI-compatible clients that join prompt + completion get "Max.Max". ~15 min plus a test — before 6.5b (the Space demo).
- [2026-11-07] **INT8 prefill in Triton** (from 4.10b). Prefill (M > 64) still uses the PyTorch cast path, so int8 TTFT is 10–130 ms above bf16 at batch 1–32. A Triton int8 GEMM for prefill, or casting each weight once per prefill and reusing it, would close it. Triton is capped at one day (D-016) — v1.1.
- [2026-11-07] **INT8 at batch 32 saves less than predicted** (from 4.10b). The decode step gets 0.9 ms faster with the Triton kernel, but the matmul bench predicts 3.0 ms (at batch 1: 2.4 of 2.6 ms). Profile one graph-replayed step at batch 32 (`scripts/profile_decode.py --batch 32`, bf16 vs int8-triton) to see where the time goes — v1.1.
- [2026-11-07] **INT8 serving comparison** (from 4.10b; the README vs vLLM / llama.cpp table is bf16 on all three). Ours with int8-triton vs llama.cpp Q8_0 vs vLLM with an 8-bit checkpoint, each in its own 8-bit format; never ours int8 against their bf16. Needs our server tested with int8 weights + slot graphs. Cheap first step (~30 min): one-stream only, ours int8 vs llama.cpp Q8_0. The full load sweep is ~2–3 h — after v1.
- [2026-11-06] **Serve the exported HF folder with our server** (from 4.8b). `--model` only knows TinyLlama, and `HFTokenizer` assumes Llama's `<s>` / `</s>`. Reading BOS / EOS from `tokenizer_config.json` instead would let the server load `runs/export/tinystories-30m` (or the Hub repo) the same way as TinyLlama, so the Space could pull weights from the Hub instead of baking them into the image. ~1 h — decide in 6.5b.
