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

- [2026-11-04] **CUDA graphs for the continuous / paged engines** (4.11 stretch). Bucket by slot count and by the longest slot's length. `SlotKVCache` already keeps its lengths on the device; `begin_step` would need a fixed `_end` (the bucket) instead of `max(self.lens) + 1`. Paged attention needs a fixed block-table shape too.
- [2026-11-04] **`torch.compile(mode="reduce-overhead")` vs manual graphs** (4.11 comparison). After 4.11 the 30M step is GPU-bound across ~430 small kernels (2.5–4 µs each). Fusion (compile, or Triton in 4.10) is the next lever. Also check: graph replay costs 0.35–1.4 ms of host time on WSL, growing with node count. Is that WSL-specific?
