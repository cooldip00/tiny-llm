# Master Plan

Status keys: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in PROGRESS.md)

Each phase ends with **Done when** checks. Don't start the next phase until they pass.

---

## Phase 0 — Environment
- [x] 0.1 Install WSL2 Ubuntu 24.04 (installed with `--location D:\WSL\Ubuntu-24.04`, because C: only had 38 GB free)
- [x] 0.2 Inside WSL: `nvidia-smi` shows the RTX 3070 Ti (driver comes from Windows)
- [x] 0.3 Install `uv`, create the project, pin Python 3.12
- [x] 0.4 Install PyTorch (2.14.1 + cu130, D-017), check `torch.cuda.is_available()` and bf16 support
- [x] 0.5 Benchmark: bf16 matmul TFLOPS and memory bandwidth → record in docs/HARDWARE.md (39.7 TFLOPS peak GEMM, 38.3 sustained, 423 GB/s read)
- [x] 0.6 Resolve the OPEN decisions in DECISIONS.md (D-013 W&B, D-014 `~/llm`)
- [x] 0.7 `git init`, add `.gitignore` (data/, checkpoints/, wandb/, *.bin)

**Done when:** `scripts/check_env.py` prints the GPU name, bf16 = True and measured TFLOPS, and the numbers are in HARDWARE.md.

---

## Phase 1 — Data & Tokenizer
- [x] 1.1 Download TinyStories (`roneneldan/TinyStories`) with HF `datasets` (2.12M train / 22K val stories, 1.9 GB of text)
- [x] 1.1b Text cleanup (D-018): fix mojibake, drop empty and non-ASCII stories → `tinyllm.data.load_clean` (2,118,859 train / 21,981 val kept)
- [x] 1.2 Train a byte-level BPE tokenizer, vocab 8192, with HF `tokenizers`, on `load_clean("train")` → `tokenizer/tinystories-bpe8192.json` (4.12 chars/token on val)
- [x] 1.3 Tokenize `load_clean(...)` into `train.bin` / `val.bin` (numpy uint16 memmap, EOS between stories) → 465.4M train / 4.68M val tokens
- [x] 1.4 Sanity: encode→decode roundtrip, total token count, average tokens per story (`scripts/tokenize_data.py` checks + `tests/test_tokens.py`)

**Done when:** the roundtrip test passes, token counts are recorded in PROGRESS.md, and the .bin files load with memmap.

---

## Phase 2 — Model (from scratch)
- [x] 2.1 `ModelConfig` dataclass (see docs/ARCHITECTURE.md) → `src/tinyllm/config.py`, presets `smoke` / `main`
- [x] 2.2 Components: RMSNorm, RoPE, SwiGLU MLP → `src/tinyllm/model.py`, `tests/test_components.py`
- [x] 2.3 Attention interface plus registry, then the baseline `mha` using `F.scaled_dot_product_attention`. Matches a naive softmax oracle; causal; bf16 on CUDA close to fp32; 52/52 tests pass.
- [x] 2.4 Decoder block, full model, tied embeddings, init (std 0.02, residual projections scaled). `Block` + `Transformer` in `model.py`; 60/60 tests pass.
- [x] 2.5 Tests: output shapes, **causality** (changing a future token doesn't change past logits), param count matches the formula. Also initial loss ≈ ln(8192), gradients reach every param, bf16 autocast on CUDA; 72/72 tests pass.
- [x] 2.6 Overfit one batch: loss → ~0 within a few hundred steps. `scripts/overfit_batch.py`: smoke 9.07 → 0.0033 (below 0.05 at step 153), main → 0.0034 (step 209), 300 steps each.

**Done when:** all tests pass and the one-batch overfit succeeds.

---

## Phase 3 — Training
- [x] 3.1 Data loader: random fixed-length windows from the memmap, pinned memory. `data.get_batch`: 3.6M tokens/s to GPU at 32×512 (~50× what training needs); 79/79 tests pass.
- [x] 3.2 Loop: bf16 autocast, AdamW (fused), warmup + cosine LR, grad accumulation, grad clip 1.0. `src/tinyllm/train.py`, hyperparameters D-019; smoke ~131K tok/s, main ~74K tok/s; 87/87 tests pass.
- [x] 3.3 Eval every N steps, checkpoint save/resume, sample text at each eval. Resume is bit-identical (tested); `--stop_at` / `--resume` let the main run train in chunks; 93/93 tests pass.
- [ ] 3.4 Logging (backend per DECISIONS.md)
- [ ] 3.5 Throughput: tokens/sec and MFU, logged
- [ ] 3.6 Smoke run: ~14M config, ~30 min
- [ ] 3.7 Main run: ~30M config, a few hours (runs while Phase 4 work happens)

**Done when:** the main run's val loss has plateaued, samples are coherent stories, and the final val loss, tokens/sec and MFU are recorded.

---

## Phase 4 — Inference Engine (see docs/INFERENCE.md)
- [ ] 4.1 Naive generation (recomputes the full sequence each step): baseline tok/s
- [ ] 4.2 Sampling: temperature, top-k, top-p
- [ ] 4.3 KV cache: measure the speedup against 4.1
- [ ] 4.4 Static batching, then continuous batching (simple request scheduler)
- [ ] 4.5 Toy paged KV-cache allocator (vLLM-style blocks)
- [ ] 4.6 INT8 weight-only quantization: speed and quality (val loss) delta
- [ ] 4.7 Speculative decoding: ~14M model drafts for the ~30M model, measure acceptance rate and speedup
- [ ] 4.8 Export to HF Llama format, then compare against vLLM and llama.cpp (GGUF)

### Track B — Real-scale inference (D-015, D-016)
- [ ] 4.9 TinyLlama 1.1B in our engine
  - [ ] 4.9a Implement `gqa` attention (pulled forward from 5.2; TinyLlama uses 4 KV heads)
  - [ ] 4.9b Load TinyLlama safetensors weights and its tokenizer into our model code
  - [ ] 4.9c Correctness: our logits match the HF Transformers reference (oracle only, within bf16 tolerance)
  - [ ] 4.9d Rerun the 4.1–4.6 benchmarks on TinyLlama; report decode tok/s as a % of the bandwidth ceiling
  - Speculative decoding (4.7) stays on our models only, because the draft model must share TinyLlama's tokenizer
- [ ] 4.10 Triton kernels — **timeboxed to about half a day, start only after 4.1–4.7 work**
  - [ ] 4.10a Fused RMSNorm + residual add (first kernel, to learn Triton)
  - [ ] 4.10b INT8 weight-only GEMV for batch-1 decode (headline kernel)
  - For each kernel: `allclose` against the PyTorch reference, achieved GB/s as a % of the measured peak, and end-to-end tok/s delta in the engine
  - Stop after these two. Further kernels go to PARKING_LOT.md

**Done when:** the benchmark tables in docs/INFERENCE.md are filled in for every step (our model, TinyLlama, Triton kernels).

---

## Phase 5 — Research Ablation (see docs/RESEARCH.md)
- [ ] 5.1 Freeze the fixed-budget config (params, tokens, steps, LR schedule)
- [ ] 5.2 Implement variants: `mla`, `gdn_hybrid` (via `fla`). `gqa` already exists from 4.9a
- [ ] 5.3 Run every variant × 2–3 seeds
- [ ] 5.4 Measure: val loss, KV-cache bytes per token, decode tok/s at 512 / 2k / 8k context
- [ ] 5.5 Plots and analysis, including threats to validity

**Done when:** the results table in docs/RESEARCH.md is complete and the plots are saved.

---

## Phase 6 — Writeup
- [ ] 6.1 README: what, why, results, how to reproduce
- [ ] 6.2 Plots: loss curves, inference speedups, ablation trade-offs
- [ ] 6.3 3–4 resume bullets with real numbers
- [ ] 6.4 Review PARKING_LOT.md and pick follow-ups

---

## Rough timeline
| Block | Phases |
|---|---|
| Day 1 morning | 0, 1, 2 |
| Day 1 midday | 3 (main run starts in the background) |
| Day 1 afternoon/evening | 4.1–4.8 |
| Day 2 | 4.9–4.10 (Track B), start 5 |
| Day 3+ | 5, 6 |
