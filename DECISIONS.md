# Decision Log

Rules: entries are never edited after they're **ACCEPTED**. To change one, add a new entry that says `Supersedes D-xxx`.
Status: `ACCEPTED` · `OPEN` · `SUPERSEDED`

---

### D-001 — Scale: small model from scratch · ACCEPTED (2026-11-03)
- **Choice:** pretrain about 10–50M param models from scratch. No fine-tuning of existing big models.
- **Why:** 8 GB VRAM and roughly 10–15 effective TFLOPS. 6·N·D math puts ~30M params on ~1B tokens at about 5 hours.
- **Rejected:** a 124M+ model today (about 2 days of compute), parked.

### D-002 — Runtime: WSL2 + Ubuntu 24.04 · ACCEPTED
- **Why:** `torch.compile`/Triton, `flash-linear-attention` and vLLM need Linux. Native Windows support is missing or unreliable.
- **Rejected:** native Windows (no Triton or vLLM), dual boot (overkill).

### D-003 — Core framework: PyTorch 2.x (CUDA 12.x wheel) · ACCEPTED
- **Why:** industry standard and the best ecosystem. The wheel bundles its own CUDA runtime.
- **Rejected:** JAX (weak on laptop GPUs and Windows/WSL), TensorFlow.

### D-004 — Model code written from scratch (nanoGPT style) · ACCEPTED
- **Why:** learning is the goal, and the research needs a swappable attention layer.
- **Rejected:** HF Transformers model classes, plus HF Trainer and PyTorch Lightning, because they hide the internals.

### D-005 — Architecture: Llama-style decoder · ACCEPTED
- RMSNorm (pre-norm), RoPE, SwiGLU, no biases, tied embeddings.
- Weight names follow HF Llama conventions so we can export to vLLM and llama.cpp (Phase 4.8).

### D-006 — Package/env manager: uv · ACCEPTED
- **Why:** fast, lockfile, reproducible. **Rejected:** conda (slow, heavy), bare pip.

### D-007 — Tokenizer: our own byte-level BPE, vocab 8192, HF `tokenizers` · ACCEPTED
- **Why:** trains in minutes. A small vocab means a small embedding table, so more of the params go to the transformer layers. It also fits in uint16.
- **Rejected:** GPT-2 tiktoken (50k vocab is too large for a ~30M model).

### D-008 — Dataset: TinyStories first · ACCEPTED
- **Why:** small models produce coherent output on it quickly.
- FineWeb-Edu is a possible later upgrade (parked).

### D-009 — Data pipeline: HF `datasets` → numpy uint16 memmap · ACCEPTED
- **Why:** simple and fast, and RAM-friendly.

### D-010 — Attention kernel: `F.scaled_dot_product_attention` · ACCEPTED
- **Why:** built-in fused kernels (FlashAttention or memory-efficient attention). Nothing to compile.

### D-011 — Linear-attention research: `flash-linear-attention` (fla) · ACCEPTED
- **Why:** ready Triton kernels for Gated DeltaNet, Mamba-2, RWKV-7 and others.

### D-012 — Inference: our own PyTorch engine, with vLLM and llama.cpp as baselines · ACCEPTED
- **Why:** building it teaches the mechanics; the baselines show the production gap.
- **Rejected:** DeepSpeed and FSDP for training (no benefit on a single GPU).

---

### D-015 — Add a TinyLlama 1.1B inference track · ACCEPTED (2026-11-03)
- **Choice:** our inference engine also serves TinyLlama 1.1B (pretrained, Llama architecture, GQA with 4 KV heads).
- **Why:** at 30M params (about 60 MB bf16) decode is limited by kernel-launch overhead, not memory bandwidth, so the trade-offs we'd measure don't match production. A 1.1B model (about 2.2 GB bf16) fits in 8 GB and behaves like real serving.
- **Inspired by:** a LinkedIn post about a C++/CUDA engine for TinyLlama. We adopt its *method*: measure the hardware ceiling first, keep slow implementations as reference answers, reproduce every number with a script. We don't adopt its C++/CUDA scope.
- **Clarifies D-004:** HF Transformers is allowed **only** as a correctness reference in tests and as a benchmark baseline, never in our model or engine code.
- **Rejected:** a full C++/CUDA engine (weeks of work, and it would mostly repeat that project).

### D-016 — Triton kernels, timeboxed · ACCEPTED (2026-11-03)
- **Choice:** at most two Triton kernels: fused RMSNorm + residual add, and an INT8 weight-only GEMV for decode. Total budget about half a day, started only after Phase 4 steps 4.1–4.7 work.
- **Why:** kernel-level skill on the resume, at low cost. Batch-1 decode is a memory-bound GEMV, so the INT8 kernel ties directly to quantization (4.6) and to the bandwidth-ceiling measurements.
- **Rejected:** CUDA C++ kernels (higher cost, same lesson); writing kernels for everything (scope creep).

---

### D-013 — Logging backend: Weights & Biases · ACCEPTED (2026-11-03)
- **Choice:** W&B (now branded CoreWeave Forge). Account `cooldip00`, org `cooldip00-personal`.
- **Why:** run overlays and seed grouping for the Phase 5 ablations; a public project link works as resume evidence.
- **Plan limits (checked 2026-11-03 on the official pricing page):** Free = $0, 5 GB/month storage, public projects allowed. We log scalars and text samples only, never checkpoints.
- **Trial:** a 30-day Forge trial ends around **2026-12-02**. Then **downgrade to Free**. Use only Free-plan features.
- **Safety:** the API key is set only with `wandb login` inside WSL. Never put it in the repo or the docs.
- **Fallback:** TensorBoard. Logging sits behind one small wrapper, so switching is a one-line change.
- **Rejected:** TensorBoard as the primary backend (weaker run comparison, no shareable link).
- **Implemented:** Step 3.4, `src/tinyllm/logger.py` (wandb 0.31.0).

### D-014 — Code location: `~/llm` inside WSL · ACCEPTED (2026-11-03)
- **Choice:** code, docs and data all live in `~/llm` (`/home/kuladip/llm`), one git repo. From Windows the path is `\\wsl.localhost\Ubuntu-24.04\home\kuladip\llm`.
- **Why:** fast Linux file I/O (the standard practice for WSL), and docs travel with the code in git. The Ubuntu disk sits on D: (`D:\WSL\Ubuntu-24.04`), so storage is on D: anyway.
- **Rejected:** working through `/mnt/d` (slow file I/O); keeping two copies of the docs (they drift).
- The old `D:\ResumeProject\tiny-llm` now only holds a `MOVED.md` pointer, plus the old copies until they're deleted.

### D-017 — PyTorch version: 2.14.1 + CUDA 13.0 · ACCEPTED (2026-11-03). Supersedes the "CUDA 12.x wheel" detail of D-003
- **Choice:** `torch==2.14.1` from the `cu130` index, pinned in `pyproject.toml`.
- **Why:** the driver supports up to CUDA 13.3. 2.14.1 is a patch release, so more settled than the newest 2.15.0 (cu132), which ecosystem libraries like `fla` and Triton may not support yet.
- **Available on 2026-11-03:** cu126 2.14.1, cu128 2.11.0, cu129 2.13.0, cu130 2.14.1, cu132 2.15.0.
- **Revisit** only if a needed library requires a different version (new entry required).

### D-018 — Text cleanup: fix mojibake, ASCII-only stories · ACCEPTED (2026-11-03)
- **Problem:** 129,940 of the 130,115 train stories with non-ASCII text contain *mojibake*: UTF-8 punctuation that was decoded as Windows-1252, e.g. `“` → `â€œ`, `’` → `â€™`. That's about 2M junk sequences. Left as is, BPE would spend vocabulary on them and the model would learn to write `â€œ`.
- **Choice:** `tinyllm.data.clean_text`, applied through `load_clean()` in every step that reads the data:
  1. Map the mojibake sequences to plain ASCII (`"` `'` `-` `...`). The bare `â€` is a closing quote that lost its last byte, so it's replaced last.
  2. Map real typographic punctuation (`“ ” ‘ ’ – — …`) to ASCII too.
  3. Strip whitespace. Drop empty stories, and drop stories that still have any non-ASCII character.
- **Result:** train 2,119,719 → 2,118,859 (860 dropped, 0.04%); val 21,990 → 21,981 (9 dropped). 0 non-ASCII characters left. Tests in `tests/test_data.py`.
- **Rejected:** keeping the raw text (wastes vocabulary, garbled output); `ftfy`-style generic repair (an extra dependency for a fixed, known set of sequences); guessing at the leftover accented characters (too rare to matter).
- **Consequence:** the model only ever sees ASCII. That's fine for TinyStories, but revisit for FineWeb-Edu (parked).

### D-019 — Training hyperparameters (defaults in `TrainConfig`) · ACCEPTED (2026-11-03), to be confirmed by the smoke run (3.6)
- **Batch:** micro-batch 32 × 512 tokens, 8 accumulation steps, so **131,072 tokens per optimizer step**. The 1B-token main run is then ~7,630 steps.
- **Optimizer:** AdamW, betas (0.9, 0.95), eps 1e-8, fused kernel on CUDA. Weight decay 0.1 on 2-D weights (matrices and the tied embedding), 0 on RMSNorm gains.
- **LR:** linear warmup over 200 steps to a peak of **1e-3**, then cosine decay to 1e-4 (10% of peak) at `max_steps`.
- **Other:** gradient clipping at 1.0 (global norm), bf16 autocast with fp32 master weights and fp32 loss, TF32 allowed for leftover fp32 matmuls.
- **Why:**
  - The batch and betas follow llama2.c, which trained 15M–110M Llama models on TinyStories with ~131K tokens per step, betas (0.9, 0.95) and clip 1.0.
  - Peak 1e-3 is higher than the 5e-4/6e-4 used at 100M+ params, because smaller models tolerate and benefit from a higher LR. Warmup and clipping keep the early steps stable.
  - Beta2 0.95 (not 0.999) reacts faster to loss spikes and is standard for LLM pretraining.
- **First measurement (Step 3.2, 30–40 steps):** smoke ~131K tok/s; main ~74K tok/s (≈ 13 TFLOPS by 6·N·D), so 1B tokens ≈ 3.7 h.
- **Rejected:** Muon (parked for Phase 5), larger 0.5M-token batches (only ~2,000 steps for 1B tokens, too few updates for a small model).
- **Revisit** if the smoke run diverges or plateaus early. Change by a new entry, and Phase 5 keeps these values fixed across variants.

### D-020 — `torch.compile` the training step, on by default · ACCEPTED (2026-11-03)
- **Choice:** `TrainConfig.compile = True` (`--no-compile` turns it off). Only the training forward/backward is compiled. Eval, sampling and checkpoint saving use the eager model, which shares the same weights.
- **Measured** (main preset, 120 steps each, run in the order eager, compiled, eager so heat-up affects both; mean tok/s from step 40 on):

  | Run | tok/s | MFU | Loss at step 119 |
  |---|---|---|---|
  | eager 1 | 72.5K | 35.3% | 3.816 |
  | **compiled** | **114.1K** | **55.5%** | 4.102 |
  | eager 2 | 70.3K | 34.2% | 4.346 |

  - Compiled is **+57% tok/s**. Compiling adds ~20 s at startup and again on each `--resume`.
  - The two eager runs with the same seed already differ by 0.53 in loss at step 119. They pass the early ~5.5 plateau at different times, because GPU kernels aren't bit-reproducible run to run. The compiled loss falls between them, so compile doesn't change the math.
- **Why it helps:** the model is small (d = 512), so time goes to many small memory-bound kernels (RMSNorm casts, RoPE, SwiGLU, residual adds). Inductor fuses them.
- **Requires:** a C compiler in WSL (`build-essential`, SETUP Step 16), because Triton builds a small launcher for each kernel.
- **Guarded by:** a CUDA test that checks compiled losses match eager and that checkpoint keys have no `_orig_mod.` prefix. The CPU tests use eager (`tiny_cfg` sets `compile=False`).
- **Consequence for Phase 5:** run-to-run noise is large early in training, so ablations need several seeds (as D-013 planned) and should compare late or final val loss, not early steps.
- **Rejected:** keeping eager as the default (the main run would take ~4.2 h instead of ~2.7 h); compiling eval and sampling too (sampling changes the sequence length each token, which forces recompiles, for little gain).
