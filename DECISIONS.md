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

---

### D-021 — v1 scope: inference core first, research ablation deferred · ACCEPTED (2026-11-04)
- **Context:** v1 is time-boxed and focused on the serving engine. Time is the constraint, not compute.
- **Choice:** two milestones, following the step dependencies (4.3 → 4.4 → 4.5; 4.9 needs 4.3 plus `gqa` from 4.9a).
  - **v1-preview, by about 2026-11-11:** 4.1 → 4.2 → 4.3 → 4.6 → 4.9a–d, a first README (6.1 draft), public repo (6.5a). The first result worth sharing.
  - **v1, by about 2026-11-18:** 4.4 + 4.5 together → 4.8 → the rest of Phase 6 (including the new 6.5 publish and 6.6 self-check).
  - **Optional, only if ahead:** 4.7 speculative decoding, 4.10 Triton (D-016 timebox still applies).
  - **Deferred, not dropped:** Phase 5 research ablation, and the new Phase 7 future-research items. Marked `[>]` in PLAN.md. Resume them after v1.
  - The dates are estimates. If a milestone slips, cut optional steps first, then move the date. Don't skip correctness checks (e.g. 4.9c) to hit it.
- **Why:**
  - Training a small GPT from scratch is a common portfolio project (nanoGPT, llama2.c), so on its own it doesn't stand out.
  - The inference engine, measured against the hardware bandwidth ceiling and against production engines (vLLM / llama.cpp), is the differentiator, and it matches skills that are in demand right now.
  - **v1-preview first:** KV cache, INT8 and TinyLlama with matched logits already make a complete, measurable story. Batching and the production comparison build on it.
  - **4.4 and 4.5 go together:** a paged KV cache only pays off when many requests of different lengths share memory, which is what continuous batching creates. Alone, 4.5 has nothing to show.
  - **4.5 is must-do, not optional:** a block allocator for the KV cache ties directly to prior PyTorch caching-allocator experience (block splitting and merging against fragmentation). That link is the core of the writeup's story.
  - Phase 5 at 30M params costs a lot of GPU time (4 variants × 2–3 seeds) and isn't needed for v1's serving story.
- **Consequence:** the Phase 6 writeup says plainly that the ablation is future work. No Phase 5 claims on the resume until it's done.
- **Rejected:** doing Phase 5 now (weeks of GPU time, not needed for v1); skipping straight to the writeup with training results only (too common to stand out); adding new research tracks before v1 (scope creep; parked instead); a single one-week deadline for everything (too tight for the dependency chain).
- **Revisit** after v1, or earlier if research becomes the priority (new entry required).

### D-022 — Do 4.4 + 4.5 before 4.6 · ACCEPTED (2026-11-04). Supersedes the step order of D-021 only
- **Context:** 4.1–4.3 were finished on 2026-11-04, a day ahead of D-021's timeline (which allowed 11-04 → 11-05).
- **Choice:** new order 4.4 → 4.5 → 4.6 → 4.9a–d → first README + public repo (**v1-preview**) → 4.8 → the rest of Phase 6 (**v1**).
  - Everything else in D-021 stands: 4.4 and 4.5 together, 4.5 must-do, the optional steps, the deferrals, the slip rule.
  - Milestone dates stay at about 2026-11-11 (v1-preview) and 2026-11-18 (v1).
- **Why:** batching and the paged allocator build directly on the KV-cache code that was just written and tested (4.3), so doing them now is cheaper than coming back later. v1-preview also becomes a stronger first result (batching + paged KV + INT8 + TinyLlama).
- **Risk:** the TinyLlama work (4.9), which carries the bandwidth-ceiling story, moves later.
- **Revisit on 2026-11-09:** if 4.9 hasn't started by then, cut scope rather than move the date: e.g. ship v1-preview without 4.9 and add it in v1 (new entry required).
- **Rejected:** keeping D-021's order unchanged (fine too, but it leaves the batching work cold for a week).

### D-023 — Inference stores weights in bf16 · ACCEPTED (2026-11-04)
- **Context:** 4.1–4.6 ran inference with the training layout: fp32 weights under bf16 autocast, so every matmul re-casts its weight on every step. 4.6 measured weights stored in bf16 at **1.28× / 1.20× / 1.31×** decode tok/s (B = 1 / 8 / 32), with val loss 1.2825 → 1.2826 and argmax agreement with plain fp32 unchanged (99.34%). TinyLlama (4.9) is 1.1B params: 4.4 GB in fp32 would not leave room on the 8 GB GPU, but 2.2 GB in bf16 does.
- **Choice:**
  - From 4.9 on, inference loads weights in bf16 (`model.to(torch.bfloat16)` after loading), still under bf16 autocast. This covers our checkpoints and TinyLlama.
  - INT8 (4.6) stays an option on top. It quantizes from the loaded weights, and its non-matmul parts follow the model dtype.
  - Training is unchanged: fp32 master weights + bf16 autocast (D-019's "Other" line).
- **Consequence:**
  - The 4.1–4.6 numbers stay as recorded, labelled "fp32 + autocast". New tables state the weight dtype.
  - Where a later step compares against an earlier row, it reruns that row in bf16 rather than mixing layouts.
  - Correctness references stay fp32 (CPU tests, and the plain-fp32 comparisons in 4.6 / 4.9c).
- **Why:** it is free speed at no quality cost. It halves weight memory, which 4.9 needs. And it is the fair baseline for INT8 and for vLLM / llama.cpp (4.8), which serve bf16/fp16 weights.
- **Rejected:**
  - Keeping fp32 + autocast: slower, and it doesn't fit TinyLlama.
  - Re-measuring 4.1–4.5 now: costs time before v1-preview, and their within-run ratios still hold.
  - fp16: bf16 matches training and needs no loss scaling or overflow care.

### D-024 — Add a decode-overhead step; redesign 4.7; TinyLlama first in 4.8 · ACCEPTED (2026-11-04). Supersedes the 4.7 and 4.8 wording in PLAN only
- **Context:** 4.1–4.6 are done on 2026-11-04, 2–3 days ahead. Two things measured along the way change what the remaining steps can show:
  - Decode is overhead-bound. A step takes ~9.5 ms at any batch size, while weight reads are ~1.5% of that (4.1, 4.6).
  - INT8 and bf16 storage change speed only through op count, not through bytes (4.6).
- **Choice:**
  1. **New step 4.11: decode-step overhead.** CUDA graphs (or `torch.compile(mode="reduce-overhead")`) on the KV-cached decode step.
     - Profile one step with `torch.profiler` before and after, to back the "overhead-bound" claim with a trace.
     - Done when outputs are identical to eager, and ms/step and tok/s are measured on our model (B = 1 / 8 / 32).
     - Order: after 4.9c (TinyLlama logits match) and before 4.9d (TinyLlama benchmarks). Otherwise 4.9d mostly measures Python overhead: TinyLlama has 22 layers, which means an estimated ~25 ms/step eager against a ~5.2 ms bandwidth floor.
     - Scope: the fixed-shape decode step of `generate()` (B = 1 and static batch) first. The continuous / paged engines change shapes per step; padding them to fixed buckets is a stretch goal.
  2. **4.7 becomes prompt-lookup (n-gram) speculative decoding.**
     - Draft tokens come from matching the last n tokens against earlier context, so drafting costs ~0. The target verifies k drafted tokens in one forward.
     - Measure the acceptance rate and the speedup on our model and on TinyLlama.
     - It stays optional (D-021).
  3. **4.8: TinyLlama first.**
     - The main comparison is TinyLlama 1.1B in our engine vs vLLM vs llama.cpp (GGUF), with the same prompts and lengths, against the bandwidth ceiling. TinyLlama is already in HF format, so no export is needed.
     - Exporting our 30M model to HF Llama format becomes a sub-step (4.8b). Done when HF's logits match ours. Converting it to GGUF is a stretch goal: llama.cpp's converter may reject our custom BPE tokenizer.
     - A one-hour install spike for vLLM (separate venv, its own torch) and llama.cpp runs early, during the 4.9 days, so environment problems surface before v1 week.
- **Dates:** unchanged. v1-preview ~2026-11-11, v1 ~2026-11-18. The 2026-11-09 checkpoint (D-022) still applies. 4.11 is on the v1-preview path, so being ahead pays for it.
- **Why:**
  - Speculative decoding with the planned ~14M draft would likely be *slower* here. With both models overhead-bound, a draft step costs ~0.75 of a target step: at k = 4 and 80% acceptance, that's ~3.4 tokens per ~4 target-steps of time, ~0.85×. A ~14M draft would also have to be trained first.
  - Comparing against vLLM on the 30M model mostly measures vLLM's CUDA graphs against our Python overhead. TinyLlama is bandwidth-bound, with public reference numbers.
  - 4.11 is what makes the 4.9d "% of bandwidth ceiling" headline meaningful.
- **Rejected:**
  - Keeping 4.7 as written: it needs a new trained model and would likely measure a slowdown.
  - Training a draft model for TinyLlama: it needs TinyLlama's tokenizer, and the training cost is out of scope.
  - Comparing only the 30M model in 4.8: an overhead comparison, not an engine comparison.
  - Moving the dates earlier: keep the lead as a buffer for 4.9, the riskiest step left.

### D-025 — Add benchmark provenance, an HTTP server, CI, and one-command tables · ACCEPTED (2026-11-04)
- **Context:** we are 2–3 days ahead (D-024). The 4.5 benchmark showed 15–20% run-to-run noise that we could explain only after the fact. The 4.8 comparison needs a like-for-like serving setup, and the repo goes public at v1-preview.
- **Choice:** four additions, placed by when they pay off:
  1. **4.12 Benchmark provenance:** the commit, versions, weight dtype, GPU clocks / temperature / power / utilisation, and host load, stored in every `runs/bench/*.json`. Done first, before 4.9.
  2. **4.13 OpenAI-compatible HTTP server** on the continuous-batching engine, with requests arriving while others decode, plus a Poisson-arrival load generator. In v1, before 4.8a, so ours and vLLM are measured with one client and one API.
     - Web layer: FastAPI + uvicorn, or the standard library. The "no high-level frameworks" rule (CLAUDE.md) covers model, training and engine code, which stay ours. An HTTP layer is plumbing.
  3. **6.7 CI:** GitHub Actions runs the CPU test suite, before the repo goes public.
  4. **6.8 One command regenerates the results tables** from `runs/bench/*.json`, in v1, as part of 6.1.
- **Dates:** unchanged (v1-preview ~11-11, v1 ~11-18). The 11-09 checkpoint still applies. Cut order if late: 6.8 first, then 4.13 (4.8a then falls back to offline single-stream and batch benchmarks); never cut 4.12 or 6.7.
- **Why:**
  - Provenance makes every later number defensible.
  - The server turns 4.8 from a speed race into a serving comparison, and makes the repo usable.
  - CI and reproducible tables are cheap credibility for a public repo.
- **Rejected:**
  - Writing our own load generator only for vLLM: one client for both engines is the point.
  - gRPC or custom protocols: the OpenAI API is what vLLM and every client speak.
  - Running CI on GPU: no free GPU runners. The CPU tests cover correctness; GPU numbers come from the benchmark JSONs.

### D-026 — TinyLlama checkpoint: Chat v1.0, pinned; our own safetensors I/O · ACCEPTED (2026-11-04)
- **Context:** 4.9b loads TinyLlama 1.1B (D-015) into our model. Two pretrained checkpoints exist: `TinyLlama-1.1B-Chat-v1.0` (one bf16 `model.safetensors`, 2,098 MiB) and the base `intermediate-step-1431k-3T` (fp32, 4,196 MiB). The `safetensors` package isn't installed; `huggingface_hub` (via `datasets`) and `tokenizers` are.
- **Choice:**
  - Use **`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`**, pinned in `tinyllm.hf_llama`, so every number refers to the same weights.
  - Read and write `.safetensors` with our own ~90-line module (`tinyllm.safetensors_io`): JSON header + raw bytes, memory-mapped, each tensor copied once to its device. No new dependency.
  - Download with `huggingface_hub` into its cache (`~/.cache/huggingface`). Weights never enter the repo (`*.safetensors` is git-ignored anyway).
  - Tokenizer: `tokenizer.json` via `tokenizers`. HF Transformers stays a test-only reference (D-015), first used in 4.9c.
- **Why:**
  - Chat v1.0 is already bf16, which is the D-023 inference layout with no conversion. It answers prompts, so demos and the 4.13 server make sense. Ready-made GGUF conversions exist for 4.8a.
  - The format is simple enough that a reader is less code than a dependency decision. The writer is also what 4.8b's export needs.
- **Rejected:**
  - The base 3T checkpoint: fp32 (twice the download, cast at load), and no chat behaviour for demos. Logit matching (4.9c) works the same on either.
  - Adding the `safetensors` package: works, but it's one more dependency for ~90 lines of code.
  - Unpinned `main`: the weights could change under a published number.

### D-027 — CUDA graphs in the continuous engine (4.11b), before 4.13 · ACCEPTED (2026-11-05). Amends the D-025 order: 4.11b → 4.13 → 4.8a
- **Context:**
  - 4.11 put the decode step in CUDA graphs for fixed-shape `generate()` only: TinyLlama B=1 went from ~45 to 128 tok/s.
  - The continuous engine is still eager and host-bound: ~22 ms/step launching kernels, 272 tok/s at 8 slots.
  - 4.13's load test reports inter-token latency and throughput, which are set by that step time. vLLM, our 4.8a baseline, uses CUDA graphs.
  - The slot cache is already mostly graph-friendly: a fixed batch of `n_slots` rows, with lengths and state kept on the GPU. Only the attention window (as far as the longest slot) grows each step, and 4.11's length buckets already solve that.
- **Choice:**
  - New step 4.11b: CUDA graphs for the continuous engine's decode step, slot cache first. Prefill and sampling stay eager.
  - Paged KV graphs are optional (+3–4 h).
  - Order: 4.11b → 4.13 → 4.8a. Start 2026-11-06, ~1 day.
  - The graph runner is its own class, so 4.13's `ContinuousEngine.step()` reuses it unchanged.
- **Why:**
  - The 4.13 load test then shows graph-speed numbers from its first run, and isn't measured twice.
  - The 4.8a comparison with vLLM measures the engine design, not Python kernel-launch overhead.
  - Reuses tested 4.11 code. A measured before / after on `bench_serving`.
  - We are 6 days ahead, so v1 (~11-18) still holds.
- **Expected limits:**
  - TTFT barely changes, because prefill stays eager and queueing dominates it.
  - Some host overhead remains (scheduler, the per-step `.tolist()`).
  - Correctness is compared with eager up to bf16 near-ties, as in 4.11.
- **Rejected:**
  - 4.13 first: its first load test would be eager and go stale a day later. Also, the engine refactor would not reduce the graph work.
  - Skipping it until after v1: the vLLM serving comparison would mostly measure launch overhead.
  - Paged graphs first: block tables add work, and the slot cache already gives the measurement.

### D-028 — Serving as a production service: hardening + observability in 4.13 / 6.5b, release strategies in v1.1 · ACCEPTED (2026-11-06). Extends D-025's 4.13 scope; full design in docs/SERVING.md
- **Context:**
  - 4.13 as planned is an API over the engine plus a load test. It doesn't say what happens when the queue is full, a client disconnects or the engine fails. There is also no way to see how the public Space is used or how fast it answers.
  - Interviews for inference / ML-platform roles ask about evaluation under load, observability and release strategies (shadow, canary, A/B), not only kernels.
  - We have one laptop GPU and one free CPU Space: no fleet, no load balancer, no hosted monitoring stack.
  - Deployment: TinyLlama 1.1B on the local server (load tests, 4.8a); our 30M model on the public Space. The 30M model is a TinyStories base model (no SFT / RLHF), so the demo is story continuation via `/v1/completions`.
- **Choice:**
  - **4.13 step 2 (v1):**
    - the server: `POST /v1/completions` with streaming
    - per-request sampling
    - hardening: `/health`, `/ready`, a bounded queue → 429, a per-client rate limit, timeouts, abort on disconnect, graceful shutdown, engine errors → 500
    - observability core: a JSON request log, `X-Request-ID`, engine snapshots, `/stats` and `/metrics` (Prometheus text format), debug mode
    - a `Router` with one variant, and `X-Model-Version` logged from day one
  - **4.13 step 3 (v1):** a load generator reporting TTFT and inter-token latency p50 / p99, throughput vs rate, and goodput at an SLO.
  - **6.5b (v1):**
    - the Space runs the same FastAPI server, with the demo page mounted on it
    - a status page with charts and an SLO
    - logs kept in a private HF dataset
    - `scripts/analyze_logs.py`
    - a privacy notice
    - a written scaling design
  - **Privacy:** metadata only, never prompt or completion text; salted IP hashes, never raw IPs; private log dataset; notice on the Space.
  - **New step 6.9 (v1.1, ~11-19 → 11-21, ~9–10 h):** blue-green, shadow, canary with auto-rollback, A/B with statistics, and a promotion pipeline, built on the router.
  - **Cost:** 4.13 grows to ~11–13 h (+3–3.5 h), 6.5b +2–3 h. v1 stays ~11-18. If late, cut first the status-page charts, then the usage report.
- **Why:**
  - Each piece imitates a production practice at single-service scale, in the real format where one exists (Prometheus text, OpenAI API, SSE). So it can be explained the same way as the real thing.
  - The router and `model_version` fields cost minutes now and make every release strategy a routing rule later, with no server rewrite.
  - Release strategies need a working, observed server first, and they aren't needed for v1's claims. So they go right after v1 instead of delaying it.
- **Expected limits:**
  - A/B can't reach significance on a personal Space's traffic. We show the mechanism, validate it with simulated users, and report the sample size a real verdict would need.
  - On one machine, shadow traffic competes with live traffic. It runs only when the queue is idle, and its impact is measured.
  - Two TinyLlama copies for blue-green may not fit the laptop's VRAM. Use the 30M model there if so.
  - Scaling beyond one server is a write-up, not a build.
- **Rejected:**
  - Running Prometheus + Grafana (or a hosted APM): infrastructure to operate, not something we build. `/metrics` stays scrape-compatible instead.
  - Everything, release strategies included, in v1: +9–10 h would move v1.
  - Greedy-only server: a public demo needs temperature / top-p, and per-request settings are cheap once rows are grouped.
  - Logging prompt text for debugging: a privacy risk for little gain. Timings and token counts explain performance problems.
  - Exposing the laptop server publicly (tunnel): a security risk, and the Space already is the public endpoint.
