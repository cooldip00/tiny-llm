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
- **Amendment (2026-11-06), 4.10b moves into v1:**
  - **Context:** the v1 run shows INT8 decode on TinyLlama at 0.75× bf16 at batch 1 and 0.36× at batch 32, and the README names it as a weakness: the weight is cast back to bf16 on every call. The rest of v1 (6.1, 6.3, 6.6, 6.5b) is about a day of work against the ~11-18 date, so there is room for the timebox.
  - **Choice:**
    - 4.10b (INT8 GEMV) comes first and 4.10a (fused RMSNorm) only if the half day has time left. 4.10b fixes a result the README already shows; 4.10a was only a warm-up.
    - 4.10 no longer waits for 4.7, which stays optional.
    - It is done before 6.3, so the resume bullets use the final numbers.
    - **Judged with CUDA graphs on**, bf16 and int8 alike. Without graphs, TinyLlama decode at batch 1 is bound by kernel launches (46 vs 128 tok/s), which would hide any kernel gain.
    - **Stop rule:** if int8 doesn't beat bf16 at batch 1 with graphs on within the half day, stop, record what was measured, and keep the README's current explanation.
  - **Rejected:** doing it after v1 (bullets would need rewriting, and the weakness would ship in the v1 README); 4.10a first (spends the timebox on the kernel that changes no published number).
- **Amendment (2026-11-07), both kernels in v1:** 4.10a runs after 4.10b even if the half day is used up (~1–1.5 h); Triton is capped at one day in total, and 4.10b's stop rule still applies. **Why:** about 11 days remain to v1 for ~18–20 h of planned work, replicas included (D-031). Expect a small end-to-end gain from 4.10a: with CUDA graphs on, launch overhead is already gone, and at batch 1 the norm reads one 2,048-wide activation per token. Its value is a second, simpler kernel under the same checks.

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
- **Amendment (2026-11-06):** five PARKING_LOT suggestions promoted, each into the step that builds what it checks:
  - failure-injection tests → 4.13 step 2
  - written SLOs with error budgets → defined in 4.13 step 3, shown in 6.5b
  - Docker Space + CI image check → 6.5b
  - serving architecture diagram → 6.1
  - offline eval gate as code → 6.9
  - About +2.5–3 h before v1 (+1 h in v1.1). v1 still ~11-18.

### D-029 — Results tables: generated RESULTS.md + README blocks from one run; INFERENCE.md stays a history · ACCEPTED (2026-11-06). Refines how 6.8 (D-025) is done
- **Context:** 6.8 planned to rewrite the INFERENCE.md tables from `runs/bench/*.json`. But INFERENCE.md is a dated history: each table is a snapshot from its step, and its prose quotes those numbers. Some snapshots have no JSON: the 11-04 TinyLlama serving JSON was overwritten by a rerun in another mode, and survives only as printed text in `sweep.log`. The README mixes numbers from different sessions, which differ by 10–20%.
- **Choice:**
  - `scripts/bench_all.py --run --power-mode X` runs every benchmark (30M and TinyLlama: generate, serving, INT8, profile; the TinyLlama logits check) into one fresh directory, `runs/bench/v1/`, from one clean commit in one session. vLLM / llama.cpp are not rerun (4.8a JSON reused).
  - Before each benchmark it waits until the GPU is ≤ 60 °C. Console output is copied to `run.log`; `manifest.json` records status, duration, cool-down, start temperature, power mode and commit per benchmark.
  - Generated outputs: `docs/RESULTS.md` (fully generated, one heat line per table: busy SM clock median / min, peak temperature, throttle reasons) and three README blocks between `<!-- bench:NAME -->` markers. Text outside the markers is never touched. Old README values stay in git history.
  - INFERENCE.md keeps its tables and gets pointer lines only.
  - Bench scripts refuse to overwrite an existing result without `--overwrite`, and default output names include the mode.
  - A benchmark is flagged if its provenance shows a hardware slowdown or a busy SM clock median below 1,400 MHz.
- **Why:** a dated history must not be rewritten, and headline numbers should come from one session with stated conditions, so ratios are like-for-like and every number has a JSON behind it.
- **Rejected:**
  - Regenerating INFERENCE.md in place: rewrites history, and some tables have no JSON.
  - Rendering from the existing scattered JSONs: mixed sessions and commits, some files with no provenance.
  - Flagging on the clock minimum: single busy samples dip far below 1,400 MHz in good runs.
- **Amendment (2026-11-06), after the first full run:**
  - **Throttle rule narrowed:** flag a hardware slowdown (`hw_slowdown`, `hw_thermal_slowdown`, `hw_power_brake_slowdown`), or a busy SM clock median below 1,400 MHz *together with* `sw_thermal_slowdown`. A low clock alone is not throttling: small or host-bound steps leave the GPU idle and it clocks down by itself (30M KV decode: 1,095 MHz median at 65 °C, no thermal reason). The old rule flagged 5 of 14 benchmarks; the new one flags 2, both reproduced by a rerun after cooling to ≤ 50 °C.
  - **Reruns are stated:** `--only` reruns into the same directory are marked in `manifest.json`, and RESULTS.md lists each one with its finish time and start temperature, so "one run" is never claimed for a number that came later.
  - **Allocator:** the two naive benchmarks run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (their full forward grows each step and fragments the caching allocator; first run crashed in cuBLAS). Recorded per benchmark in the manifest and `run.log`.

### D-030 — Plots: static SVGs drawn by bench_all from the same run as the tables · ACCEPTED (2026-11-06). How 6.2 is done
- **Context:** 6.2 asked for plots of loss curves, inference speedups and ablation trade-offs. The README is the first thing a reader sees, and GitHub strips JavaScript and iframes from it. The ablation (5.5) hasn't run yet. `checkpoints/` (with `train.log`) is git-ignored.
- **Choice:**
  - `src/tinyllm/plots.py` draws five plots with matplotlib (a dev dependency, version pinned by `uv.lock`). `scripts/bench_all.py` writes them to `docs/img/` whenever it renders the tables, from the same `runs/bench/v1/` JSON, so a plot never disagrees with its table.
  - README: TinyLlama decode at batch 1 against the bandwidth ceiling (top of Highlights), decode throughput vs batch size, and the training loss curve. RESULTS.md: those two, the serving comparison, and the decode-step profile (why CUDA graphs help).
  - One transparent SVG per plot, readable on GitHub's light and dark themes: mid-gray text, blue = ours, orange = HF, gray = context; the colour pair is checked for colour-vision deficiency on both backgrounds. Every mark carries its value as text, and each embed has alt text with the plotted numbers.
  - Byte-identical re-renders (fixed SVG hash salt, no date, text as paths), so `bench_all.py` leaves git clean when nothing changed.
  - The training curve comes from `docs/data/train_curve.json` (11 KB, made by `scripts/extract_train_curve.py` from `train.log`, with the log's SHA-256), so it can be redrawn from the repo. `.gitignore` re-includes `docs/data/`.
  - Ablation plots move to 5.5, which produces their data.
- **Why:** static images are what a README can show; generating them from the tables' JSON keeps one source of truth; one SVG for both themes is half the files and can't drift.
- **Rejected:**
  - Interactive plots (Plotly, Vega): the README can't run them; maybe later on a blog, the Space or Pages.
  - Mermaid `xychart`: no log scale, no value labels, limited styling.
  - W&B screenshots: not reproducible from the repo, and they need the account.
  - Separate light and dark images in a `<picture>` element: two files per plot for no gain over a transparent SVG.
- **Amendment (2026-11-06), the load-test plot:** a sixth plot, `load_sweep.svg`, in the README under "Results: vs vLLM and llama.cpp" (not in RESULTS.md, whose tables come from a different session). Two panels against offered request rate: completed requests / s with an "every request served" diagonal, and TTFT p95 on a log scale with the 1 s SLO line. Ours, vLLM and llama.cpp with 8 sequences each.
  - Data: `docs/data/load_sweep.json` (2 KB), made by `scripts/extract_load_sweep.py` from the 4.8a `runs/bench/load/4.8a/*-sweep.json` (git-ignored): one summary row per rate, with each file's SHA-256 and commit. The engine labels replace the sweep's `model` field, which holds a local path.
  - A third colour, aqua `#1aa775`, for the third engine; blue / orange / aqua passes the palette checks on white and `#0d1117`. Each engine also has its own marker and a label at the line end.
  - Its alt text is written by hand in the README, outside the generated blocks: the data is a fixed past session, not `runs/bench/v1/`.

### D-031 — Replicas on Hugging Face Spaces behind our own router Space, in v1 · ACCEPTED (2026-11-07). Extends D-028's 6.5b (one Space, a written scaling design)
- **Context:** D-028 plans one Docker Space and only a written scaling design (replicas, routing, autoscaling on queue depth). The server already has what a load balancer needs: `/health`, `/ready`, `/stats` with queue depth, and a 429 when the queue is full. About 11 days remain to v1.
- **Choice:**
  - After 6.5b's single Docker Space works: **two replica Spaces** from the same image, plus a **router Space**.
  - The router Space runs our `Router` (4.13) with a remote-replica mode:
    - it polls each replica's `/ready` and `/stats`
    - it sends each request to the ready replica with the shortest queue
    - it retries on another replica after a 429 or a connection error
    - it treats a sleeping replica as down and wakes it
    - it adds an `X-Replica` header beside `X-Model-Version`
  - The demo page and status page move to the router Space. **Spaces only:** no separate frontend.
  - A load test through the router: goodput with 1 vs 2 replicas, on CPU, labelled a routing demo, not a performance number. Performance claims stay with the laptop GPU numbers.
  - About +5–7 h, before 6.3, so the resume bullets can include it. The 6.1 diagrams come after it: the single-server diagram (one replica), plus a small one of visitor → router Space → replica Spaces.
  - **Cut order if late:** the 1-vs-2 load test, then the router and replicas move to v1.1. The single Space stays.
- **Why:**
  - It makes SERVING.md's written scaling design run.
  - In 6.9 (v1.1), the canary becomes its own replica Space: a better demo than two variants inside one process.
  - It needs no GPU, so it can fill 4.10b's cool-down waits.
- **Caveats:**
  - Free CPU Spaces sleep when idle, and waking one takes tens of seconds to minutes. Some visitors will see a cold start.
  - Read Hugging Face's terms on several Spaces serving one app before creating them; keep it to two replicas.
  - The README says the replicas demonstrate routing, not that the traffic needs them.
- **Rejected:**
  - A Vercel frontend with a function as proxy: streaming every token through the function uses the Hobby plan's 4 h / month of CPU, and the demo stops when it runs out. It also adds a second codebase and CORS.
  - Routing in the browser (the page reads each Space's `/stats`): no server-side canary or rollback, no central logs, and CORS.
  - A Cloudflare Worker as router: a second language and codebase.
  - Leaving it to v1.1: there is time before ~11-18.
- **Amendment (2026-11-08), after the feasibility test and the terms check (6.5b step 5):** the Choice above stands; these points add to it or narrow it. Full design: `docs/SERVING.md`, "Scaling beyond one server".
  - **Feasibility test: GO** (private test Space, `cpu-basic`). Without a token a private Space answers 404; with a fine-grained token, completions work and SSE streams one chunk per token. Paused, it answers 503 `text/html` in ~1.1 s. A paused Space woken through `restart_space` served its first 200 at 35.1 s, after a read timeout and a 502. Wake from sleep is still to be measured (2026-11-10). A real router Space calling a replica is first tested in the build.
  - **Terms caveat closed (read 2026-11-08):** ToS, Content Policy, Supplemental Terms with Hub Premium Features, Spaces docs. No page forbids several Spaces or a router Space, and none sets a Space count for PRO. They don't address forwarding between Spaces or keep-alive. The Content Policy lists "proxies… to bypass restrictions" and "excessive bulk activity" as abuse, and HF's own replicas are paid-only, so three limits apply:
    - no keep-alive: replicas sleep normally, are woken only for real visitor traffic, and polling stops 10 minutes after the last visitor
    - two replicas, fixed; no automatic creation of Spaces
    - presented as a learning demo of routing, not as a way to avoid paying for replicas
  - **Replicas are private**, duplicated from `CoolDip00/tiny-llm`, never from the test Space (its Dockerfile is test-only).
  - **One router token:** fine-grained, read and write on the two replica Spaces only (write is needed for restart), stored as a secret of the router Space.
  - **Retry narrowed:** at most one, on a different replica, only before the first byte reaches the visitor, and only after a `server_busy` 429 or a connection error. A `rate_limit_error` 429 goes to the visitor.
  - **Wake narrowed:** in the background, at most one wake per replica per 5 minutes, and a request never waits on a waking replica. If none is ready, the visitor gets a 503 with `Retry-After`.
  - **Rate limit and visitor hash move to the router.** It forwards the hash and `X-Request-ID`; replicas turn off their per-visitor limit and keep the queue limit.
  - **SLOs visitors see are measured at the router**, including retries and the extra hop. A "waking up" 503 counts against availability.
- **Amendment (2026-11-08), the existing public Space became the router:** the load test used a temporary private router Space. After it passed, the public `CoolDip00/tiny-llm` was switched from the single server to the router role, so the temporary router was deleted and the end state is three Spaces: `tiny-llm` (public router) and two private replicas. There is no separate fallback Space; `push_space.py --role server` puts the single server back. Replicas were pushed from the repo with `push_space.py --role replica`, not duplicated. **Why:** the README and model card keep one public link, and the router pushed was the code that was load tested (empty `git diff` for `src`, the Dockerfiles and `space/`).
- **Amendment (2026-11-10), waking an asleep replica:** the sleep check showed that an asleep private Space sends no "paused" page. A request is held while the Space wakes, then answered (test Space, owner's login: 200 at 25.1 s). The router's 2 s poll gave up first and marked both replicas down, so it never woke them: a visitor got 503 `waking_up`, and the replicas stayed asleep. Change: a replica whose last poll got no answer at all (connection error or timeout) is also woken when visitor traffic needs it, with the same limit of one wake per 5 minutes. Before restarting, the router reads the Space's stage from the Hub and restarts only `SLEEPING` or `PAUSED`. **Why:** the Hub's stage is a reliable sign that a Space is asleep, and the reply is not. Checking the stage keeps the rule that a running Space is never restarted. Still no keep-alive: wakes happen only for real visitor traffic.

### D-032 — RMSNorm defaults to the Triton kernel; every GPU table rerun in one session before v1 · ACCEPTED (2026-11-07)
- **Context:** 4.10a fused RMSNorm and residual add + RMSNorm into Triton kernels (`src/tinyllm/kernels/rmsnorm.py`). On TinyLlama with CUDA graphs, decode is 1.07× / 1.06× / 1.04× at batch 1 / 8 / 32 (128 → 137 tok/s at batch 1). The batch-1 step drops from 1,164 to 827 GPU ops; GPU busy time drops from 7.74 to 7.25 ms at the same 1,875 MHz clock. With the PyTorch norm, each norm runs about 7 small kernels. The 4.10a code shipped with `"torch"` as the default, so no existing number changed.
- **Choice:**
  - `RMSNorm(backend=None)` now takes `DEFAULT_NORM_BACKEND = "triton"`. The kernel runs only on CUDA with no gradient needed. Training, CPU and a missing Triton all fall back to the PyTorch code, so training and the CPU Space are unchanged. `set_norm_backend(model, "torch")` still switches back.
  - Provenance: every bench script records `norm_backend` (`norm_backend()` in `tinyllm.model`: `"triton"`, `"torch"` or `"mixed"`, with `"torch"` reported when the kernel won't run). The server's `/stats` reports it per variant.
  - **Before 6.5b, every GPU table is rerun in one fresh session** with `bench_all.py --run` into a new `runs/bench/v1b/`:
    - The tables the default changes: tl-naive, tl-kv, tl-kv-graph, tl-serving, tl-int8, tl-int8-graph, tl-profile, tl-logits, 30m-naive, 30m-kv, 30m-kv-graph, 30m-serving, 30m-int8, 30m-profile.
    - `--run` also reruns the three it doesn't change (tl-hf: the HF baseline; tl-int8-gemv: matmuls alone; tl-rmsnorm: the ablation, which measures both norms itself), because a fresh run directory renders only from its own JSONs. That's +25 min and keeps every table in one session.
    - Plus the 4.8a load sweep against vLLM and llama.cpp, all three engines in the same session.
    - About 115 min of benchmarks plus cool-downs (~2.5 h), plus the load sweep; fans on Max, one GPU job at a time.
  - The ablation table (PyTorch vs Triton norm) stays in RESULTS.md and INFERENCE.md 4.10a.
- **Why:**
  - vLLM fuses the residual add with RMSNorm in its own kernel, and llama.cpp fuses its norms too. In the comparison table we were the only engine running unfused norms; the default now matches what the other two do.
  - The repo's job is to show the best version of the engine. The expected gain at one stream is about 112 → 120 tok/s (58% → about 62% of the bandwidth ceiling), which narrows vLLM's lead from 1.44× to about 1.35×. That is an expectation, not a result, until the rerun.
  - One session for all tables means no table mixes the two norms, and it also replaces the old heat-flagged runs (30m-serving, tl-int8-graph).
- **Caveats:**
  - The kernel computes the same function, but the fp32 sum runs in a different order, so bf16 outputs differ by rounding: 1 of 8 greedy rows identical to the PyTorch norm over 256 tokens; teacher-forced argmax agreement 99.28%. The tests compare against the reference with tolerances (3 ulps in half dtypes), not bitwise.
  - Triton compiles each width and dtype on first use (about 1 s, then cached on disk). The benchmark warm-ups and the server's graph capture absorb it.
  - Validation loss computed under `no_grad` on the GPU runs the Triton norm, so it can differ from the training-time value in about the 4th–5th decimal. The training step itself (gradients) uses PyTorch.
  - The trained 30M checkpoint is not affected: the weights and the `state_dict` keys are unchanged (tested).
  - Until the rerun, the committed v1 tables were measured with the PyTorch norm (`norm_backend` absent from their provenance).
- **Rejected:**
  - Keeping `"torch"` as the default and reporting the kernel only as an ablation: the headline comparison against vLLM and llama.cpp would compare our unfused engine with their fused ones.
  - Switching the default without a rerun: tables would mix the two norms, and a reader couldn't tell which numbers include the kernel.
  - Rerunning only the TinyLlama tables: the 30M tables would be on a different configuration from the TinyLlama ones for no saving that matters (~30 min).

### D-033 — What v1 ships; the Docker image is CPU-only · ACCEPTED (2026-11-07)
- **Context:** after D-032 the engine has three GPU-only speedups: GQA (in the model, used by TinyLlama), the Triton kernels and CUDA graphs. The live demo runs on a free CPU Space, where none of them apply. The question was which model and which features each v1 deliverable carries, and whether the 30M model should be retrained with GQA first.
- **Choice:**

  | What | Model | GQA | Triton | CUDA graphs |
  |---|---|---|---|---|
  | Live demo: HF free CPU Space (6.5b), then 2 replicas + router (D-031) | our 30M | no | no (CPU fallback) | no |
  | HF model repo | 30M weights + model card | — | — | — |
  | GitHub repo: source + Docker image | 30M and TinyLlama 1.1B | yes | yes | yes |

  - **The Docker image for v1 is CPU-only**, the same image the Space runs. On a GPU machine, the source install (`uv sync`) runs every GPU path. A CUDA image goes to PARKING_LOT.
  - **No retraining of the 30M model** for v1, neither for GQA nor for the norm.
  - No paid GPU Space.
- **Why:**
  - The norm kernel computes the same function, so there's nothing to retrain for it.
  - The 30M model is plain multi-head attention, and at 30M its KV cache is tiny, so GQA would save almost nothing there. GQA is already shown working on TinyLlama (4.9a), which is where it matters. A 30M GQA-vs-MHA training run is a research question for Phase 5, after v1 (D-021).
  - A CPU image is small and builds in CI with no GPU. A CUDA image is 6–8 GB, needs `docker run --gpus` and the NVIDIA container toolkit to test, and adds about 1–2 h for something few visitors would run.
- **Caveats:**
  - The README must say plainly that the Space shows the serving stack on a small model on CPU, and that the speed numbers come from the laptop GPU (RTX 3070 Ti Laptop), not the Space.
  - A visitor with a GPU who wants to try the fast paths uses the source install, not Docker.
- **Rejected:**
  - A CUDA image in v1: the size and the extra testing path aren't worth it before ~11-18.
  - Retraining the 30M model with GQA before v1: about a day of GPU time for no visible change in the demo.
  - A paid GPU Space: ongoing cost for a portfolio demo.
