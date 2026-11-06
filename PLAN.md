# Master Plan

Status keys: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in PROGRESS.md) · `[>]` deferred to future work (see D-021; not dropped)

**Current priority (D-021, 2026-11-04):** **v1-preview** by about 2026-11-11, **v1** by about 2026-11-18 (see Phase 4 priority and the revised timeline). Phase 5 and Phase 7 are deferred until after v1.

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
- [x] 3.4 Logging (backend per DECISIONS.md). `src/tinyllm/logger.py`: W&B (online/offline/disabled) or `none`; train/val scalars + a samples table; resume continues the same W&B run; 97/97 tests pass.
- [x] 3.5 Throughput: tokens/sec and MFU, logged. `ModelConfig.flops_per_token` (PaLM convention); `train/tflops` + `train/mfu` logged. Smoke ~135K tok/s, 13.1 TFLOPS, **31% MFU**; main ~73K tok/s, 15.0 TFLOPS, **35–36% MFU**; 98/98 tests pass.
- [x] 3.6 Smoke run: ~14M config, ~30 min. 2,000 steps / 262M tokens in 35 min; val 9.03 → **1.542**; coherent stories; sustained ~124K tok/s, 28.6% MFU. W&B run `cel67ssj`.
- [x] 3.7 Main run: ~30M config, a few hours (runs while Phase 4 work happens). 7,600 steps / 996M tokens in **2 h 27 min** (compiled); val 9.01 → **1.2825** (plateaued: −0.0026 over the last 400 steps); train 1.280 ≈ val, so no overfitting after ~2.1 epochs; sustained **115K tok/s, 56% MFU**; coherent stories. W&B run `ugd124lz`, weights `checkpoints/main/best.pt`.
- **Phase 3 done** (2026-11-03): main val 1.2825, 115K tok/s, 56% MFU recorded.

**Done when:** the main run's val loss has plateaued, samples are coherent stories, and the final val loss, tokens/sec and MFU are recorded.

---

## Phase 4 — Inference Engine (see docs/INFERENCE.md)

**Priority (D-021):**
- **v1-preview (~2026-11-11), in order:** 4.1 → 4.2 → 4.3 → 4.6 → 4.9a–d, then a first README and public repo
- **v1 (~2026-11-18), in order:** 4.4 + 4.5 together → 4.8, then the rest of Phase 6
- **Optional, only if ahead:** 4.7, 4.10
- Why 4.4 + 4.5 together: paged KV only pays off when continuous batching mixes requests of different lengths.
- Why 4.8 is must-do: comparing against vLLM / llama.cpp is what makes every other number credible.
- **Order update (D-022, 2026-11-04):** 4.1–4.3 finished a day early, so the order is now 4.4 → 4.5 → 4.6 → 4.9a–d → README + public (**v1-preview**, ~11-11) → 4.8 → rest of Phase 6 (**v1**, ~11-18). Dates unchanged.

- [x] 4.1 Naive generation (recomputes the full sequence each step): baseline tok/s (B=1 P=128: 104 tok/s decode; overhead-bound, see docs/INFERENCE.md)
- [x] 4.2 Sampling: temperature, top-k, top-p (`generate.sample_next`, 13 tests in `tests/test_generate.py`)
- [x] 4.3 KV cache: measure the speedup against 4.1 (0.93× at B=1, 3.1× at B=32 P=128; see docs/INFERENCE.md)
- [x] 4.4 Static batching, then continuous batching (simple request scheduler). Continuous is 1.34× static at 8 slots and 1.12× at 32, on a 64-request mixed workload (row util 59% → 86%); see docs/INFERENCE.md
- [x] 4.5 Toy paged KV-cache allocator (vLLM-style blocks). With the same 64 MiB of KV as 8 contiguous slots, it runs 25 sequences at once and 1.46× the throughput; 97% of claimed KV holds tokens (vs 52%). The gather costs ~7%. See docs/INFERENCE.md
- [x] 4.6 INT8 weight-only quantization: speed and quality (val loss) delta. Val loss Δ +0.0001 and matmul weights −50%, but decode is 0.78× at B=1: three PyTorch ops per Linear on an overhead-bound model. Rerun on TinyLlama in 4.9d; a fused kernel is 4.10b. See docs/INFERENCE.md
- [ ] 4.7 Speculative decoding: ~14M model drafts for the ~30M model, measure acceptance rate and speedup
  - **Revised by D-024:** prompt-lookup (n-gram) drafting instead of a ~14M draft model. Drafting costs ~0, and the target verifies k tokens per forward. Measure the acceptance rate and speedup on our model and on TinyLlama. Still optional.
- [ ] 4.8 Export to HF Llama format, then compare against vLLM and llama.cpp (GGUF)
  - **Revised by D-024:**
    - [x] 4.8-spike vLLM + llama.cpp install spike, outside the repo (`~/spike/`). Done when each runs TinyLlama @ `fe8a4ea` and answers one `/v1/completions` request, and the project env is unchanged (snapshot before / after: package list, `check_env.py`, full test suite). Only `cuda-toolkit-13-0` from the wsl-ubuntu repo, never a driver package. _Was only in the 11-06 → 11-08 timeline row and the 4.8a text, so it was missed when that block finished early; tracked here from 2026-11-05._ _Done 2026-11-05: vLLM 0.33.0 and llama.cpp `61d4472` both serve `/v1/completions`; env snapshot unchanged, 221 passed. Issues and first numbers in PROGRESS._
    - [ ] 4.8a TinyLlama in our engine vs vLLM vs llama.cpp (GGUF): same prompts and lengths, each as a % of the bandwidth ceiling. Install spike for both (vLLM in a separate venv) during the 4.9 days.
    - [ ] 4.8b Export our 30M model to HF Llama format; done when HF logits match ours. GGUF conversion is a stretch (custom tokenizer).
- [x] 4.11 Decode-step overhead (D-024): CUDA graphs or `torch.compile(mode="reduce-overhead")` on the KV-cached decode step. Profile one step before and after with `torch.profiler`. Done when outputs equal eager, and ms/step and tok/s are measured at B = 1 / 8 / 32. **Order: after 4.9c, before 4.9d.** Fixed-shape `generate()` first; bucketed shapes for the continuous / paged engines are a stretch. _Done 2026-11-04: `src/tinyllm/cuda_graph.py`, with manual CUDA graphs, a static KV cache and length buckets, enabled by `generate(cuda_graph=True)`. 30M bf16 decode is **4.9–8.1× faster** (8.0 → 0.99 ms/step at B=1). TinyLlama B=1 went from 38–48 to 124–133 tok/s (65–69% of the ceiling). Eager was host-bound, with the GPU idle 58–92% and downclocked. Replay is bit-identical to the static eager step. Against variable-shape eager, outputs match except at bf16 near-ties (margin ≤ 1 ulp): a different SDPA kernel. Continuous / paged engines not done._
- [x] 4.12 Benchmark provenance (D-025): every `runs/bench/*.json` records the git commit (and whether the tree is dirty), torch/CUDA versions, the weight dtype, and the GPU's SM clock / temperature / power / utilisation (`nvidia-smi`) plus the host CPU load, sampled before and after the timed runs. **Do first, before 4.9.** _Done 2026-11-04: `src/tinyllm/provenance.py` (before / during / after, `nvidia-smi -lms` sampler, throttle reasons, Windows CPU %), wired into bench_generate / bench_serving (also per row) / bench_quant. First finding: B=1 decode runs at ~860 MHz SM because the GPU idles between kernels._
- [x] 4.11b CUDA graphs in the continuous-batching engine (D-027). **Order: before 4.13.** _Done 2026-11-06: `SlotGraphDecoder` in `cuda_graph.py`, `run_continuous(cuda_graph=True | decoder)`, slot cache only (paged not done). TinyLlama, 64 requests: **581 vs 245 tok/s at 8 slots (2.4×), 956 vs 679 at 32 (1.4×)**. Tokens are identical to eager continuous batching (same divergence from solo at every position), because both use the same masked SDPA kernel. Capture 0.25–0.27 s per slot count. 232 passed. Details in PROGRESS._
  - `SlotGraphDecoder` in `cuda_graph.py`: fixed GPU buffers for tokens, slot lengths and slot state; one graph per window bucket (shared memory pool, as in 4.11); each step replays the smallest bucket that covers the longest slot. Prefill and sampling stay eager.
  - `run_continuous(..., cuda_graph=True)`, slot cache only. Paged KV is optional (+3–4 h) and raises "not supported" until done.
  - `bench_serving.py --cuda-graph`: continuous eager vs continuous + graphs on TinyLlama, same 64 requests, with provenance.
  - Done when greedy outputs equal eager up to bf16 near-ties (as in 4.11), requests joining / leaving and bucket switches mid-run are tested, the full suite passes, and the TinyLlama throughput with graphs is measured next to eager (272 tok/s at 8 slots today). Estimate ~1 day.
- [ ] 4.13 OpenAI-compatible HTTP server (D-025):
  - Turn `run_continuous` into an engine object with `add_request()` / `step()`, so requests can arrive while others decode.
  - `POST /v1/completions` takes token ids or text, with optional streaming.
  - Load generator: Poisson arrivals, reporting TTFT, inter-token latency and throughput. It is also the client for 4.8a (vLLM serves the same API).
  - Done when served outputs equal `generate()` for greedy requests, and the load test runs against both our server and vLLM. **v1, before 4.8a.**
  - Scope extended by D-028 (2026-11-06; design in [docs/SERVING.md](docs/SERVING.md)). Estimate now ~11–13 h in total.
  - [x] Step 1: `ContinuousEngine` (`add_request` / `step` / `abort`). _Done 2026-11-06, 239 passed._
  - [x] Step 2: the server (`src/tinyllm/server.py`, `src/tinyllm/obs.py`; adds fastapi, uvicorn, httpx): _Done 2026-11-06: 22 server tests (real uvicorn on 127.0.0.1) cover every done-when item, 3 reruns all green; full suite 276 passed; CLI smoke on `checkpoints/main/best.pt` (GPU): stream text == non-stream text._
    - `POST /v1/completions`: text or token ids, `max_tokens`, `temperature`, `top_p`, `top_k`, `ignore_eos`, SSE streaming ending with `[DONE]`, `finish_reason`, `usage`
    - per-request sampling, with rows grouped by identical settings
    - hardening: `/health`, `/ready`, a bounded queue → 429 + `Retry-After`, a per-client rate limit, a per-request timeout, abort on client disconnect, graceful shutdown (drain), an engine error → 500 + not ready
    - observability core: a JSON request log line (queue wait, TTFT, inter-token p50 / p95, tokens, finish reason, load at arrival), `X-Request-ID`, engine snapshots every ~5 s, `/stats` (5 min / 1 h windows), `/metrics` (Prometheus text format), `--log-level debug` with per-step timings and slow-request timelines, `/debug/state` (localhost only)
    - a `Router` with one variant; `X-Model-Version` (checkpoint hash + git commit) on every response and in every log line
    - CLI: `python -m tinyllm.server --model tinyllama | --ckpt PATH --slots 8 --cuda-graph`
    - failure injection (D-028 amendment, +30–45 min): a test-only fault hook in the engine thread (slow step, exception at step N, simulated OOM), so the timeout, 500, not-ready and drain paths are tested by forcing the fault
    - Done when: greedy served outputs equal `generate()`, streamed tokens equal the non-streamed ones, and tests cover 429 on a full queue, the rate limit, timeout, disconnect → slot freed, shutdown mid-request, each injected fault, and the metrics / log fields.
  - [x] Step 3: _Done 2026-11-06: SLOs in docs/SERVING.md; `scripts/load_test.py` + 3 tests (279 passed). TinyLlama, 8 slots + CUDA graphs, NitroSense Performance: **rated load 4 req/s** (506 tok/s, TTFT p95 0.80 s), saturation ~5.4 req/s / ~690 tok/s, past it 429s and 0 failures. The vLLM half of the 4.13 done-when runs in 4.8a._ The load generator (`scripts/load_test.py`): Poisson arrivals over a range of rates. Reports TTFT and inter-token p50 / p99, throughput vs rate, goodput at a latency SLO, and 429s / queue wait past capacity. The same client is used against vLLM in 4.8a.
    - SLOs written first (D-028 amendment, ~15 min): 2–3 targets in docs/SERVING.md (e.g. TTFT p95, availability, 429 rate), each with its error budget. Goodput is measured against the TTFT target.

### Track B — Real-scale inference (D-015, D-016)
- [ ] 4.9 TinyLlama 1.1B in our engine
  - [x] 4.9a Implement `gqa` attention (pulled forward from 5.2; TinyLlama uses 4 KV heads) _Done 2026-11-04: `src/tinyllm/attention/gqa.py`, HF head order, SDPA `enable_gqa` (measured 0.07–0.74× the time of repeat_kv + SDPA). Equals `mha` exactly at `n_kv_heads == n_heads`; all three caches and both engines match solo._
  - [x] 4.9b Load TinyLlama safetensors weights and its tokenizer into our model code _Done 2026-11-04: Chat v1.0 @ fe8a4ea (D-026). Our own safetensors I/O; meta-device build, strict check. 1,100,048,384 params, 2.05 GiB VRAM, load 3.8 s; coherent greedy chat._
  - [x] 4.9c Correctness: our logits match the HF Transformers reference (oracle only, within bf16 tolerance) _Done 2026-11-04: ours bf16 is bit-identical to HF bf16 (prefill + cached decode, 415 positions); same error vs fp32 (KL 2.6–3.3e-4); top-1 flips only at near-ties (margin ≤ 0.019); greedy text identical on 5/5 prompts. Fixed the transformers-5 `rope_parameters` config layout._
  - [x] 4.9d Rerun the 4.1–4.6 benchmarks on TinyLlama; report decode tok/s as a % of the bandwidth ceiling _Done 2026-11-04: B=1 P=128 decode: CUDA graphs 128 tok/s = 66.8% of the 192 tok/s ceiling (2.8× HF `generate`); eager KV and HF both ~23% (host-bound at ~22 ms/step). Continuous batching 1.10–1.37× static. INT8 Δ val loss +0.0010 but 0.54–0.80× bf16 speed (unfused cast; 4.10b). Found the WSL VRAM-spill slowdown and fixed it in the benchmark. See docs/INFERENCE.md 4.9d._
  - Speculative decoding (4.7) stays on our models only, because the draft model must share TinyLlama's tokenizer
- [ ] 4.10 Triton kernels — **timeboxed to about half a day, start only after 4.1–4.7 work**
  - [ ] 4.10a Fused RMSNorm + residual add (first kernel, to learn Triton)
  - [ ] 4.10b INT8 weight-only GEMV for batch-1 decode (headline kernel)
  - For each kernel: `allclose` against the PyTorch reference, achieved GB/s as a % of the measured peak, and end-to-end tok/s delta in the engine
  - Stop after these two. Further kernels go to PARKING_LOT.md

**Done when:** the benchmark tables in docs/INFERENCE.md are filled in for every step (our model, TinyLlama, Triton kernels).

---

## Phase 5 — Research Ablation (see docs/RESEARCH.md)

> `[>]` **Deferred (D-021, 2026-11-04):** future work after v1. The steps below stay as planned. Note: 5.2's `gqa` is still built in 4.9a.

- [ ] 5.1 Freeze the fixed-budget config (params, tokens, steps, LR schedule)
- [ ] 5.2 Implement variants: `mla`, `gdn_hybrid` (via `fla`). `gqa` already exists from 4.9a
- [ ] 5.3 Run every variant × 2–3 seeds
- [ ] 5.4 Measure: val loss, KV-cache bytes per token, decode tok/s at 512 / 2k / 8k context
- [ ] 5.5 Plots and analysis, including threats to validity

**Done when:** the results table in docs/RESEARCH.md is complete and the plots are saved.

---

## Phase 6 — Writeup
- [ ] 6.1 README: what, why, results, how to reproduce
  - a serving architecture diagram (D-028 amendment, ~20 min, after 4.13): client → router → engine thread → slots / KV, and where each metric is measured; also in docs/SERVING.md
- [ ] 6.2 Plots: loss curves, inference speedups, ablation trade-offs
- [ ] 6.3 3–4 resume bullets with real numbers
- [ ] 6.4 Review PARKING_LOT.md and pick follow-ups
- [ ] 6.5 Publish (added by D-021), in this order:
  - [ ] 6.5a Public GitHub repo, pinned on the profile; README leads with the results table and plots. _Public 2026-11-05 (v1-preview): README leads with the results tables; plots still to add (v1)._
  - [ ] 6.5b Hugging Face: our model with a model card, plus a small Gradio Space demo
    - _D-028 (2026-11-06; design in [docs/SERVING.md](docs/SERVING.md)), about +2–3 h:_
      - The Space runs our FastAPI server (same code as 4.13) with the demo page mounted on it. The 30M model is framed as story continuation (a TinyStories base model), with example openings.
      - A status page with live charts from `/stats` (requests / min, latency, queue depth).
      - An SLO (e.g. 95% of TTFT under 2 s), with compliance shown on the status page.
      - Logs kept in a private HF dataset (`CommitScheduler`, ~10 min).
      - `scripts/analyze_logs.py` usage report.
      - A privacy notice: metadata only, salted IP hashes.
      - A written scaling design: replicas, routing, autoscaling on queue depth.
      - If late, cut first the status-page charts, then the usage report.
    - _D-028 amendment (2026-11-06), about +1.5–2 h:_
      - A Docker Space, built from the same image as the local server; a one-line `docker run` in the README.
      - A CI job that builds the image, starts it on the CPU with a tiny random model, and checks `/health`, `/ready` and one completion.
      - The status page shows the error budget left for each SLO from 4.13 step 3.
  - [ ] 6.5c W&B project set to public; link the training curves from the README
  - [ ] 6.5d Blog post (e.g. "Building an LLM inference engine on a laptop GPU"), linked from the README and resume
  - [ ] 6.5e LinkedIn: three posts, one per milestone (training done → KV cache + paged KV → TinyLlama results)
  - [ ] 6.5f r/LocalLLaMA (and r/MachineLearning with a [P] tag) once the TinyLlama numbers are in
  - [ ] 6.5g Direct outreach: send the repo to engineers at companies building inference products
- [ ] 6.6 Self-check before sharing (added by D-021): explain every file out loud without notes, e.g. why the KV cache helps, why decode is memory-bandwidth bound, how the paged allocator handles fragmentation, why tied embeddings, the 6N + 12LdT FLOPs formula, why compile helped. Anything that can't be defended stays off the resume.

- [x] 6.7 CI (D-025): GitHub Actions runs the CPU test suite on every push, with a status badge in the README. **Before 6.5a (v1-preview).** _Done 2026-11-05: workflow green; the badge is in the first README._
- [ ] 6.8 One command regenerates every results table (D-025): `scripts/bench_all.py` runs each benchmark script and writes the INFERENCE.md tables from `runs/bench/*.json`, so the README numbers can be reproduced from the repo. Part of 6.1, v1.
- [ ] 6.9 Release strategies (D-028, **v1.1**, after v1: ~11-19 → 11-21, ~9–10 h; design in [docs/SERVING.md](docs/SERVING.md)), built on 4.13's `Router`:
  - blue-green: two engines loaded, an admin switch, drain, instant rollback (~2 h)
  - shadow: mirror 10–20% of traffic when the live queue is idle; a report on token agreement, the latency difference and the impact on live latency (~2–3 h)
  - canary: a weighted router (5% → 25% → 100%), automatic rollback on error rate or SLO vs live, demonstrated with the load generator and a deliberately degraded variant (~2 h)
  - A/B: sticky assignment by hashed client id, thumbs up / down, a significance test and CIs, validated with simulated users, plus the sample size a real verdict needs (~3 h)
  - variant id = checkpoint hash + git commit
  - a written promotion pipeline: offline gate → shadow → canary → live
  - the offline gate as code (D-028 amendment, ~1 h): a variant must pass val loss (or perplexity) within a threshold, greedy agreement with the current model on a fixed prompt set, and the benchmark, before shadow or canary

**Done when (6.5–6.6):** the repo is public with results, at least one post is live, and every resume bullet is backed by a number in PROGRESS.md.

---

## Phase 7 — Future Research (after v1)

> `[>]` **Deferred (D-021).** Not part of v1. Details and caveats are in [PARKING_LOT.md](PARKING_LOT.md) and the "Future directions" section of [docs/RESEARCH.md](docs/RESEARCH.md).

- [>] 7.1 Phase 5 ablation (above), first in line once v1 ships
- [>] 7.2 Weight-Space Programming replication: can a non-GPT editor (e.g. Claude) fix Qwen3.5-0.8B's reasoning loops by editing weights directly? (see PRIOR_ART PA-2)
- [>] 7.3 Pick one or two experiments from the "machines that learn, not imitate" research report (see PARKING_LOT and RESEARCH.md); check each one's prior work first

---

## Rough timeline
| Block | Phases |
|---|---|
| Day 1 morning | 0, 1, 2 |
| Day 1 midday | 3 (main run starts in the background) |
| Day 1 afternoon/evening | 4.1–4.8 |
| Day 2 | 4.9–4.10 (Track B), start 5 |
| Day 3+ | 5, 6 |

### Revised timeline (D-021, 2026-11-04)
The table above was the original plan. Phases 0–3 were done on Day 1 (2026-11-03).

| Block | Steps |
|---|---|
| 2026-11-04 → 11-05 | 4.1, 4.2, 4.3 (baseline, sampling, KV cache); push repo to GitHub |
| 2026-11-06 → 11-07 | 4.6 INT8; 4.9a–b (`gqa`, load TinyLlama weights and tokenizer) |
| 2026-11-08 → 11-11 | 4.9c–d (logits match HF, benchmarks vs the bandwidth ceiling); first README; repo public → **v1-preview** |
| 2026-11-12 → 11-15 | 4.4 + 4.5 (continuous batching + paged KV) |
| 2026-11-16 → 11-18 | 4.8 vLLM / llama.cpp comparison; rest of Phase 6 (6.1–6.3, 6.5, 6.6) → **v1** |
| If ahead | 4.7, 4.10 |
| After v1 | Phase 5, Phase 7 |

### Revised timeline (D-022, 2026-11-04)
4.1–4.3 and the GitHub push were done on 2026-11-04. The table above stays as the D-021 plan; this one replaces its order.

| Block | Steps |
|---|---|
| 2026-11-04 → 11-06 | 4.4 + 4.5 (static and continuous batching, paged KV) |
| 2026-11-07 → 11-08 | 4.6 INT8; 4.9a–b (`gqa`, load TinyLlama weights and tokenizer) |
| 2026-11-09 → 11-11 | 4.9c–d (logits match HF, benchmarks vs the bandwidth ceiling); first README; repo public → **v1-preview**. Checkpoint on 11-09 (D-022) |
| 2026-11-12 → 11-18 | 4.8 vLLM / llama.cpp comparison; rest of Phase 6 (6.1–6.3, 6.5, 6.6) → **v1** |

### Revised timeline (D-024 + D-025, 2026-11-04)
4.4–4.6 were done on 2026-11-04, 2–3 days ahead. The tables above stay as history; this one replaces their order. The dates are unchanged.

| Block | Steps |
|---|---|
| 2026-11-04 → 11-05 | 4.12 benchmark provenance; 4.9a `gqa`; 4.9b load TinyLlama; 4.9c logits match HF |
| 2026-11-06 → 11-08 | 4.11 CUDA graphs on decode; 4.9d TinyLlama benchmarks vs the ceiling; vLLM + llama.cpp install spike |
| 2026-11-09 → 11-11 | 6.7 CI; first README; repo public → **v1-preview**. Checkpoint on 11-09 (D-022) |
| 2026-11-12 → 11-18 | 4.11b graphs in the continuous engine (D-027, starts 11-06); 4.13 HTTP server + load generator; 4.8a TinyLlama vs vLLM / llama.cpp; 4.8b export; 6.8 one-command tables; rest of Phase 6 → **v1** |
| If ahead | 4.7 prompt-lookup decoding, 4.10 Triton |
| 2026-11-19 → 11-21 | 6.9 release strategies (D-028) → **v1.1** |
| After v1 | Phase 5, Phase 7 |

Every day: ~30 min explaining that day's work out loud (feeds 6.6).
