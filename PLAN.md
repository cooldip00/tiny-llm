# Master Plan

Status keys: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in PROGRESS.md) · `[>]` deferred to future work (see D-021; not dropped)

**Current priority (D-021, 2026-11-04):** **v1-preview** by about 2026-11-11, **v1** by about 2026-11-18 (see Phase 4 priority and the revised timeline). Phase 5 and Phase 7 are deferred until after v1.

Each phase ends with **Done when** checks. Don't start the next phase until they pass.

**Every step's done-when also includes (2026-11-06):** the README is updated if the step changes what a reader would see: a new module, doc, result, feature or command. `tests/test_readme.py` checks that the README names every module and links every doc; results and features are checked by hand. _Added after the README missed the server (4.13), the HF export (4.8b) and the vLLM / llama.cpp comparison (4.8a)._

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
- [x] 4.8 Export to HF Llama format, then compare against vLLM and llama.cpp (GGUF) _(all sub-steps done 2026-11-06)_
  - **Revised by D-024:**
    - [x] 4.8-spike vLLM + llama.cpp install spike, outside the repo (`~/spike/`). Done when each runs TinyLlama @ `fe8a4ea` and answers one `/v1/completions` request, and the project env is unchanged (snapshot before / after: package list, `check_env.py`, full test suite). Only `cuda-toolkit-13-0` from the wsl-ubuntu repo, never a driver package. _Was only in the 11-06 → 11-08 timeline row and the 4.8a text, so it was missed when that block finished early; tracked here from 2026-11-05._ _Done 2026-11-05: vLLM 0.33.0 and llama.cpp `61d4472` both serve `/v1/completions`; env snapshot unchanged, 221 passed. Issues and first numbers in PROGRESS._
    - [x] 4.8a TinyLlama in our engine vs vLLM vs llama.cpp (GGUF): same prompts and lengths, each as a % of the bandwidth ceiling. Install spike for both (vLLM in a separate venv) during the 4.9 days. _Done 2026-11-06: B=1 decode ours 58%, vLLM 84%, llama.cpp bf16 79% of the ceiling; rated load at 8 seats ours 4, vLLM 6, llama.cpp 4 req/s. Table in [docs/SERVING.md](docs/SERVING.md) "Load testing" and PROGRESS._
    - [x] 4.8b Export our 30M model to HF Llama format; done when HF logits match ours. GGUF conversion is a stretch (custom tokenizer). _Extended 2026-11-06: the tokenizer is exported too; also done when `AutoTokenizer` gives the same ids as ours._ _Done 2026-11-06 (`scripts/export_hf.py`): max |Δlogit| 1.4e-5, top-1 100%; greedy 8/8 identical to HF `generate()`; tokenizer 0 mismatches on the whole validation split. GGUF not attempted._
- [x] 4.11 Decode-step overhead (D-024): CUDA graphs or `torch.compile(mode="reduce-overhead")` on the KV-cached decode step. Profile one step before and after with `torch.profiler`. Done when outputs equal eager, and ms/step and tok/s are measured at B = 1 / 8 / 32. **Order: after 4.9c, before 4.9d.** Fixed-shape `generate()` first; bucketed shapes for the continuous / paged engines are a stretch. _Done 2026-11-04: `src/tinyllm/cuda_graph.py`, with manual CUDA graphs, a static KV cache and length buckets, enabled by `generate(cuda_graph=True)`. 30M bf16 decode is **4.9–8.1× faster** (8.0 → 0.99 ms/step at B=1). TinyLlama B=1 went from 38–48 to 124–133 tok/s (65–69% of the ceiling). Eager was host-bound, with the GPU idle 58–92% and downclocked. Replay is bit-identical to the static eager step. Against variable-shape eager, outputs match except at bf16 near-ties (margin ≤ 1 ulp): a different SDPA kernel. Continuous / paged engines not done._
- [x] 4.12 Benchmark provenance (D-025): every `runs/bench/*.json` records the git commit (and whether the tree is dirty), torch/CUDA versions, the weight dtype, and the GPU's SM clock / temperature / power / utilisation (`nvidia-smi`) plus the host CPU load, sampled before and after the timed runs. **Do first, before 4.9.** _Done 2026-11-04: `src/tinyllm/provenance.py` (before / during / after, `nvidia-smi -lms` sampler, throttle reasons, Windows CPU %), wired into bench_generate / bench_serving (also per row) / bench_quant. First finding: B=1 decode runs at ~860 MHz SM because the GPU idles between kernels._
- [x] 4.11b CUDA graphs in the continuous-batching engine (D-027). **Order: before 4.13.** _Done 2026-11-06: `SlotGraphDecoder` in `cuda_graph.py`, `run_continuous(cuda_graph=True | decoder)`, slot cache only (paged not done). TinyLlama, 64 requests: **581 vs 245 tok/s at 8 slots (2.4×), 956 vs 679 at 32 (1.4×)**. Tokens are identical to eager continuous batching (same divergence from solo at every position), because both use the same masked SDPA kernel. Capture 0.25–0.27 s per slot count. 232 passed. Details in PROGRESS._
  - `SlotGraphDecoder` in `cuda_graph.py`: fixed GPU buffers for tokens, slot lengths and slot state; one graph per window bucket (shared memory pool, as in 4.11); each step replays the smallest bucket that covers the longest slot. Prefill and sampling stay eager.
  - `run_continuous(..., cuda_graph=True)`, slot cache only. Paged KV is optional (+3–4 h) and raises "not supported" until done.
  - `bench_serving.py --cuda-graph`: continuous eager vs continuous + graphs on TinyLlama, same 64 requests, with provenance.
  - Done when greedy outputs equal eager up to bf16 near-ties (as in 4.11), requests joining / leaving and bucket switches mid-run are tested, the full suite passes, and the TinyLlama throughput with graphs is measured next to eager (272 tok/s at 8 slots today). Estimate ~1 day.
- [x] 4.13 OpenAI-compatible HTTP server (D-025): _Done 2026-11-06: greedy outputs equal `generate()` (step 2 tests); the same load test ran against our server, vLLM and llama.cpp in 4.8a._
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
- [x] 4.10 Triton kernels — **timeboxed to about half a day; in v1, before 6.3 (D-016 amendment, 2026-11-06); no longer waits for 4.7**
  - [x] 4.10b INT8 weight-only GEMV for batch-1 decode (headline kernel, **first**) _Done 2026-11-07: Triton kernel reads int8 directly; with CUDA graphs, decode 1.45× / 1.27× / 1.06× bf16 at B=1 / 8 / 32 (PyTorch cast path 0.39× / 0.42× / 0.52×); decode-step matmuls 1.87× bf16 at 81% of 423 GB/s. Stop rule passed. Fixed a WSL VRAM spill in `bench_quant` (time each variant alone). See docs/INFERENCE.md 4.10b._
    - Baseline first: rerun `scripts/bench_all.py --only tl-int8 --power-mode performance` once the GPU is cool (the v1 tl-int8 run was thermally throttled).
    - `bench_quant.py` gains a CUDA-graph mode; bf16 and int8 are both measured with graphs on (without them, batch-1 decode is bound by kernel launches and hides the kernel).
    - **Stop rule:** if int8 doesn't beat bf16 at batch 1 with graphs on within the half day, stop, record the numbers in PROGRESS, and leave the README's INT8 explanation as it is.
  - [x] 4.10a Fused RMSNorm + residual add, after 4.10b (~1–1.5 h); Triton capped at one day in total (D-016 amendment, 2026-11-07) _Done 2026-11-07: two Triton kernels (norm; residual add + norm), h bitwise equal to x + a, output within 3 ulps; with CUDA graphs, TinyLlama decode 1.07× / 1.06× / 1.04× at B=1 / 8 / 32; batch-1 step 1,164 → 827 GPU ops; add+norm alone 8.7× at 1 row, 97% of 423 GB/s at 4,096 rows. Made the default backend (D-032); every GPU table is rerun before 6.5b (4.10c). See docs/INFERENCE.md 4.10a._
  - [x] 4.10c **Rerun every GPU table in one session on the new default (D-032), before 6.5b** (~115 min of benchmarks plus cool-downs, ~2.5 h, plus the load sweep). Ask before starting; fans on Max; one GPU job at a time. _Done 2026-11-07: all 17 benchmarks into `runs/bench/v1b` (1 h 44 min), the load sweep into `runs/bench/load/4.10c` (1 h 11 min), plus logits vs HF with the PyTorch norm (bit-identical; the Triton norm is within 0.34). TinyLlama batch 1 127 → 137 tok/s (71% of the ceiling); one stream through the server 111.7 → 119.6 tok/s, vLLM's lead 1.44× → 1.35×, llama.cpp's 1.35× → 1.28×; rated loads unchanged except vLLM 8 seqs 6 → 8 req/s (the 1 s SLO edge). README and RESULTS render from `v1b`. See docs/INFERENCE.md 4.10c, docs/SERVING.md._
    - Back up `runs/bench` first. `bench_all.py --run --dir runs/bench/v1b --power-mode performance-fans-max-room-fan-overhead` (a fresh directory: nothing in `v1/` is overwritten). All 17 entries; tl-hf, tl-int8-gemv and tl-rmsnorm don't depend on the default but are rerun so every table comes from one session.
    - The 4.8a load sweep, ours vs vLLM vs llama.cpp, in the same session, one engine on the GPU at a time.
    - Check every JSON's provenance says `norm_backend: triton` and every heat line before quoting. Re-render RESULTS.md and the README from `v1b`; move the `--dir` default (`bench_all.py`, `results.render_results`) to `v1b`, and correct the README's "~1.5–2 h" for a full run (now ~115 min plus cool-downs); show the diff, commit.
    - Expected (not a result): one stream ~112 → ~120 tok/s, vLLM's lead 1.44× → ~1.35×.
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
- [ ] 5.5 Plots and analysis, including threats to validity (the ablation trade-off plots moved here from 6.2, D-030: add them to `tinyllm.plots`)

**Done when:** the results table in docs/RESEARCH.md is complete and the plots are saved.

---

## Phase 6 — Writeup
- [ ] 6.1 README: what, why, results, how to reproduce
  - a serving architecture diagram (D-028 amendment, ~20 min, after 4.13): client → router → engine thread → slots / KV, and where each metric is measured; also in docs/SERVING.md
    - _D-031 (2026-11-07): drawn after the replicas. This diagram is one replica; a second, small one shows visitor → router Space → replica Spaces._
- [x] 6.2 Plots: loss curves, inference speedups, ablation trade-offs. _Done 2026-11-06 (D-030): five SVGs in `docs/img/`, drawn by `bench_all.py` from the run behind the tables; three in the README, four in RESULTS.md. The ablation plots moved to 5.5._
- [ ] 6.3 3–4 resume bullets with real numbers
- [ ] 6.4 Review PARKING_LOT.md and pick follow-ups
- [ ] 6.5 Publish (added by D-021), in this order:
  - [x] 6.5a Public GitHub repo, pinned on the profile; README leads with the results table and plots. _Public 2026-11-05 (v1-preview): README leads with the results tables; plots still to add (v1). Plots added 2026-11-06 (6.2); pinned 2026-11-07._
  - [ ] 6.5b Hugging Face: our model with a model card, plus a small Gradio Space demo
    - _D-033 (2026-11-07):_ after 4.10c. The Space and the Docker image run our 30M model on CPU (no GQA, Triton or CUDA graphs there: the CPU fallbacks); the image is **CPU-only** (a CUDA image is in PARKING_LOT). The README says the speed numbers come from the laptop GPU, not the Space. No 30M retrain for v1.
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
    - _D-031 (2026-11-07), about +5–7 h, after the single Docker Space works, before 6.3:_
      - Two replica Spaces from the same image, plus a router Space running our `Router` with a remote-replica mode: polls `/ready` and `/stats`, shortest queue wins, retries elsewhere on a 429 or connection error, wakes a sleeping replica, adds `X-Replica`. CPU tests against fake replicas.
      - The demo and status pages move to the router Space. Spaces only, no separate frontend.
      - Load test through the router: goodput with 1 vs 2 replicas, labelled a routing demo.
      - Before creating them: read Hugging Face's terms on several Spaces serving one app.
      - If late, cut first the 1-vs-2 load test, then move the router and replicas to v1.1.
  - [ ] 6.5c W&B project set to public; link the training curves from the README
  - [ ] 6.5d Blog post (e.g. "Building an LLM inference engine on a laptop GPU"), linked from the README and resume
  - [ ] 6.5e LinkedIn: three posts, one per milestone (training done → KV cache + paged KV → TinyLlama results)
  - [ ] 6.5f r/LocalLLaMA (and r/MachineLearning with a [P] tag) once the TinyLlama numbers are in
  - [ ] 6.5g Direct outreach: send the repo to engineers at companies building inference products
- [ ] 6.6 Self-check before sharing (added by D-021): explain every file out loud without notes, e.g. why the KV cache helps, why decode is memory-bandwidth bound, how the paged allocator handles fragmentation, why tied embeddings, the 6N + 12LdT FLOPs formula, why compile helped. Anything that can't be defended stays off the resume.

- [x] 6.7 CI (D-025): GitHub Actions runs the CPU test suite on every push, with a status badge in the README. **Before 6.5a (v1-preview).** _Done 2026-11-05: workflow green; the badge is in the first README._
- [x] 6.8 One command regenerates every results table (D-025): `scripts/bench_all.py` runs each benchmark script and writes the INFERENCE.md tables from `runs/bench/*.json`, so the README numbers can be reproduced from the repo. Part of 6.1, v1.
  - _Done 2026-11-06 (D-029 + amendment): full run 11:29–12:57 UTC at commit `a765817`, NitroSense performance, 14 / 14 OK; `30m-serving` and `tl-int8` rerun with `--only` after cooling to ≤ 50 °C and reproduced. Output in `docs/RESULTS.md` and the README blocks; JSON in git-ignored `runs/bench/v1/`. See the PROGRESS log._
  - _Plan agreed 2026-11-06. Build ~3–4 h, then one ~1.5–2 h unattended GPU run._
  - **Why the plan changed:** INFERENCE.md is a dated history. Each table is a snapshot from its step, and the prose quotes those numbers. Regenerating it in place would rewrite history. Some snapshots also have no JSON.
  - **Steps:**
    1. **Fix output-name collisions first.**
       - `bench_serving.py --paged` / `--cuda-graph` share one default filename, and so do `bench_generate.py` reruns at other batch sizes. That is how the 11-04 TinyLlama serving JSON was lost.
       - Default names will include the mode, e.g. `serving_paged`, `serving_graph`, and a `_b…`/`_p…` tag for a non-default batch or prompt sweep. `bench_all.py` passes explicit `--out` paths under `runs/bench/v1/{30m,tinyllama}/`.
       - Bench scripts will refuse to overwrite an existing result unless given `--overwrite`.
    2. **`scripts/bench_all.py`**
       - `--run --power-mode X`: runs every benchmark into a fresh `runs/bench/v1/`, so all numbers come from one commit and one session.
         - 30M model: generate (naive, kv, kv-graph), serving, paged, INT8.
         - TinyLlama: hf, naive, kv, kv-graph (B up to 32), serving (with graphs), INT8.
         - The 4.9c logits check.
         - Not vLLM / llama.cpp: they need `~/spike`, and the 4.8a JSON from 2026-11-06 is reused.
       - Default (no `--run`): rebuilds the tables from the JSON in seconds, with no GPU. `--only <bench>` reruns one benchmark.
       - **Cool-down gate:** before each benchmark, wait until the GPU is below ~60 °C (adds ~10–15 min in total).
       - **Throttle warning:** flags a benchmark whose provenance shows a hardware slowdown (`hw_slowdown`, `hw_thermal_slowdown`, `hw_power_brake_slowdown`) or a busy SM clock *median* below 1,400 MHz, and suggests `--only`. The median, not the minimum: single busy samples dip to ~345 MHz even in good runs. _Narrowed after the run (D-029 amendment): a low median counts only together with `sw_thermal_slowdown`; on its own it is a host-bound GPU clocking down._
       - Copies all console output to `runs/bench/v1/run.log`. The 11-04 numbers survived only because `sweep.log` existed.
    3. **Where the output goes**
       - New `docs/RESULTS.md`: fully generated, and its header says so. One heat line per table: SM clock median / minimum, peak temperature, throttle reasons.
       - README results tables sit between `<!-- bench:NAME -->` markers and are filled from the same run.
       - INFERENCE.md keeps its tables. It gets two added lines: one pointing to RESULTS.md, and one pointing the 4.9d continuous-batching table at `sweep.log`.
    4. **CPU tests (they run in CI):**
       - rendering from small made-up JSON files
       - marker replacement leaves the rest of the file byte-identical
       - a missing JSON gives a clear message, not an empty table
       - default output names differ per mode
    5. **Docs:** D-029, because the tables go to RESULTS.md rather than INFERENCE.md. Then tick this step and add a PROGRESS entry with the run's numbers and conditions.
  - **Before the GPU run:** laptop plugged in, NitroSense in Performance mode, laptop raised or on a cooling pad, nothing else on the GPU, no vLLM / llama.cpp running.
  - **Expect:**
    - README numbers will shift (sessions differ by 10–20%), and the 11-04 figure of 1.37× gets replaced.
    - Correctness checks and ratios within one run are the reliable results. Absolute tok/s is labelled with that session's conditions.
  - **Recovered history** (printed text only, no JSON or provenance):
    - The 11-04 TinyLlama serving rows are in `runs/bench/tinyllama/sweep.log`: 198 / 272, 360 / 475, 669 / 738 tok/s (static / continuous at 8 / 16 / 32 slots).
    - The eager B=32 P=128 runs gave 1,586, 1,411, 1,298, 1,130 and 1,089 tok/s, as printed in the 11-04 session.
    - To do: back up `sweep.log` outside the repo.
  - **Lessons for every future benchmark:**
    - each mode gets its own output path
    - never overwrite raw results
    - copy console output to a log
    - compare absolute numbers only within one session
- [ ] 6.9 Release strategies (D-028, **v1.1**, after v1: ~11-19 → 11-21, ~9–10 h; design in [docs/SERVING.md](docs/SERVING.md)), built on 4.13's `Router`:
  - blue-green: two engines loaded, an admin switch, drain, instant rollback (~2 h)
  - shadow: mirror 10–20% of traffic when the live queue is idle; a report on token agreement, the latency difference and the impact on live latency (~2–3 h)
  - canary: a weighted router (5% → 25% → 100%), automatic rollback on error rate or SLO vs live, demonstrated with the load generator and a deliberately degraded variant (~2 h). With D-031's replicas, the canary runs as its own replica Space.
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
| 2026-11-12 → 11-18 | 4.11b graphs in the continuous engine (D-027, starts 11-06); 4.13 HTTP server + load generator; 4.8a TinyLlama vs vLLM / llama.cpp; 4.8b export; 6.8 one-command tables; 4.10b then 4.10a Triton before 6.3 (D-016 amendments); 4.10c rerun of every GPU table on the Triton norm (D-032) before 6.5b; replicas + router Space after 6.5b's single Space (D-031); rest of Phase 6 → **v1** |
| If ahead | 4.7 prompt-lookup decoding |
| 2026-11-19 → 11-21 | 6.9 release strategies (D-028) → **v1.1** |
| After v1 | Phase 5, Phase 7 |

Every day: ~30 min explaining that day's work out loud (feeds 6.6).
