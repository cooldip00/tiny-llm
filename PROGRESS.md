# Progress

## Current
- **Phase:** 4 — Inference engine (Phase 3 complete)
- **Step (2026-11-07):** toward **v1** (~2026-11-18): 4.10b → 4.10a → 6.5b → replicas (D-031) → 6.1 → 6.3 → 6.6 → 6.5c–g. _Earlier: 4.11b → 4.13 → 4.8a → 4.8b → 6.8, all done 2026-11-06._ **v1-preview reached 2026-11-05**, 6 days early: repo public with the README, CI badge passing.
- **Next action:** the vLLM + llama.cpp install spike, which was planned for the 4.9 days but not done (PLAN 4.8a). It's one hour, in separate environments outside the repo, to surface install problems early. Then 4.13. _Spike done 2026-11-05: both engines run TinyLlama and serve `/v1/completions`; project env unchanged (see the log). Next: 4.13._ **Next action (2026-11-06):** 4.11b step 1, `SlotGraphDecoder` in `cuda_graph.py` (PLAN 4.11b). _4.11b done 2026-11-06 (see the log)._ **Next action:** push the local commits (one CI run), then 4.13 step 1, the engine object with `add_request()` / `step()`. _Pushed; 4.13 step 1 done 2026-11-06._ **Next action:** 4.13 step 2, the FastAPI server with `POST /v1/completions` (adds fastapi + uvicorn, D-025). _Step 2 done 2026-11-06 (see the log)._ **Next action:** 4.13 step 3: SLOs in docs/SERVING.md first, then `scripts/load_test.py`. _Step 3 done 2026-11-06 (see the log); the vLLM run of the same load test is part of 4.8a._ **Next action:** 4.8a, TinyLlama on vLLM / llama.cpp with the same `load_test.py`. _4.8a done 2026-11-06, which also closes 4.13 (see the log)._ **Next action:** 4.8b, export the 30M model to HF Llama format, then 6.8. _4.8b done 2026-11-06 (see the log)._ **Next action:** 6.8, one-command tables. _The full plan was agreed 2026-11-06 and is saved under PLAN 6.8, with the steps, the cool-down gate, the recovered history and the lessons. Not started: build it first, then the GPU run._ _6.8 done 2026-11-06 (see the log)._ **Next action:** push the 3 local commits (one CI run), then 6.2 plots for the README (6.5a), then 6.1 / 6.3 toward **v1**. _Pushed (CI green); 6.2 done 2026-11-06 (see the log)._ **Next action:** push 6.2 and check the plots on GitHub in light and dark, then 6.1 (README: serving architecture diagram, stale Status line, demo GIF), then 6.3. _README completeness pass done 2026-11-06 (see the log; Status line fixed)._ **Next action:** push both local commits (one CI run), check the 6 SVGs on GitHub in light and dark, then 6.1 (serving architecture diagram, demo GIF), then 6.3. _Pushed 2026-11-06 (`5a0ce75`, CI green); the 6 SVGs checked on GitHub in dark mode._ **Next action:** 6.1 (serving architecture diagram, demo GIF), then 4.10b Triton INT8 GEMV (half-day timebox, judged with CUDA graphs on; D-016 amendment), then 6.3. _Order changed 2026-11-07 (D-016 second amendment, D-031; see the log). CI green for `e8a20c4`; repo pinned (6.5a done)._ **Next action:** 4.10b (a cool `tl-int8` baseline first) → 4.10a → 6.5b single Docker Space → replicas + router Space (D-031) → 6.1 diagrams → 6.3.
- _Previous: 6.7 CI, first README, repo public (6.5a, without plots yet): done 2026-11-05, see the log._
- **Later (4.6):** per-channel INT8 weights for the Linear layers. Expect little speed gain on the 30M model (overhead-bound); the real test is TinyLlama in 4.9d. _Done 2026-11-04: as expected, quality Δ ≈ 0 and no speed gain here (0.78× at B=1); rerun on TinyLlama in 4.9d._
- **Blockers:** none
- **Open decisions:** none
- **Order (D-022):** 4.4 + 4.5 moved before 4.6 (4.1–4.3 finished a day early). Dates unchanged. Checkpoint 2026-11-09: if 4.9 hasn't started, cut scope instead of moving v1-preview. **D-024:** 4.9a → 4.9b → 4.9c → 4.11 (CUDA graphs) → 4.9d → README + public.
- **Scope (D-021):** **v1-preview** by about 2026-11-11 (4.1 → 4.2 → 4.3 → 4.6 → 4.9a–d, first README, public repo); **v1** by about 2026-11-18 (4.4 + 4.5 → 4.8, rest of Phase 6). 4.7 and 4.10 only if ahead. Phase 5 and Phase 7 deferred. _2026-11-07: 4.10 (both kernels) and the HF replicas are now in v1 (D-016 amendments, D-031); 4.7 stays optional._
- **Daily:** ~30 min explaining the day's work out loud (feeds 6.6).
- **Housekeeping:** push the repo to a public GitHub repo (no remote is set up yet; 20 commits so far). _Done 2026-11-04: pushed to github.com/cooldip00/tiny-llm, private for now; make it public at v1-preview (6.5a)._
- **Housekeeping:** delete the old doc copies in `D:\ResumeProject\tiny-llm` (keep `MOVED.md`). My automatic delete was blocked.
- **Reminder:** the W&B/Forge trial ends around 2026-12-02. Downgrade to Free then.

---

## Log
Newest first. One entry per work block: what was done, numbers, next.

### 2026-11-07 — Plan: both Triton kernels and HF replicas in v1 (D-016 amendments, D-031)
- **Triton (D-016 amendments, 11-06 and 11-07):** 4.10b INT8 GEMV first, then 4.10a fused RMSNorm (~1–1.5 h); no longer gated on 4.7; before 6.3; Triton capped at one day. Judged with CUDA graphs on, after a cool `tl-int8` rerun (the v1 run was throttled). Stop rule: if int8 doesn't beat bf16 at batch 1 within the half day, stop and record. No retraining: both kernels run on the existing weights.
- **Replicas (D-031):** after 6.5b's single Docker Space, two replica Spaces plus a router Space running our `Router` (shortest queue from `/stats`, retry on 429 or a dead replica, wake a sleeping one). Spaces only; a Vercel frontend was rejected (its 4 h / month of function CPU, a second codebase). About +5–7 h, before 6.3.
- **6.1 diagrams** move after the replicas: the single-server diagram becomes one replica, plus a small router → replicas diagram.
- **Housekeeping:** CI green for `5a0ce75` and `e8a20c4`; repo pinned (6.5a done); `sweep.log` backed up outside WSL (hash checked).
- **Next:** 4.10b, starting with the cool `tl-int8` baseline.

### 2026-11-06 — README completeness: vs vLLM / llama.cpp, the server, the export (D-030 amendment)
- **Why:** the README audit after 6.2 found results and features that never reached it: the 4.8a comparison with vLLM and llama.cpp, the 4.13 server, the 4.8b export, and modules / docs missing from the layout. It's the third time something was left out, so there is now a test and a PLAN rule.
- **README:**
  - New section "Results: vs vLLM and llama.cpp": the 5-row table from SERVING.md, 4 bullets (one stream, seats set capacity, past saturation, caveats), and the load plot.
  - Highlights: an HTTP server bullet (429s past capacity keep TTFT bounded; links the section). How it works: one server bullet linking SERVING.md. Training table: an Export row (HF logits within 1.4e-5, identical tokenizer ids).
  - Layout: `server.py`, `obs.py`, `logger.py`, `logit_compare.py`, `provenance.py`, `results.py`, `plots.py`; scripts row names `load_test.py` and `export_hf.py`; the docs row links all 8 docs. Reproduce: start the server, run the load test. Status line and Roadmap no longer list the comparison as to do.
- **Load plot** (`docs/img/load_sweep.svg`, D-030 amendment): `scripts/extract_load_sweep.py` → `docs/data/load_sweep.json` (2 KB, 8 rates × 3 engines, no local paths); two panels, completed req/s and TTFT p95 (log) vs offered rate. Checked as PNG on white and `#0d1117`. Our TTFT line is flat past saturation only because the full queue rejects with 429s, and TTFT counts served requests only, so the panel is titled "served requests" and the line end says it's capped; otherwise it reads as faster.
- **Prevention:** `tests/test_readme.py` (every `src/tinyllm` module in the layout, every `docs/*.md` linked, every README image exists); PLAN rule: a step's done-when includes the README.
- **Checks:** `bench_all.py` twice: "1 of 6 changed", then "0 of 6 changed", README / RESULTS / SVGs byte-identical. Tests +4 (load extractor, 3 README checks). Full suite **301 passed**.
- **Next:** push both commits (one CI run), check the 6 SVGs on GitHub in light and dark, then 6.1 / 6.3.

### 2026-11-06 — 6.2 plots for the README and RESULTS.md (D-030, 6.2 done)
- **What:** `src/tinyllm/plots.py` draws five SVGs with matplotlib 3.11.2 (new dev dependency); `bench_all.py` writes them to `docs/img/` with the tables, from the same `runs/bench/v1/` JSON:
  - README: `decode_b1` (B=1 by engine against the 192 tok/s ceiling, top of Highlights), `batch_scaling` (ours vs HF at B = 1 / 8 / 32, log scales, ceiling × batch), `train_loss` (training section).
  - RESULTS.md: `decode_b1`, `batch_scaling`, `serving` (TinyLlama scheduling; 30M contiguous vs paged in the same 64 MiB), `decode_profile` (GPU idle 76% eager vs 3% with graphs).
- **Design:** one transparent SVG per plot for GitHub light and dark (checked as PNG previews on white and `#0d1117`); blue = ours, orange = HF, gray = context; values on every mark; alt text with the numbers, generated from the JSON.
- **Determinism:** fixed SVG hash salt, no date, text as paths. `bench_all.py` twice: the second pass prints "0 of 5 changed" and the README is unchanged.
- **Training curve:** `scripts/extract_train_curve.py` → `docs/data/train_curve.json` (761 train / 39 val points, 11 KB, with the log's SHA-256), since `checkpoints/` is git-ignored. `.gitignore`'s `data/` also matched `docs/data/`; re-included with `!docs/data/` (`data/train.bin` still ignored).
- **Tests:** +4 (`tests/test_plots.py`: deterministic render, missing row names its benchmark, `bench_all.render` writes the plots then changes nothing, log parsing keeps a resumed step's last value). Full suite **297 passed** (README count updated).
- **Moved / parked:** the ablation plots to 5.5 (PLAN). An animated paged-KV explainer to PARKING_LOT (blog / Space; the README can't run JS).
- **Next:** push, check the SVGs on GitHub in both themes, then 6.1 / 6.3.

### 2026-11-06 — 6.8 GPU run: every results table from one run (6.8 done)
- **Run:** `bench_all.py --run --power-mode performance`, 11:29–12:57 UTC, commit `a765817`, clean tree, NitroSense **performance**, cool-down to ≤ 60 °C before each benchmark. **14 / 14 OK.** RTX 3070 Ti Laptop, driver 610.91, torch 2.14.1+cu130. JSON, `run.log` and `manifest.json` in git-ignored `runs/bench/v1/`.
- **Reruns** (`--only`, cool-down to ≤ 50 °C; marked in the manifest and listed in the RESULTS.md header): `30m-serving` (start 43 °C) and `tl-int8` (start 50 °C). Both reproduced the first run, so their ⚠ is the laptop, not a bad session:
  - `tl-int8`: bf16 46 / 374 / 1,482 tok/s vs int8 35 / 279 / 533 at B = 1 / 8 / 32 → **0.75× / 0.75× / 0.36×** (first run 0.75 / 0.74 / 0.35). Still `hw_thermal_slowdown` at 86 °C. The README's old 0.54× at B=32 came from a different session.
  - `30m-serving`: paged/16 at 32 slots **998 tok/s = 1.62×** continuous at 8 slots (615) in the same 64 MiB, 25 peak active, 42 preemptions, KV in use 52% → 97%. 84 °C with `sw_thermal_slowdown`.
- **Headline numbers** (now in the README and RESULTS.md):
  - TinyLlama B=1 P=128 decode: HF 44, naive 31, KV 48, **KV + CUDA graphs 127 tok/s = 66% of the 192 tok/s ceiling, 2.9× HF** (11-04: 128, 66.8%, 2.8×).
  - kv-graph grid (P = 32 / 128 / 512): B=1 130 / 127 / 112; B=8 875 / 814 / 629; B=32 2,332 / 1,984 / 1,214 tok/s.
  - Serving (TinyLlama, 64 requests): static 189, continuous 261 (**1.38×**), cont+graph 581 tok/s at 8 slots.
  - Profile: TinyLlama eager GPU idle 76%, graphs 3.4%, **3.60×**.
  - Logits vs HF: prefill (151 positions) and decode (264) `torch.equal`; greedy 5 / 5.
- **Heat:** 84–87 °C peaks in the long TinyLlama benchmarks; the cool-down gate controls only the start temperature. The OOM warnings in `tl-kv` / `tl-kv-graph` come from the untimed kv-vs-naive correctness check, not from timed rows.
- **Throttle rule narrowed** (D-029 amendment): the first rule flagged 5 / 14, three of them only for a low clock with no thermal reason (30M KV decode idles at 1,095 MHz, 65 °C). Now: a hardware slowdown, or a low busy median together with `sw_thermal_slowdown`. 2 flagged.
- **Docs:** README "221 tests" → 293, `bench_all.py` in Reproduce, roadmap item removed. RESULTS.md header lists the reruns. PLAN 6.8 ticked.
- **Not done:** `stash@{0}` (first render, before the reruns) kept until you say drop it. Not pushed.
- **Next:** push (one CI run), then 6.2 plots (6.5a still lacks them), then 6.1 / 6.3.

### 2026-11-06 — 6.8 build: one-command tables (D-029); GPU run not done yet
- **Overwrite guard:** `provenance.claim_output()` makes the five bench scripts (`bench_generate`, `bench_serving`, `bench_quant`, `tinyllama_vs_hf`, `profile_decode`) refuse an existing output unless given `--overwrite`, checked at startup before any GPU work. Default names now include the mode (`serving_graph_paged.json`, `kv_b8-32.json`, …). `bench_generate` also records warmup / repeats in provenance.
- **`src/tinyllm/results.py`** (pure rendering, no GPU): the list of 14 benchmarks, loading with a clear "missing file → `--only NAME`" message, heat lines (busy SM clock median / min, peak temperature, throttle reasons), and the tables: TinyLlama B=1 vs the ceiling, decode grids, INT8, serving, logits vs HF, decode profile. Renders `docs/RESULTS.md` and the three README blocks.
- **`scripts/bench_all.py`:** `--run --power-mode X` runs everything into `runs/bench/v1/` (refuses a dirty tree or a non-empty dir), waits for ≤ 60 °C before each benchmark, copies output to `run.log`, writes `manifest.json`, stops at the first failure. `--only a,b` reruns some; the default only renders. `--list` shows the commands (~90 min plus cool-downs).
- **Docs:** README results wrapped in `<!-- bench:highlights | tinyllama | serving -->` markers, "Full tables" now points to RESULTS.md. INFERENCE.md: a pointer to RESULTS.md, and the 4.9d continuous table's source (`sweep.log`). D-029. PLAN 6.8 wording fixed (throttle rule uses the busy-clock median).
- **Checks:** 8 CPU tests in `tests/test_results.py` (render from made-up JSON, marker replacement leaves the rest byte-identical and is idempotent, bad markers refused, missing file names the rerun, heat warnings, overwrite guard, per-mode default names, unique outputs). Full suite **292 passed**. With no results, the render exits 1 and lists all 14 missing files with their `--only` commands. On the real README with fake data, text outside the markers is unchanged.
- Committed `4d72033`.
- **First GPU run crashed (10:51 UTC):** `30m-naive` (bf16, never benchmarked before; 4.1 was fp32) failed at B=32 P=128 with `CUBLAS_STATUS_INTERNAL_ERROR` in the lm_head matmul, after 7 good rows. Cause: the naive engine's full forward grows by one position per step, so no cached block fits the next request. The caching allocator keeps reserving new blocks until the GPU is full: **7,991 MiB in use for a 475 MiB peak**, `free: 0` on a 192 MB request. PyTorch catches its own OOM and retries after emptying its cache (that is 4.1's `free: 0` warning on the same row), but cuBLAS's internal allocation does not.
  - Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the two naive benchmarks only (the other engines use fixed-size caches; graph capture is untouched). Same row, isolated, two runs each: **695 MiB**, no OOM, 940 / 921 tok/s vs 963 / 935 without it. `bench_all` records the env per benchmark in the manifest and the log. +1 test.
  - The failed attempt is kept in `runs/bench/v1-failed-1/`.
  - The GPU reached 84 °C during that one row, from a < 60 °C start: the cool-down gate controls only the start.
- **Next:** commit the fix, rerun into a clean `runs/bench/v1/`, then fill this entry's numbers and tick 6.8.

### 2026-11-06 — 4.8b: our 30M model exported to HF Llama format
- Pushed `e336501` (4.8a): CI green.
- Most of the model side existed from 4.9c (`save_as_hf`, HF loads our export with matching logits on random small models). New:
  - **Bug fixed:** `save_as_hf` wrote no token ids, so HF used LlamaConfig's defaults (bos 1, eos 2) and its `generate()` would stop on the wrong token. It now writes `eos_token_id` / `bos_token_id` (null when absent) to `config.json` and a `generation_config.json` (pad = EOS). It also refuses layer types other than mha / gqa.
  - `save_tokenizer_as_hf`: our tokenizer was trained with `tokenizers` (D-007), so the file already is a `tokenizer.json`. The added `tokenizer_config.json` sets `PreTrainedTokenizerFast` and EOS `<|endoftext|>` (id 0), with no BOS (like our server) and no space clean-up on decode.
  - `scripts/export_hf.py`: `checkpoints/main/best.pt` → `runs/export/tinystories-30m/` (git-ignored; 119.6 MB fp32 safetensors + tokenizer). Then it verifies on CPU in fp32 against HF Transformers (reference only, D-015) and writes `runs/export/tinystories-30m-verify.json` with provenance.
  - 3 CPU tests (they run in CI): the exported tokenizer gives our ids and text back (leading spaces, Unicode, EOS in text); HF `generate()` stops on our eos id; non-Llama layers are refused. Full suite **284 passed**.
- **Verification** (step 7599, val loss 1.282; 30M params, tied embeddings):

  | Check | Result |
  |---|---|
  | Logits, full forward (16 validation stories, 2,741 positions) | max abs diff 1.4e-5, top-1 agreement 100%, KL ≈ 0 |
  | Logits, prefill 8 + cached decode | max abs diff 1.0e-5, top-1 agreement 100% |
  | Greedy generation, 8 unseen prompts, ≤ 200 new tokens | 8/8 identical to HF `generate()`; 7 stopped at EOS |
  | Tokenizer, whole validation split (21,981 stories, 4.66 M tokens) | 0 id mismatches, 0 decode mismatches |
  | Round trip through our `load_from_dir` | identical weights |

  - So the export is a standard Llama: anyone can use it with `AutoModelForCausalLM` / `AutoTokenizer`, without our code.
- Not done: GGUF (stretch; llama.cpp's converter identifies BPE tokenizers by a fingerprint and won't know ours). Uploading to the Hub with a model card is 6.5b. Parked: let our server load the exported folder (PARKING_LOT 2026-11-06).
- **Next:** 6.8, one-command tables.

### 2026-11-06 — 4.8a: TinyLlama on ours vs vLLM vs llama.cpp (closes 4.13)
- `scripts/load_test.py --sequential N --weight-bytes B`: one request at a time; decode tok/s excluding the first token, as a % of the bandwidth ceiling (423 GB/s ÷ weight bytes). 2 tests (summary logic; a live sequential run against the tiny server). Full suite **281 passed**.
- **API check** (all three servers, same request): exactly `max_tokens` tokens, 14 prompt tokens, usage in the stream, finish `length`, `[DONE]`. vLLM and llama.cpp send usage in a last chunk with no text; the client already handled it.
- **Setup:** one server at a time, NitroSense **Performance** mode, fans Auto. TinyLlama @ `fe8a4ea`, bf16 unless noted. Prefix caching off (vLLM `--no-enable-prefix-caching`, llama.cpp `--no-cache-prompt`; confirmed in the server logs). llama.cpp: 8 slots × 2048 context, KV not unified. B=1: 20 × 256 tokens. Sweep: 128 tokens, 60 s per rate, local SLOs. 0 failures in every run; GPU back to 0 MiB after each server.

  | Setup | Seats | B=1 decode | % of ceiling | B=1 TTFT p50 | Rated load | Saturation | ITL p50 / p99 at 4 req/s | Past saturation |
  |---|---|---|---|---|---|---|---|---|
  | ours, 8 slots, graphs | 8 | 111.7 tok/s | 58.1% | 41 ms | 4 req/s | ~5.5 req/s, ~710 tok/s | 9 / 41 ms | 429s, TTFT p95 ~5.9 s |
  | ours, 32 slots, graphs | 32 | — | — | — | 8 | ≥ 12.2 req/s, 1,555 tok/s at 16 | 13 / 46 ms | 429s, TTFT p95 2.6–2.9 s |
  | vLLM 0.33, 8 seqs | 8 | 160.7 | 83.5% | 16 ms | 6 | ~8.5 req/s, ~1,090 tok/s | 7 / 15 ms | no 429s, TTFT p95 48 s at 16 |
  | vLLM 0.33, default (256) | 256 | 160.9 | 83.7% | 16 ms | 32 | ~34 req/s, ~4,370 tok/s | 7 / 9 ms | TTFT p95 21 s at 48 |
  | llama.cpp bf16, 8 slots | 8 | 151.0 | 78.5% | 12 ms | 4 | peak 5.4 req/s at 6, falls to 3.5 at 16 | 10 / 30 ms | no 429s, TTFT p95 203 s at 16 |
  | llama.cpp Q8_0, 8 slots | 8 | 233.9 | 64.7% of 361.6 | 11 ms | 4 | peak 6.2 req/s, falls to 3.8 at 16 | 9 / 29 ms | no 429s, TTFT p95 182 s at 16 |

  - **B=1:** vLLM is 1.44× us; llama.cpp bf16 1.35×. That's kernel speed: FlashAttention and fused ops vs our SDPA over padded slots.
  - **Under load, seats decide.** Little's law: 8 ÷ ~1.4 s per request ≈ 5.7 req/s for us, 8 ÷ ~0.9 s ≈ 8.9 for vLLM-8. 32 slots doubled our rated load (4 → 8 req/s), with ITL p50 rising 9 → 15 ms. vLLM's default (paged KV pool, 256 seqs) had only ~16 requests in flight at 16 req/s, and saturates at ~34 req/s.
  - At 8 seats we tie llama.cpp bf16 under load (rated 4 req/s each, ~690–710 tok/s) despite its faster B=1.
  - **Overload:** we shed (429s, bounded TTFT). vLLM queues without limit. llama.cpp's throughput drops by ~35% past its peak; cause not investigated (not our client: it handled 947 streams against vLLM).
  - **Throttling:** every run was thermally throttled (84–87 °C; `sw_power_cap`, `sw_thermal_slowdown`, some `hw_slowdown`). SM clock medians 1,515–1,770 MHz in the sweeps. At B=1 ours ran at 1,875 MHz / 76 °C, the others at 1,530–1,695 MHz, so our B=1 number is if anything flattered. Differences within ~15% are ties; the 1.3–1.5× gaps are not.
  - All records carry git `ccdc521+dirty` (this uncommitted `--sequential` mode). Raw JSON + server logs in `runs/bench/load/4.8a/` (git-ignored).
- Written up in docs/SERVING.md, "Load testing" → "Results". Prefix caching explained in SERVING.md (not implemented; A for v1.1, B in Phase 7). Parked: the leading space dropped from served text (PARKING_LOT 2026-11-06).
- **4.13 closed:** its done-when (greedy outputs = `generate()`, load test against our server and vLLM) is met.
- **Next:** 4.8b (export the 30M model to HF Llama format), then 6.8.

### 2026-11-06 — 4.13 step 3: SLOs + load generator; first load test on TinyLlama
- Pushed `8a97d70` + `f33e881`: CI green (49 s; GPU / TinyLlama tests skip there, the 22 server tests run).
- docs/SERVING.md, "SLOs" (written before measuring; changing a target needs a DECISIONS entry):
  - TTFT p95 ≤ 1 s local / ≤ 2 s Space
  - availability ≥ 99%
  - ≤ 1% 429s at or below rated load
  - error budgets; the burn-rate rule for the 6.5b status page
- `scripts/load_test.py`:
  - Poisson arrivals per rate (seeded); streamed, `ignore_eos`, fixed `max_tokens`, so the same work for our server and vLLM.
  - Client-side TTFT / ITL / latency p50 / p95 / p99, tok/s, 429s, failures, SLO checks per rate, and the **rated load** (the highest rate where it and every lower rate meet all SLOs).
  - JSON with provenance in `runs/bench/load/`. `--power-mode` is required.
  - 3 tests (Poisson rate, summary / SLO / rated-load logic, a live run against the tiny server that must shed at 100 req/s). Full suite **279 passed**.
- **Load test, TinyLlama 1.1B** (bf16, `--slots 8 --cuda-graph`, max queue 32; NitroSense **Performance** mode, fans Auto, GPU otherwise idle):
  - Setup: 8 story-opening prompts, 128 tokens each, 60 s per rate.

    | rate (req/s) | sent | ok | 429 | fail | tok/s | TTFT p50 / p95 / p99 | ITL p50 / p99 | latency p99 | SLO |
    |---|---|---|---|---|---|---|---|---|---|
    | 0.5 | 23 | 23 | 0 | 0 | 49 | 83 / 162 / 249 ms | 9 / 31 ms | 1.55 s | ok |
    | 1 | 56 | 56 | 0 | 0 | 119 | 48 / 85 / 100 ms | 9 / 23 ms | 1.41 s | ok |
    | 2 | 125 | 125 | 0 | 0 | 264 | 50 / 97 / 124 ms | 9 / 42 ms | 1.53 s | ok |
    | 3 | 173 | 173 | 0 | 0 | 366 | 51 / 222 / 604 ms | 9 / 44 ms | 1.98 s | ok |
    | 4 | 242 | 242 | 0 | 0 | 506 | 50 / 795 / 1050 ms | 9 / 42 ms | 2.47 s | ok |
    | 6 | 381 | 348 | 33 | 0 | 689 | 4.50 / 5.81 / 5.90 s | 10 / 43 ms | 7.31 s | miss: TTFT, 429 |
    | 8 | 454 | 362 | 92 | 0 | 693 | 5.25 / 5.83 / 5.90 s | 10 / 45 ms | 7.33 s | miss: TTFT, 429 |

  - **Rated load 4 req/s** (~506 tok/s). Saturation ~5.4 req/s, ~690 tok/s. That's above the 4.11b offline 581 tok/s at 8 slots, because here every request is the same length, so there's no long-tail drain.
  - Past capacity the server behaves as designed: the queue fills, the rest get 429, **0 failures**, and TTFT is capped at ~5.9 s = 32 queued ÷ 5.4 req/s. The server's `/stats` agrees (1,458 requests, error rate 0, 429 rate 8.6%, queue wait p95 5.7 s).
  - ITL p50 stays at 9–10 ms (~105 tok/s per stream) at every load. ITL p99 of 30–45 ms comes from prefills of newly admitted requests stalling the decode step. The prompts are ~15 tokens, so the stall is one eager prefill forward, not prompt length: capturing the prefill as a CUDA graph would fix it, chunked prefill would not (PARKING_LOT 2026-11-06).
  - At 0.5 req/s TTFT p50 is *higher* (83 vs 48 ms). Likely the GPU dropping clocks between sparse requests; provenance has the clock samples.
  - Server: 1,554 JSON log lines (1,458 request lines, no text); VRAM 2.5 GB, RSS 1.5 GB; clean drain on SIGTERM.
- Observation for later: a 32-place queue lets admitted requests wait ~6 s, far past the 1 s TTFT SLO. A queue sized from the SLO (≈ TTFT target × throughput ≈ 5) would shed more but keep admitted requests within the SLO. Parked (PARKING_LOT 2026-11-06).
- Measured the same day with `--max-queue 5` (same setup, Performance mode; `runs/bench/load/tinyllama-8slots-graph-q5.json`):

  | Rate (req/s) | Queue | OK | 429 | TTFT p95 | Latency p99 | tok/s |
  |---|---|---|---|---|---|---|
  | 6 | 32 | 348 | 33 | 5.81 s | 7.31 s | 689 |
  | 6 | 5 | 323 | 37 | **0.92 s** | **2.52 s** | 670 |
  | 8 | 32 | 362 | 92 | 5.83 s | 7.33 s | 693 |
  | 8 | 5 | 334 | 156 | **1.13 s** | **2.69 s** | 693 |

  - Throughput is unchanged (saturation ~5.4 req/s); the short queue only moves waiting time out of TTFT and into 429s, which a client can retry. Past capacity, TTFT p95 drops 5–6× and latency p99 ~2.8×.
  - At 4 req/s neither queue sheds and both meet the SLOs (rated load 4 req/s either way). The TTFT p95 there (0.45 vs 0.80 s) differs with the arrival sample, so it isn't a fair comparison.
  - At 8 req/s TTFT p95 is still 1.13 s: about 5 / 5.4 ≈ 0.93 s of queue wait plus prefill and step jitter. A queue of 4 would fit the 1 s target. The server default stays 32 for now. The Space default (6.5b) should come from this rule: queue ≈ TTFT target × saturation rate, minus some headroom.
- Next: 4.8a, the same `load_test.py` against vLLM (and llama.cpp) on TinyLlama.

### 2026-11-06 — 4.13 step 2, parts 3 + 4: the HTTP server and failure injection
- `src/tinyllm/server.py` (FastAPI + uvicorn, D-025 / D-028; design in docs/SERVING.md):
  - `EngineWorker`: one thread owns the `ContinuousEngine`; handlers talk to it through an inbox; tokens come back over a per-request `asyncio.Queue` via `call_soon_threadsafe`.
  - `POST /v1/completions`: text or token ids, `max_tokens`, `temperature`, `top_p`, `top_k`, `ignore_eos`, `return_token_ids`; SSE streaming ending with `[DONE]` (incremental detokenizer: the streamed text equals the non-streamed text); `finish_reason` stop / length; `usage`; OpenAI error shape.
  - Hardening: a bounded queue (slots + `--max-queue`) → 429 + `Retry-After`; a per-client token-bucket rate limit → 429; a per-request timeout → 504 with the slot freed; client disconnect → abort (499 in metrics); graceful drain on shutdown; an engine error → 500, `/health` + `/ready` 503, new requests 503; OOM → 503 for in-flight requests, then recovery.
  - Observability: a JSON log line per request (no text; salted client hash), `X-Request-ID` (the client's if sane), `X-Model-Version` (checkpoint hash + git commit), engine snapshots every 5 s, `/stats`, `/metrics`, per-step debug timings, slow-request timelines, `/debug/state` (localhost only).
  - `Router` with one variant (6.9 adds release rules here). CLI: `python -m tinyllm.server --model tinyllama | --ckpt PATH`.
  - `Faults` (test only): a slow step, an exception at step N, an OOM at step N.
- `engine.py`: `load()` (queued, busy) and `kv_used_frac()`.
- Deps: fastapi 0.162.0, uvicorn 0.54.0; dev: httpx 0.28.1.
- `tests/test_server.py`, 22 tests against real uvicorn on 127.0.0.1, port 0 (httpx's ASGI transport buffers streams):
  - greedy == `generate()` for id and text prompts, and with 7 concurrent requests on 3 slots
  - stream == non-stream; EOS → `stop`
  - 9 kinds of 400
  - 429 on a full queue; the rate limit; 504 (non-stream and stream)
  - disconnect (non-stream and stream) → slot freed
  - shutdown mid-request → 200
  - metrics, stats and log fields
  - the injected engine error and the injected OOM
- Bug found by the tests: the non-stream disconnect check only ran when no token had arrived for 0.25 s, so a busy request never noticed. It now checks on a timer.
- Three reruns, all green (~11 s each). Full suite **276 passed** (116.9 s).
- CLI smoke on `checkpoints/main/best.pt` (GPU, 4 slots): 200 with headers, the stream text equals the non-stream text, `/stats` filled, JSON logs, clean shutdown.
- Next: 4.13 step 3, SLOs first, then `scripts/load_test.py`.

### 2026-11-06 — 4.13 step 2, part 1: per-request sampling
- `engine.py`:
  - New frozen `Sampling(temperature, top_k, top_p)`, which checks its own values.
  - New `Request.sampling` (None = the engine's settings).
  - `ContinuousEngine._sample` makes one `sample_next` call per distinct setting. When every row has the same settings, the result and RNG use are exactly as before. Idle decode rows use the engine default.
  - `run_static` rejects per-request settings instead of ignoring them.
- New tests:
  - greedy and `top_k=1` rows mixed with hot sampled rows (staggered arrivals) equal solo `generate()`
  - hot rows do differ, and the same seed gives the same tokens
  - explicit settings equal to the engine's sample exactly as without them
  - bad settings are rejected
- Full suite **247 passed**.
- Part 2, `obs.py` (stdlib only):
  - `RequestRecord`: one JSON log line per request, never text; the client is a salted hash
  - `EngineSnapshot`
  - `Metrics`, thread-safe: rolling 5 min / 1 h windows for `/stats`; counters and TTFT / latency histograms by variant for `/metrics` (Prometheus text format)
  - `JsonFormatter` and `setup_logging`
  - 7 tests, all passing.
- Next: part 3, `server.py`.

### 2026-11-06 — Serving design: production layer, observability, release strategies (D-028)
- Design discussion before 4.13 step 2. Recorded in **D-028**, **[docs/SERVING.md](docs/SERVING.md)**, PLAN 4.13 / 6.5b / 6.9 and PARKING_LOT.
- What is deployed:
  - TinyLlama 1.1B on the local server, for load tests and 4.8a
  - our 30M model on the public CPU Space, framed as story continuation (TinyStories base model, no SFT / RLHF)
- 4.13 step 2 now also includes per-request sampling, hardening (429 queue limit, rate limit, timeouts, disconnect → abort, drain), an observability core (JSON request log, request ids, `/stats`, `/metrics`, debug mode) and a one-variant router with `X-Model-Version`. 4.13 is now ~11–13 h.
- 6.5b adds a status page, an SLO, private log retention, a usage report, a privacy notice and the scaling write-up (+2–3 h).
- New 6.9 (v1.1, ~11-19 → 11-21): blue-green, shadow, canary with auto-rollback, A/B.
- v1 date unchanged (~11-18).
- Next: 4.13 step 2.

### 2026-11-06 — 4.13 step 1: `ContinuousEngine`
- `engine.py`: `run_continuous`'s loop is now `ContinuousEngine` with `add_request()` (any time; returns an id; rejects requests that can never fit), `step()` (admit + one decode step; returns the new tokens as `Event(rid, token, done)`), `abort()` (frees the slot or drops the queued request) and `results` (popped by the caller). `run_continuous` is a thin wrapper, so every 4.4 / 4.5 / 4.11b test checks the refactor unchanged.
- TTFT and latency now count from each request's arrival. With all requests added at t=0 they equal the old numbers.
- Sampling settings stay per engine. Per-request sampling is not needed for the greedy done-when; decide in step 2 if the server needs it.
- New tests: late arrivals (slots and paged, with and without EOS) equal solo `generate()`, the event stream equals the result, abort of a running and a queued request, rejection when added, graphs equal eager under staggered arrivals (CUDA). Full suite **239 passed**.
- Next: step 2, the server.

### 2026-11-06 — 4.11b CUDA graphs in the continuous engine
- `SlotGraphDecoder` (`cuda_graph.py`): one graph per attention-window bucket (128, 256, … max_len) in a shared pool. `SlotKVCache` gets a fixed `window` (the bucket) and `advance_host()`, the host half of `end_step` (a replay runs only the device half). Prefill and sampling stay eager.
- `run_continuous(cuda_graph=True)` builds a decoder per call; passing a `SlotGraphDecoder` reuses its cache and graphs (the benchmark captures once, outside the timed runs). Paged KV raises "kv='slots' only".
- Tests: replay is bit-identical to the windowed eager step; the slot graph equals the eager slot step (gap 0); `run_continuous` with graphs equals eager at 1 / 3 / 8 slots, with and without EOS, with joins / leaves and bucket switches mid-run; decoder reuse doesn't recapture; TinyLlama teacher-forced, 0 flips in 240 slot-steps. **Graphs equal eager exactly here**, unlike 4.11's `generate()`: the eager slot step already uses a mask, so both use the same SDPA kernel.
- Full suite: one run had `test_model.py::test_bf16_autocast_cuda` fail (an unseeded model, checked within 0.1 of log V; passes alone). The new CPU-model test now builds its model inside `torch.random.fork_rng()` so it leaves the global RNG unchanged. Rerun: **232 passed** (157 s).
- `bench_serving.py --model tinyllama --cuda-graph` (64 requests, 9,004 tokens, median of 3, NitroSense **Performance** mode):

  | Slots | static | continuous (eager) | cont + graph | graph vs eager | latency mean / p95 (s), graph | capture |
  |---|---|---|---|---|---|---|
  | 8 | 181 | 245 | **581** | 2.37× | 7.4 / 13.5 | 0.27 s |
  | 32 | 631 | 679 | **956** | 1.41× | 4.8 / 8.2 | 0.25 s |

  - Same tokens as eager at both sizes: identical divergence from solo `generate()` (22/64 and 32/64 identical).
  - The graph rows run the GPU at 93–97% utilisation (eager: 44–65%), hot enough to throttle (`sw_thermal_slowdown`, 84 °C, SM 1575–1725 vs 1785–1875 MHz for eager). The gain is measured at a lower clock.
  - The eager rows are ~9% below the 2026-11-04 run (245 vs 272, 181 vs 198 at 8 slots). Eager is host-bound, so CPU speed and load move it.
  - A first run was discarded because the laptop switched from Default to Performance mode mid-run (memory clock 7001 → 7201 MHz). In Default mode, eager at 8 slots gave 209 tok/s. Benchmarks before 2026-11-06 ran in Performance mode; from now on each entry names the mode.
- Next: push the local commits, then 4.13.

### 2026-11-05 — Planned 4.11b and 4.13 (D-027)
- 4.13 plan made: an engine object with `add_request()` / `step()`, a FastAPI server with `/v1/completions` (streaming, `ignore_eos`), a Poisson load generator (httpx), and CPU tests. Estimate ~7–9 h.
- Found that the continuous engine has no CUDA graphs: eager at ~22 ms/step, so served latency would mostly measure kernel-launch overhead, and vLLM uses graphs.
- D-027: new step 4.11b (graphs in the continuous engine, slot cache first, ~1 day) **before** 4.13, so 4.13's first load test shows graph-speed numbers.
- Hosting: the server runs locally in WSL for benchmarks. The public demo stays 6.5b (HF Space).
- Next: 4.11b on 2026-11-06.

### 2026-11-05 — 4.8-spike: vLLM + llama.cpp install
- Everything is in `~/spike/` outside the repo, each in its own venv. Model: TinyLlama @ `fe8a4ea` from the HF cache.
- **System change:** `cuda-toolkit-13-0` (nvcc 13.0.88) from NVIDIA's wsl-ubuntu apt repo, installed into `/usr/local/cuda-13.0`, plus cmake 3.28. No driver packages; libcuda is still `/usr/lib/wsl/lib`. Not on PATH, and `.bashrc` is untouched.
- **vLLM 0.33.0:**
  - Its venv is 8.9 GB with its own torch 2.14.0+cu130; the install took 21 min.
  - Offline greedy decoding works, ~153 tok/s at B=1. That's rough: one run with prefill included, not comparable yet.
  - `vllm serve` is ready in 77 s and answers `/v1/completions`. The GPU is back to 0 MiB after it stops.
  - **Issue:** the sampler uses FlashInfer, which JIT-compiles on first use and fails with `ninja` not found, even though vLLM logs "FlashInfer kernels are disabled". Workaround: `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy doesn't need it). For 4.8a, also try `vllm download-kernels` (prebuilt) and record which sampler ran.
- **llama.cpp `61d4472`:**
  - CUDA build for sm_86 (llama-cli/server/bench/quantize).
  - **Issue:** CMake found the *Windows* CUDA 12.4 through the `/mnt/c` entries WSL adds to PATH. Fix for the build only: drop `/mnt/` from PATH and set `CUDAToolkit_ROOT=/usr/local/cuda-13.0`.
  - Converter venv: 0.9 GB, pins torch 2.11 CPU and transformers 4.57.6.
  - GGUF files: bf16 2,100 MiB, Q8_0 1,116 MiB.
  - `llama-bench`, pp128 / tg256, all layers on GPU, 3 reps:

    | GGUF | tg256 tok/s | pp128 tok/s |
    |---|---|---|
    | bf16 | 135 ± 6 | ~3,600 (± 2,100, noisy) |
    | Q8_0 | 238 ± 7 | ~8,500 (± 3,600) |

  - `llama-server` is ready in 4 s and answers `/v1/completions` (126 tok/s for that request).
  - Its greedy text is word for word the same as vLLM's.
- **First look, to verify in 4.8a under the same protocol:**
  - At bf16, B=1, vLLM (~153) and llama.cpp (135) are close to our 128.
  - llama.cpp Q8_0 at 238 tok/s is the bar for the fused INT8 kernel. Our INT8 path is 0.80× bf16 today.
- **Project env unchanged** (before vs after snapshot):
  - Same `pip list`; same hashes of `uv.lock`, `pyproject.toml`, `.bashrc` and `.profile`; same torch info.
  - **221 passed** after the spike as before.
  - check_env: bf16 matmul 39.3 vs 40.0 TFLOPS, copy/read 409/419 vs 409/426 GB/s (noise).
  - Torch still loads its own pip CUDA libraries (`nvidia/cu13/lib`), not `/usr/local/cuda`.
  - First test run after the spike: 136 s vs 84 s before; the rerun took 97 s. Likely a cold file cache after the 3 GB of GGUF files (7 GB RAM), not an env change.
- Cleanup if wanted: `rm -rf ~/spike ~/.cache/vllm`, then `apt remove cuda-toolkit-13-0 cuda-keyring` + `apt autoremove`.
- Next: 4.13 (D-025).

### 2026-11-05 — Repo public → v1-preview
- github.com/cooldip00/tiny-llm is public:
  - The page loads without a login, and the CI badge shows "passing" to anonymous visitors.
  - About box set: a description and 10 topics.
- v1-preview was planned for about 2026-11-11, so this is 6 days early.
- 6.5a is still open until the README plots are in (v1, PARKING_LOT "README polish").
- Next: the vLLM + llama.cpp install spike, then 4.13 (D-025).

### 2026-11-05 — First README (6.1 draft) + MIT license
- `README.md` is results first, for the 30-second reader (6.5a):
  - CI badge and 4 highlight bullets.
  - Results tables: TinyLlama decode against the ceiling, serving, training.
  - Two Mermaid diagrams: the model block and the serving request path.
  - Repo layout, reproduce commands, roadmap.
- Every number is checked against INFERENCE.md and this log.
- `LICENSE`: MIT. Without a license, a public repo is "all rights reserved".
- Left for v1, in PARKING_LOT "README polish": plots, a demo GIF, Space and blog links, a social preview image, design-decision and plain-language sections.
- Gap: `runs/` is git-ignored, so the raw benchmark JSONs aren't in the repo. Decide in 6.8 whether to commit them.
- 6.7 is closed by the badge.

### 2026-11-05 — Step 6.7 CI
- `.github/workflows/ci.yml`: runs on push, pull request and manual dispatch, on ubuntu-24.04 with uv and Python 3.12.
  - It installs CPU torch 2.14.1 before the project, so the CUDA 13.0 wheels from pyproject (D-017) are never pulled.
  - The job fails if any `nvidia-*` package is installed.
- Expected skips (22):
  - 18 need CUDA.
  - 2 need `data/*.bin`.
  - 2 need TinyLlama in the HF cache.
- Results:
  - Local simulation (fresh clone, empty HF cache): 199 passed, 22 skipped, no nvidia packages.
  - GitHub: first run 43 s total (pytest 14.7 s). After the bump to `actions/checkout@v7` and `astral-sh/setup-uv@v10.3.3` (clears the Node 20 warning), 199 passed, 22 skipped in 20.4 s.
- `gh` CLI is set up locally with a read-only fine-grained token (tiny-llm only; Actions and Contents read) to check runs.
- Next: the first README, including the badge, closes 6.7. Then make the repo public (v1-preview).

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
