# Serving: production layer, observability, release strategies (design)

Design for step 4.13 (server), 6.5b (Hugging Face Space) and 6.9 (release strategies), agreed on 2026-11-06 (D-028).
This is a plan, not a record of what is built. PLAN.md ticks each part when it lands.

## Constraints

- One laptop GPU (local server, TinyLlama 1.1B) and one free CPU Hugging Face Space (our 30M model).
- No multiple servers, so no load balancer, no Prometheus / Grafana deployment, no distributed tracing.
- We build the parts a single service needs, using the real formats where they exist, and describe the multi-server design in writing.

## What is deployed where

| Target | Model | Purpose |
|---|---|---|
| Local server in WSL (`localhost:8000`) | TinyLlama 1.1B Chat (bf16, CUDA graphs) | Load tests, the 4.8a comparison with vLLM / llama.cpp (same API, same client) |
| Public HF Space (free CPU) | Our 30M model | Public demo, shows the full pipeline (our training, our engine) |

- The server takes `--model tinyllama` or `--ckpt <path>`, so it's one codebase for both.
- The 30M model is a **base model trained only on TinyStories**, with no instruction tuning or RLHF. It continues stories and does not answer questions or follow instructions, so the API is `/v1/completions` (text in, continuation out), not chat. TinyLlama is chat-tuned: clients put its chat template into the prompt. The demo presents the 30M model as story continuation, with example openings.
- The laptop is never exposed through a tunnel. The Space is the public endpoint.

## 4.13 step 2: the server

```
client ──HTTP──▶ FastAPI handlers ──▶ Router (1 variant now) ──▶ engine thread: ContinuousEngine.step() loop
   ◀── JSON / SSE stream ◀── per-request asyncio queue ◀── Events (rid, token, done)
```

- **Engine thread:** one thread owns the `ContinuousEngine` and calls `step()` in a loop, so the engine stays single-threaded. Handlers submit requests through a thread-safe inbox, and tokens come back to each request's asyncio queue (`call_soon_threadsafe`).
- **API:** `POST /v1/completions` (OpenAI-compatible subset):
  - `prompt`: text or token ids
  - `max_tokens`, `temperature`, `top_p`, `top_k`, `ignore_eos`
  - `stream`: SSE chunks ending with `data: [DONE]`
  - responses include `finish_reason` (`stop` = EOS, `length` = max_tokens) and `usage` (prompt / completion tokens)
- **Per-request sampling:** each request keeps its own settings. Rows with the same settings are sampled together (usually one group), so greedy requests are unaffected.
- **Hardening:**
  - `/health` (process alive) and `/ready` (model loaded, graphs captured)
  - a bounded queue: when it's full, new requests get **429** with `Retry-After` instead of everyone slowing down
  - a per-client rate limit (token bucket on the hashed client id) → 429, so one visitor can't hold every slot
  - a per-request timeout, after which the request is aborted and its slot freed
  - a client disconnect → `engine.abort(rid)`
  - graceful shutdown: stop accepting, finish in-flight requests (with a deadline), then exit
  - an exception in `step()` fails the in-flight requests with **500** plus the request id and is logged with a stack trace. The server then reports not ready instead of hanging.
- **Router hook:** handlers talk to a `Router`, not to the engine. It has one variant now. Every response carries `X-Model-Version` (checkpoint hash + git commit). Release strategies (6.9) then only add routing rules.

## Observability (4.13 step 2 + 6.5b)

| # | Piece | What | Production analogue | Where |
|---|---|---|---|---|
| 1 | Request log | One JSON line per request: request id, variant / model version, arrival, queue wait, TTFT, inter-token latency p50 / p95, total latency, prompt / completion tokens, finish reason, status / error, hashed client id, queue depth and busy slots at arrival | Structured logging (ELK, Datadog) | 4.13 |
| 2 | Request id | Generated per request, returned as `X-Request-ID`, in every log line, so one request's timeline can be found from a user report | Tracing, single-service version | 4.13 |
| 3 | Engine snapshots | Every ~5 s: queue length, busy slots, tok/s, KV used, RAM / VRAM | Infrastructure metrics | 4.13 |
| 4 | `/stats`, `/metrics` | `/stats`: JSON over rolling windows (5 min, 1 h) with request count, latency percentiles, error and reject rates. `/metrics`: the same in Prometheus text format, so a real Prometheus could scrape it unchanged | Metrics endpoint | 4.13 |
| 5 | Debug mode | `--log-level debug`: per-step timings (prefill ms, decode ms, admissions, preemptions). Requests slower than the SLO always log their full timeline. `/debug/state` (localhost only) dumps the queue and slots | Debug logs, slow-request log | 4.13 |
| 6 | Status page | `GET /status`: an HTML page served by our server, with live charts from `/stats/history`: requests / min, latency over time, queue depth, SLO compliance and error budget (see below) | Grafana dashboard | 6.5b |
| 7 | SLO | e.g. 95% of requests get a first token within 2 s on the Space. The status page shows the share meeting it | SLOs / error budgets | 6.5b |
| 8 | Log retention | The free Space's disk is wiped on restart, so the logs go to a **private** HF dataset: `--log-dir DIR --log-dataset REPO_ID` (image: `-e LOG_DATASET=…`) writes hourly JSONL and uploads it every 10 min with `huggingface_hub.CommitScheduler`, plus once in the graceful drain; kept 30 days (`src/tinyllm/logship.py`, below) | Log shipping and retention | 6.5b step 4 |
| 9 | Usage report | `scripts/analyze_logs.py --dataset REPO_ID [--days N] [--json]`: restarts and uptime, requests / day by status, visitors / day, TTFT and latency p50 / p95 / p99, tokens, finish reasons, SLO compliance and error budget over the period, errors grouped, busiest hours (UTC), queue and slot use | Daily ops report | 6.5b step 4 |

### Log shipping (6.5b step 4)

- **Files:** `logs/<YYYY-MM-DD>/<start>-<replica>-<HH>.jsonl` in the dataset (UTC). One file per server start per hour, so restarts never overwrite each other and an upload only touches this run's files. `<replica>` is the Space name (`SPACE_ID`), so several replicas (D-031) can share one dataset.
- **Shipped lines:** `request` and `slow request` (with its timeline), engine snapshots thinned to one per 30 s (the status page keeps the 5 s ones in memory), `model loaded`, `server ready` (with the TTFT threshold), `shutting down: draining`, `stopped`, and errors **without** their traceback. Never `--log-level debug` lines. stderr keeps the full log as before.
- **Failures never touch serving.** No token, no access or no network: one warning `log upload off: logs stay local`. A failed upload is logged to stderr by `huggingface_hub` and retried with the next one.
- **Loss:** a graceful stop (SIGTERM: the Space restarting or rebuilding) uploads once more after the drain. A hard kill loses up to the last 10 minutes.
- **Retention:** once a day (first at start) the uploader deletes dataset files older than 30 days in one commit, and squashes the dataset's history (`HfApi.super_squash_history`) when its oldest commit is older than 30 days, so deleted files stop using storage. It runs on the uploader's thread, so it never overlaps an upload.
- **Token:** `HF_TOKEN`, a Space secret set by the owner: a fine-grained token with write access to the log dataset only. Locally, the `hf auth login` token.
- **Size (estimate, 2026-11-07; measured on the real dataset: request 574 B, snapshot 278 B):** a request line is ~600 B, a snapshot ~250 B: ~0.7 MB / day of snapshots plus ~0.6 MB per 1,000 requests. 30 days ≈ 40 MB at light traffic, ~2 GB if very busy, well under the 100 GB private storage.

### Privacy rules

- We log metadata only (timings, token counts, status, finish reason, error message), **never prompt or completion text**. Error messages name the field and the limit, never the input (`tests/test_logs.py` sends a marker string through every path: 200, stream, 400 validation, 400 too long, 429, 504 and a client disconnect, and checks it never reaches the shipped files).
- Visitors are identified by a salted hash of the IP (`TINYLLM_LOG_SALT`, a Space secret). Raw IPs are never stored. The salt must stay secret: there are few enough IPv4 addresses to try them all against a known salt. With the salt set, a visitor hashes the same across restarts, so visitors can be counted over days; without it, the salt is random per start.
- The hash is pseudonymous, not anonymous: it tells repeat visits apart. The log dataset is private and kept 30 days.
- The demo page states what is logged ("What is logged" in its footer).

## Load testing (4.13 step 3)

- Poisson arrivals at a set of request rates, using the same client against our server and vLLM (4.8a).
- Reports:
  - TTFT and inter-token latency at p50 / p99
  - throughput against request rate
  - **goodput**: the highest rate that still meets the latency target
  - behaviour past capacity (429s, queue wait)

### Results: TinyLlama on three engines, rerun (4.10c, 2026-11-07)

The 4.8a sweep below, rerun with the same client, prompts, settings, rates and engine builds, after RMSNorm switched to the Triton kernel (D-032), in the same session as the `runs/bench/v1b` tables. One engine on the GPU at a time, cooled to 60 °C before each. NitroSense Performance, fans Max, room fan overhead. Provenance: `runs/bench/load/4.10c/` (commit `3928889`); the plot data `docs/data/load_sweep.json` comes from it.

| Setup | Seats | B=1 decode (% of ceiling) | Rated load | Saturation | ITL p50 / p99 at 4 req/s | Past saturation |
|---|---|---|---|---|---|---|
| ours, `--slots 8 --cuda-graph` | 8 | 119.6 tok/s (62.2%) | 4 req/s | ~5.9 req/s, ~760 tok/s | 9 / 37 ms | 429s (43% at 12, 60% at 16); TTFT p95 capped at ~5.6 s by the 32-place queue |
| ours, `--slots 32 --cuda-graph` | 32 | (same engine) | 8 req/s | ≥ 13.0 req/s, 1,659 tok/s at 16 | 12.5 / 42 ms | 429s (15% at 16); TTFT p95 1.2 s at 12, 2.4 s at 16 |
| vLLM 0.33, `--max-num-seqs 8` | 8 | 161.8 tok/s (84.1%) | 8 req/s (TTFT p95 0.95 s) | ~8.7 req/s, ~1,110 tok/s | 7 / 14 ms | no 429s; unbounded queue, TTFT p95 46 s at 16 req/s |
| vLLM 0.33, default (256 seqs) | 256 | 161.0 tok/s (83.7%) | 32 req/s | ~34 req/s, ~4,410 tok/s | 7 / 9 ms | TTFT p95 20 s at 48 req/s |
| llama.cpp `61d4472`, bf16 GGUF, `-np 8` | 8 | 153.4 tok/s (79.8%) | 4 req/s | peak 5.5 req/s at 6, then **falls** to 3.7 at 16 | 10 / 29 ms | no 429s; TTFT p95 188 s at 16 req/s |
| llama.cpp, Q8_0 GGUF, `-np 8` | 8 | 236.2 tok/s (65.3% of 361.6) | 4 req/s | peak 6.5 req/s at 8, then falls to 4.3 at 16 | 9 / 27 ms | no 429s; TTFT p95 158 s at 16 req/s |

What changed from 4.8a:
- **Our one stream: 111.7 → 119.6 tok/s (+7%)**, the D-032 gain through the HTTP server. vLLM's lead goes from 1.44× to 1.35× and llama.cpp's from 1.35× to 1.28×; the other engines moved by under 2%.
- **Rated loads are the same except vLLM at 8 seqs, 6 → 8 req/s.** That is the SLO edge, not a faster vLLM: its TTFT p95 at 8 req/s was 1.07 s in 4.8a and 0.95 s here, against a 1 s target, and its saturation moved by 2% (8.5 → 8.7 req/s). Read it as "6–8 req/s".
- Our saturation with 8 slots rose ~7% (5.5 → 5.9 req/s), in line with the faster step; with 32 slots, 12.2 → 13.0 req/s.
- **Caveat:** thermally throttled again (GPU max 83–86 °C, SM clock medians 1,575–1,875 MHz; one sweep, ours at 32 slots, also logged a hardware slowdown flag). Differences within ~15% are ties, so the comparison is unchanged: vLLM is faster per stream, seats set the capacity, and we bound latency with 429s.

### Results: TinyLlama on three engines (4.8a, 2026-11-06)

Superseded by the 4.10c rerun above (kept as the pre-D-032 record).

Same client (`scripts/load_test.py`), same 8 story prompts (14 tokens), `ignore_eos`, bf16 weights unless noted, NitroSense Performance mode. Prefix caching is off: vLLM `--no-enable-prefix-caching`, llama.cpp `--no-cache-prompt`; we don't have it.
- **B=1:** 20 requests × 256 tokens, one at a time. Decode speed excludes the first token (prefill). Ceiling = 423 GB/s ÷ weight bytes: 192.3 tok/s for bf16, 361.6 tok/s for Q8_0.
- **Sweep:** Poisson arrivals, 128 tokens per request, 60 s per rate, local SLOs (TTFT p95 ≤ 1 s).

| Setup | Seats | B=1 decode (% of ceiling) | Rated load | Saturation | ITL p50 / p99 at 4 req/s | Past saturation |
|---|---|---|---|---|---|---|
| ours, `--slots 8 --cuda-graph` | 8 | 111.7 tok/s (58.1%) | 4 req/s | ~5.5 req/s, ~710 tok/s | 9 / 41 ms | 429s; TTFT p95 capped at ~5.9 s by the 32-place queue |
| ours, `--slots 32 --cuda-graph` | 32 | (same engine) | 8 req/s | ≥ 12.2 req/s, 1,555 tok/s at 16 | 13 / 46 ms | 429s; TTFT p95 2.6–2.9 s |
| vLLM 0.33, `--max-num-seqs 8` | 8 | 160.7 tok/s (83.5%) | 6 req/s | ~8.5 req/s, ~1,090 tok/s | 7 / 15 ms | no 429s; unbounded queue, TTFT p95 48 s at 16 req/s |
| vLLM 0.33, default (256 seqs) | 256 | 160.9 tok/s (83.7%) | 32 req/s | ~34 req/s, ~4,370 tok/s | 7 / 9 ms | TTFT p95 21 s at 48 req/s |
| llama.cpp `61d4472`, bf16 GGUF, `-np 8` | 8 | 151.0 tok/s (78.5%) | 4 req/s | peak 5.4 req/s at 6, then **falls** to 3.5 at 16 | 10 / 30 ms | no 429s; TTFT p95 203 s at 16 req/s |
| llama.cpp, Q8_0 GGUF, `-np 8` | 8 | 233.9 tok/s (64.7% of 361.6) | 4 req/s | peak 6.2 req/s, then falls to 3.8 at 16 | 9 / 29 ms | no 429s; TTFT p95 182 s at 16 req/s |

What it shows:
- **One stream (B=1) is kernel speed.** vLLM reaches 84% of the bandwidth ceiling, we reach 58%. vLLM uses FlashAttention and fused kernels; we use SDPA over padded slots and graph only the decode step.
- **Under load, seats set the capacity.** By Little's law, saturation ≈ seats ÷ time per request: 8 ÷ ~1.4 s ≈ 5.7 req/s for us, 8 ÷ ~0.9 s ≈ 8.9 for vLLM at 8 seqs. Going from 8 to 32 slots doubled our rated load. vLLM's default admits up to 256 sequences into a paged KV pool (70% of VRAM), so it is limited by compute, not by seats.
- **Past saturation the engines differ by design.** We shed with 429s and keep TTFT bounded. vLLM queues without limit, so TTFT grows with the backlog. llama.cpp's throughput *drops* under overload (cause not investigated; our client handled 947 streams against vLLM in the same run).
- At equal seats we tie llama.cpp bf16 under load (rated 4 req/s each, ~690 tok/s saturation), even though it is 1.35× faster at B=1.
- **Caveat:** every run was thermally throttled (GPU 84–87 °C, SM clock medians 1,515–1,770 MHz; provenance in `runs/bench/load/4.8a/`). Treat differences within ~15% as ties. The 1.3–1.5× gaps above are well outside that.

## SLOs (4.13 step 3, written before the load test)

Targets are set **before** measuring. If the load test shows a target can't be met at any useful rate, changing it needs a DECISIONS entry, so the goalposts don't move quietly.

All three are measured on the server, from the request log and `/stats` (arrival at the server → token sent; network time excluded). The load generator also measures them on the client, as a cross-check.

| # | SLO | Target: local (TinyLlama, GPU) | Target: Space (30M, free CPU) | Source |
|---|---|---|---|---|
| 1 | **TTFT p95** (successful requests; includes queue wait) | ≤ 1 s | ≤ 2 s | `ttft_s` in the request log; `/stats` → `ttft_s.p95` |
| 2 | **Availability**: share of requests not failed by the server (status ≥ 500, including 504 timeouts) | ≥ 99% | ≥ 99% | `/stats` → `1 − error_rate` |
| 3 | **Shed rate at rated load**: share of 429s while the offered rate is at or below the rated load | ≤ 1% | ≤ 1% | `/stats` → `reject_rate` |

- **Rated load** = the goodput from the load test: the highest Poisson rate at which SLO 1 still holds. SLOs 2 and 3 are judged up to that rate. Above it, 429s are the intended behaviour (shed load instead of making everyone slow), and only SLO 2 still applies.
- Client errors (400) and client disconnects (499) count for none of the three.
- **Error budget** = 1 − target over the window. With 99% availability that's 1 failed request in 100. For example, 1,000 requests in a window may include 10 server failures.
  - Local: the window is one load-test run. A run that spends more than its budget fails the SLO, and the report says which one.
  - Space (6.5b): a rolling 7-day window on the status page, showing the budget left for each SLO. **Burn-rate rule** (design, Google SRE style): flag when the 1 h error rate is above 14.4× the budgeted rate. That's Google's fast-burn threshold, which spends 2% of a 30-day budget in an hour; on our 7-day window it's 14.4 / 168 ≈ 9% of the budget in an hour.
- What uses the SLOs:
  - the slow-request timeline log (`--slow-s`, set to the TTFT target in practice)
  - goodput in the load test
  - the canary's automatic rollback in 6.9: roll back when the canary breaks an SLO that live meets

### Demo and status pages (6.5b step 3)

Both pages are static HTML + JS in `src/tinyllm/web/`, served by the server itself, with no CDN and relative URLs (so the D-031 router can serve them in front of the replicas).

- `GET /`: the demo. Story continuation streamed from `/v1/completions` (`stream: true`), with length and temperature sliders, a Stop button (the server sees a disconnect: 499, the slot is freed), TTFT and tokens/s, and plain messages for 429 / 503.
- `GET /status`: the status page. It polls `GET /stats/history` every 5 s:
  - the last 60 minutes per minute: requests (served / 429 / 5xx), TTFT p50 / p95 and latency p95 against the SLO line
  - engine snapshots from the last hour: busy slots and queue length
  - one card per SLO: compliance, Met / Missed, error budget left, 1 h burn rate with the fast-burn flag
- `/stats/history` is computed at most once a second, however many pages poll it. Page and stats requests are not counted in the request metrics or the SLOs.
- SLO maths (`Metrics.history`, tested in `tests/test_status.py`):
  - TTFT: bad = a served request (200) whose TTFT is above `--slo-ttft-s` (1 s by default; the Docker image sets 2 s); total = served requests
  - availability: bad = status ≥ 500; total = 200 + 429 + ≥ 500
  - shed: bad = 429, same total. Shown but **not judged**: the rated load of the Space is unknown until the D-031 load test, and above it 429s are intended.
  - budget left = 1 − bad / ((1 − target) × total). It goes negative once the budget is spent (the page shows "used up").
  - burn rate = (bad / total over the last hour) / (1 − target). Flagged above 14.4× when the hour has ≥ 10 counted requests.
- **Limit: the 7-day window starts at the last restart.** Counters are kept in memory per hour, and a free Space restarts often (sleep, rebuilds), so in practice the window is "since the last restart". The page says so. Restoring the full window from the logs dataset (step 4) is parked.

## Prefix caching (not implemented; noted 2026-11-06 in 4.8a)

**What it is.** Before generating, the server runs the whole prompt through the model once (prefill) and keeps the result for each prompt token in the KV cache. Normally that cache is thrown away when the request ends. With prefix caching, the server keeps it, indexed by the prompt's tokens. When a later request starts with the same tokens, it reuses that KV and only prefills the new part.

Example: 100 chat users share the same 500-token system prompt. Without prefix caching, every request prefills 500 + N tokens. With it, the first request prefills 500 + N and later ones prefill only their own N. TTFT and GPU work drop a lot.

**Where we stand.** We don't have it: every request prefills its full prompt. vLLM (on by default in V1) and llama.cpp (per-slot prompt reuse) both do. The 4.8a API check saw llama.cpp report 13 of 14 prompt tokens cached on a repeated prompt.

**Why it matters little for us now:**
- The load-test prompts are ~15 tokens, so there is little to reuse. For the matched 4.8a comparison it is turned off in the other engines where they allow it (vLLM `--no-enable-prefix-caching`), and each run records the setting.
- The Space demo is story continuation with short openings and no long shared system prompt.
- It matters for chat with long system prompts, multi-turn conversations (each turn resends the history), RAG instructions and few-shot prompts.

**Two ways to add it:**

| | A. Reuse within a slot (llama.cpp style) | B. Shared blocks (vLLM style) |
|---|---|---|
| How | When a slot frees up, keep its KV. If the next request in that slot starts with the same tokens, prefill only the rest. | KV in fixed-size blocks, indexed by a hash of the prefix. Any request can share any cached block; refcounts track users, and old blocks are evicted when memory runs short. |
| Builds on | The slot cache (works with CUDA graphs) | The paged cache from 4.5 (no CUDA graphs yet, so slower today) |
| Hit rate | Only if the matching request lands in the same slot | Any request, any slot; many requests share one copy |
| Estimate | ~3–4 h | ~6–8 h, plus ~3–4 h of CUDA graphs for paged KV to keep it fast |

Common to both:
- **Gains:** large TTFT and compute savings on long shared prefixes. Why it's correct: the reused tokens keep the same positions, so their RoPE values are still valid.
- **Costs:**
  - The engine's padded prefill would have to start each row at a different offset (the cached length). This is the main work.
  - Cached KV takes memory and needs an eviction policy.
  - New correctness risk: served output must still equal `generate()` exactly.
  - Privacy side channel: a fast TTFT reveals that someone recently sent the same prefix. vLLM adds a per-user "cache salt" to the block hash to prevent this; we would need the same.

**Decision (2026-11-06):** not before v1; it adds about nothing for short-prompt workloads. A is the right size for v1.1. B only together with CUDA graphs for the paged cache (Phase 7). Parked in PARKING_LOT (the 4.5 entry, item d). With several replicas, prefix-aware routing (below) is what makes it pay off.

## Scaling beyond one server (design only, written up in 6.5b)

- **Replicas:** the server keeps no state between requests, so N copies can run behind a load balancer.
- **Routing:**
  - least-loaded (by queue depth from `/metrics`)
  - prefix-aware, so requests sharing a prompt prefix land where its KV is cached
- **Autoscaling:** on queue depth or the SLO burn rate, not CPU %.
- **Capacity:** one replica's goodput from the load test × target headroom → replica count.

## Release strategies (6.9, v1.1, after v1)

Variants we already have: our model bf16 vs INT8 (4.6), two checkpoints, eager vs CUDA graphs, slot vs paged KV.

| Strategy | Build | Shown for real? |
|---|---|---|
| Blue-green | Two engines loaded; a localhost-only admin call switches the live one; in-flight requests finish on the old one; instant rollback | Yes. Two 30M models fit easily. Two TinyLlama copies plus caches may be tight on the laptop's VRAM |
| Shadow | Mirror a sampled share of traffic (10–20%) to the shadow engine and log both outputs and timings; a report gives token agreement and the latency difference. The shadow runs only when the live queue is empty, and its effect on live latency is measured (on one machine it competes for compute) | Yes |
| Canary | A weighted router (5% → 25% → 100%); a watcher compares the canary's error rate and SLO against live and **rolls back automatically**. Demonstrated with the load generator and a deliberately degraded variant | Yes |
| A/B test | Sticky assignment by hashed client id; thumbs up / down in the demo; a significance test and confidence intervals | The mechanism, yes. A real verdict, no: a personal Space gets too few visitors. Validated with simulated users from the load generator; the report states the sample size a real verdict would need |
| Versioning / rollback | Variant id = checkpoint hash + git commit, in `X-Model-Version` and every log line | Yes |

**Promotion pipeline:** offline gate (tests, val loss within a threshold, benchmark) → shadow (agreement, latency) → canary 5% → 25% → 100% with auto-rollback → live. A/B is used when the question is "is it better for users", not "is it safe".

Estimate ~9–10 h:
- router with blue-green: ~2 h
- shadow and its report: ~2–3 h
- canary with rollback: ~2 h
- A/B with feedback and statistics: ~3 h
