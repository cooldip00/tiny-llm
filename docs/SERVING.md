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
| 6 | Status page | An HTML page served by our server, with live charts from `/stats`: requests / min, latency over time, queue depth, SLO compliance | Grafana dashboard | 6.5b |
| 7 | SLO | e.g. 95% of requests get a first token within 2 s on the Space. The status page shows the share meeting it | SLOs / error budgets | 6.5b |
| 8 | Log retention | The free Space's disk is wiped on restart, so JSONL logs are uploaded to a **private** HF dataset every ~10 min (`huggingface_hub.CommitScheduler`, write token as a Space secret) | Log shipping and retention | 6.5b |
| 9 | Usage report | `scripts/analyze_logs.py`: visitors, requests, latency percentiles, errors, busiest hours | Daily ops report | 6.5b |

### Privacy rules

- By default we log metadata only (timings, token counts, status), **never prompt or completion text**.
- Visitors are identified by a salted hash of the IP (salt kept as a Space secret). Raw IPs are never stored.
- The log dataset is private. The Space page states what is logged.

## Load testing (4.13 step 3)

- Poisson arrivals at a set of request rates, using the same client against our server and vLLM (4.8a).
- Reports:
  - TTFT and inter-token latency at p50 / p99
  - throughput against request rate
  - **goodput**: the highest rate that still meets the latency target
  - behaviour past capacity (429s, queue wait)

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
