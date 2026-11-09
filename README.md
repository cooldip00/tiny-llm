# tiny-llm

[![CI](https://github.com/cooldip00/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/cooldip00/tiny-llm/actions/workflows/ci.yml)

A Llama-style LLM trained from scratch on one laptop GPU (RTX 3070 Ti Laptop, 8 GB), and an inference engine for it, benchmarked against the GPU's memory-bandwidth ceiling. Plain PyTorch, no inference libraries; Hugging Face Transformers appears only as a test reference and baseline.

**Try it:** [live demo on Hugging Face Spaces](https://huggingface.co/spaces/CoolDip00/tiny-llm) ([status page](https://cooldip00-tiny-llm.hf.space/status)). It runs the 30M model on two private free-CPU Spaces behind our own public router Space, and continues short stories. It is a base model trained only on TinyStories, so it doesn't answer questions. The speed numbers below come from the laptop GPU, not the Spaces, except the one section that load-tests the Spaces. After two idle days the Spaces go to sleep, so the next visitor waits while they wake up.

**Why:** inference libraries hide where the time goes. Building each layer by hand, from training through the inference engine to the HTTP server and its router, makes each one measurable, and the bandwidth ceiling shows how much speed is left to find.

**Highlights**
<!-- bench:highlights -->
![Bar chart, TinyLlama 1.1B decode at batch 1: HF Transformers generate 44 tok/s; Ours, naive 35 tok/s; Ours, KV cache 54 tok/s; Ours, KV cache + CUDA graphs 137 tok/s; bandwidth ceiling 192 tok/s](docs/img/decode_b1.svg)

- **TinyLlama 1.1B decodes at 137 tok/s at batch 1: 71% of the bandwidth ceiling, 3.1× HF `generate`** (KV cache + CUDA graphs).
- **Our engine's bf16 logits match HF Transformers to within 0.34** on TinyLlama (same top token at 100.0% of prefill and 99.2% of cached-decode positions). The difference is our Triton RMSNorm (D-032): with the PyTorch norm the logits are bit-identical to HF's.
- **Serving:** continuous batching gives 1.37× static batching. A paged KV cache fits 25 sequences where contiguous slots fit 8, for 1.64× the throughput in the same memory.
<!-- /bench:highlights -->
- **Training:** a 30M-parameter model on ~1B TinyStories tokens in 2 h 27 min at **56% MFU**.
- **HTTP server:** OpenAI-compatible, with streaming. Past capacity it rejects requests with a 429, so time to first token stays bounded, where vLLM's and llama.cpp's grow with the queue ([vs vLLM and llama.cpp](#results-vs-vllm-and-llamacpp)).
- **Deployment:** the public demo is our own router in front of two replica Spaces on free CPUs. Two replicas hold 3 req/s within the latency targets where one holds 1, and peak served rises 1.85× to 2.00×, with no failed requests ([1 vs 2 replicas](#results-1-vs-2-replicas-on-hugging-faces-cpus)).

> Status: **v1-preview**. Still to come: CUDA graphs for the paged engine and an attention ablation (see [Roadmap](#roadmap)).

## Results: TinyLlama 1.1B inference

<!-- bench:tinyllama -->
Decode speed at batch 1 with a 128-token prompt, 256 new tokens, median of 5 runs. The ceiling is the measured bandwidth divided by the weight size: 423 GB/s ÷ 2,098 MiB = **192 tok/s per sequence**, because each decoded token has to read every weight once.

| Engine | tok/s | % of ceiling | ms / step |
|---|---|---|---|
| HF Transformers `generate` (bf16, SDPA) | 44 | 23% | 22.5 |
| Ours, naive (full forward per token) | 35 | 18% | 28.8 |
| Ours, KV cache | 54 | 28% | 18.6 |
| **Ours, KV cache + CUDA graphs** | **137** | **71%** | **7.3** |

- **Eager decode is host-bound.** At batch 1, HF and our eager engine take 22.5 and 18.6 ms per step (ours 19.8 ms at batch 32). The profiler shows the GPU idle 80% of an eager step, waiting for kernel launches. Capturing the decode step as a CUDA graph removes that overhead; the GPU is then idle 3% of the step.
- **Batching:** with graphs, 886 tok/s at batch 8 and 2,138 tok/s at batch 32 (HF: 357 and 1,466).
- **INT8 weights:** val loss +0.0010, 98.3% argmax agreement with bf16. Our Triton kernel reads the int8 weight and dequantizes inside the matmul; with CUDA graphs, decode is 1.47× bf16 at batch 1 and 1.05× at batch 32. Casting the weight back to bf16 on every call (plain PyTorch) gives 0.37× and 0.50×: it moves more bytes than bf16, not fewer.
- **Correctness:** bf16 logits differ from HF's bf16 logits by at most 0.34 over 415 test positions. The difference is our Triton RMSNorm (D-032): with the PyTorch norm the logits are bit-identical to HF's. Every top-1 disagreement with fp32 is a near-tie (margin ≤ 0.043 logits).

![Line chart, decode tok/s at batch 1, 8, 32: ours with CUDA graphs 137, 886, 2,138; HF Transformers 44, 357, 1,466; log scales, with the bandwidth ceiling times batch size](docs/img/batch_scaling.svg)
<!-- /bench:tinyllama -->

Full tables from the same run, with GPU clocks and temperatures: [docs/RESULTS.md](docs/RESULTS.md). The method, caveats and the dated results of each step: [docs/INFERENCE.md](docs/INFERENCE.md).

## Results: serving

<!-- bench:serving -->
64 requests of mixed lengths (9,004 tokens to generate), median of 3 runs.

| Scheduler / cache | Setting | tok/s | vs baseline |
|---|---|---|---|
| Static batching (TinyLlama) | 8 slots | 218 | — |
| Continuous batching (TinyLlama) | 8 slots | 299 | 1.37× |
| Continuous batching + CUDA graphs (TinyLlama) | 8 slots | 637 | 2.13× vs eager continuous |
| Contiguous KV slots (30M model) | 64 MiB of KV, 8 sequences | 638 | — |
| Paged KV cache (30M model) | same 64 MiB, 25 sequences | 1,049 | 1.64× |

The paged cache stores KV in fixed-size blocks through a per-sequence block table, as in vLLM, so a sequence holds only the blocks it has filled. The share of reserved KV holding real tokens goes from 52% to 97%.
<!-- /bench:serving -->

## Results: vs vLLM and llama.cpp

TinyLlama 1.1B in bf16. All three engines serve the same OpenAI-compatible API to the same load-test client (`scripts/load_test.py`): 14-token prompts, 128 new tokens per request, Poisson arrivals, prefix caching off. "Within SLO" is the highest request rate with TTFT p95 ≤ 1 s. Full table, method and caveats: [docs/SERVING.md](docs/SERVING.md).

| Engine | Seats | One stream, tok/s (% of ceiling) | Within SLO | Saturation | Past saturation |
|---|---|---|---|---|---|
| **Ours** (continuous batching + CUDA graphs) | 8 | 120 (62%) | 4 req/s | ~5.9 req/s, ~760 tok/s | sheds load with 429s; the 32-place queue caps TTFT p95 at ~5.6 s |
| **Ours** | 32 | (same engine) | 8 req/s | ≥ 13.0 req/s, 1,659 tok/s | 429s (15% at 16 req/s); TTFT p95 2.4 s |
| vLLM 0.33 | 8 | 162 (84%) | 8 req/s* | ~8.7 req/s, ~1,110 tok/s | queues without limit; TTFT p95 46 s at 16 req/s |
| vLLM 0.33, default | 256 | 161 (84%) | 32 req/s | ~34 req/s, ~4,410 tok/s | TTFT p95 20 s at 48 req/s |
| llama.cpp (bf16 GGUF) | 8 | 153 (80%) | 4 req/s | peaks at 5.5 req/s, then **falls** to 3.7 | TTFT p95 188 s at 16 req/s |

\* At the edge of the SLO: TTFT p95 0.95 s at 8 req/s (6 req/s within SLO in the 4.8a run, at 1.07 s).

![Two line charts, TinyLlama 1.1B with 8 sequences per engine, offered load 0.5 to 16 requests/s. Left, completed requests/s at 16 offered: vLLM 8.7; ours 5.7, with 43–60% of requests rejected with a 429 past 12 offered; llama.cpp 3.7, after a peak of 5.5. Right, time to first token p95 of served requests at 16 offered, on a log scale: llama.cpp 188 s, vLLM 46 s, ours 5.6 s, flat from 6 offered because a full queue answers new requests with a 429; SLO 1 s](docs/img/load_sweep.svg)

- **One stream:** vLLM is 1.35× faster and llama.cpp 1.28×. vLLM uses FlashAttention and fused kernels; we use PyTorch SDPA over padded slots and capture only the decode step as a graph.
- **Under load, seats set the capacity:** with 8 seats each we hold the same load as llama.cpp (4 req/s within SLO); vLLM holds 6–8 req/s, its rated load sitting on the 1 s line. vLLM's default admits 256 sequences into a paged KV pool, which is why it holds 32 req/s.
- **Past saturation:** we reject with 429s and keep latency bounded; vLLM and llama.cpp queue, and latency grows with the backlog.
- **Caveats:** one session on 2026-11-07, run straight after the tables above with the same build, with the GPU thermally throttled in every run (83–86 °C), so differences under ~15% are ties. Our one-stream number here (120 tok/s) goes through the HTTP server with streaming; the 137 tok/s above is the engine alone.

## Results: 1 vs 2 replicas on Hugging Face's CPUs

The deployed setup, not the laptop GPU: the 30M model on free CPU Spaces (2 vCPU and 8 slots per replica) behind the router, load-tested from the laptop with 128 new tokens per request and 60 s per rate, with the router's per-visitor limit off. The SLOs: TTFT p95 within 2 s, 99% availability, at most 1% of requests refused.

| Run | Replicas | Highest rate meeting every SLO | Peak served | Peak tok/s | Failed |
|---|---|---|---|---|---|
| A | 2 | 4 req/s | 4.61 req/s | 590 | 0 |
| B | 1 | 1 req/s | 2.31 req/s | 295 | 0 |
| C | 2 | 3 req/s | 4.27 req/s | 546 | 0 |

- **Two replicas hold 3 req/s, and one holds 1.** 4 req/s is borderline for two: met in run A, missed in run C at a TTFT p95 of 2.30 s.
- **Peak served is 1.85× to 2.00× one replica's.**
- **Overload stays clean:** every refusal was the router's own 429, and no request failed.
- **Caveats:** TTFT is measured on the laptop, so it includes the internet path, Hugging Face's proxy and the router hop. Each rate is one 60 s sample on shared hardware, and runs A and C show how much that moves. The rates tested were 1, 2, 3, 4 and 6 req/s, so the edges are coarse. Method and the full table: [docs/SERVING.md](docs/SERVING.md#results-1-vs-2-replicas-on-hfs-cpus-d-031-stage-3-2026-11-08).

## Results: training

| | |
|---|---|
| Model | 29.9M params: 8 layers, d_model 512, 8 heads, vocab 8,192 (our own BPE tokenizer) |
| Data | TinyStories, 996M tokens (~2.1 epochs) |
| Time | 2 h 27 min on the laptop GPU |
| Throughput | 115K tok/s, **56% MFU** (torch.compile: +57% over eager) |
| Final val loss | 1.2825 (train 1.280: no overfitting) |
| Export | Hugging Face Llama format (`scripts/export_hf.py`): HF Transformers' logits match ours to 1.4e-5, and the exported tokenizer gives identical ids on all 21,981 validation stories |

![Line chart: training and validation loss of the 30M model over 996M tokens, falling from 9.0 to a final validation loss of 1.2825, with train and validation loss close together throughout](docs/img/train_loss.svg)

The full run is public on Weights & Biases: [main-3.7](https://forge.coreweave.com/wandb/cooldip00-personal/tinyllm/runs/ugd124lz) has the loss, learning-rate, MFU and GPU charts, the config, and sample stories every 200 steps.

Sample after training, prompted with "Once upon a time": *"…a little girl named Lily. She was very curious… she found a shiny rock that looked like it was magic… she saw a butterfly that was also very shiny…"*

## How it works

The model block, the same layout as Llama. TinyLlama's weights load into it without any reshuffling ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

```mermaid
flowchart LR
    x([x]) --> n1[RMSNorm] --> attn["Attention<br/>RoPE, GQA"] --> a1((+))
    x --> a1
    a1 --> n2[RMSNorm] --> mlp["SwiGLU MLP"] --> a2((+))
    a1 --> a2
    %% Invisible boxes: keep the GitHub buttons off the last node
    a2 ~~~ gap0["<br/>"]
    style gap0 fill:none,stroke:none
    a2 --> out([out])
    a2 ~~~ gap["<br/><br/><br/><br/>"]
    style gap fill:none,stroke:none
```

A request's path through the serving engine:

```mermaid
flowchart LR
    req([Requests]) --> q[Queue]
    q --> sched["Scheduler<br/>continuous batching"]
    sched -->|new requests| prefill["Prefill<br/>batched, padded"]
    prefill --> kv[("KV cache<br/>slots or paged blocks")]
    sched -->|running sequences| decode["Decode step<br/>one token each"]
    kv <--> decode
    decode -->|finished, frees its KV| sched
    decode --> out([Tokens])
    %% Invisible box: keeps the GitHub zoom buttons off the last node
    decode ~~~ gap["<br/><br/><br/><br/>"]
    style gap fill:none,stroke:none
```

- **KV cache:** each step computes attention only for the new token and reads the cached keys and values of earlier ones.
- **Continuous batching:** a finished sequence leaves the batch at once, and a waiting request takes its place, instead of the whole batch waiting for its longest member.
- **Paged KV:** when memory runs out, the newest sequence is preempted and recomputed later (vLLM's default policy).
- **CUDA graphs:** the whole decode step is recorded once and replayed, so the CPU launches one graph instead of hundreds of kernels.
- **HTTP server:** the engine runs in its own thread behind a FastAPI app that serves an OpenAI-compatible `/v1/completions`, streamed or not. When its queue is full it answers 429 instead of slowing everyone down; `/health`, `/ready` and Prometheus `/metrics` report its state ([docs/SERVING.md](docs/SERVING.md)).

One server, and where each metric is measured. The clock starts when the HTTP app receives the request, so these times leave out the network; the load test's figures are measured at the client and include it:

```mermaid
flowchart LR
    client([Client]) <-->|"POST, then<br/>JSON or SSE"| app["HTTP app (FastAPI)<br/>Clock starts at arrival<br/>429 if the queue is full<br/>Request log,<br/>/stats, /metrics"]
    app --> queue["Queue<br/>Queue wait: arrival<br/>to first step"]
    queue --> thread["Engine thread<br/>TTFT, token gaps, latency:<br/>stamped after each step<br/>Every ~5 s: queue length,<br/>busy slots, tok/s, KV used"]
    thread <--> kv[("Slots and<br/>KV cache")]
    thread -->|tokens| app
    %% Invisible box: keeps the GitHub zoom buttons off the last node
    thread ~~~ gap["<br/><br/><br/><br/>"]
    style gap fill:none,stroke:none
```

The public demo is a router Space in front of two private replica Spaces, each running the server above. The replicas demonstrate routing; the demo's traffic doesn't need them:

```mermaid
flowchart LR
    v([Visitor]) -->|HTTPS| r["Public router Space<br/>shortest queue, one retry,<br/>rate limit, demo and status pages,<br/>SLOs visitors see"]
    r -->|HTTPS + token| a["Private replica Space 1<br/>30M model, 8 slots"]
    r -->|HTTPS + token| b["Private replica Space 2<br/>30M model, 8 slots"]
    r -.->|request logs,<br/>no prompts or IPs| d[("Private HF dataset")]
    %% Invisible box: keeps the GitHub zoom buttons off the last node
    r ~~~ gap["<br/><br/>"]
    style gap fill:none,stroke:none
```

## Repository layout

| Path | What's there |
|---|---|
| `src/tinyllm/model.py`, `attention/`, `config.py` | The transformer, swappable attention, model configs |
| `src/tinyllm/train.py`, `data.py`, `tokenizer.py`, `logger.py` | Training loop (bf16, torch.compile, MFU logging), data loading, BPE tokenizer, W&B logging |
| `src/tinyllm/generate.py`, `kv_cache.py` | Naive and KV-cached generation |
| `src/tinyllm/engine.py`, `paged_kv.py` | Static and continuous batching; paged KV cache with a block allocator |
| `src/tinyllm/cuda_graph.py` | Decode step captured as a CUDA graph |
| `src/tinyllm/quant.py` | INT8 weight-only quantization |
| `src/tinyllm/kernels/` | Triton kernels: INT8 matmul for decode; RMSNorm fused with the residual add (the default on GPU; training and CPU use PyTorch) |
| `src/tinyllm/hf_llama.py`, `safetensors_io.py`, `logit_compare.py` | Loads Llama-format checkpoints (TinyLlama) and exports to them; compares logits with a reference model |
| `src/tinyllm/server.py`, `router.py`, `web_common.py`, `obs.py`, `logship.py` | OpenAI-compatible HTTP server (streaming, 429s past capacity, timeouts); our router in front of the replica Spaces (runs the public demo); JSON logs and Prometheus metrics; log upload to a private HF dataset (no text, no IP) |
| `Dockerfile`, `Dockerfile.router`, `space/`, `hf/` | CPU image of the server; image of the router; the Spaces' cards; the Hugging Face model card |
| `src/tinyllm/provenance.py`, `results.py`, `plots.py` | Records where each benchmark number came from (commit, versions, GPU clocks); renders the results tables and plots |
| `scripts/` | Data, tokenizer, benchmark, load-test (`load_test.py`), HF export (`export_hf.py`), Space deployment (`push_space.py`) and usage-report (`analyze_logs.py`) scripts |
| `tests/` | 640 tests: every optimized path is checked against the slow reference path |
| `docs/` | Write-ups with the method and dated results of each step: [ARCHITECTURE](docs/ARCHITECTURE.md), [INFERENCE](docs/INFERENCE.md), [SERVING](docs/SERVING.md). Generated tables: [RESULTS](docs/RESULTS.md). Also [HARDWARE](docs/HARDWARE.md), [SETUP](docs/SETUP.md), [RESEARCH](docs/RESEARCH.md) (the attention ablation plan) and [PRIOR_ART](docs/PRIOR_ART.md) (related projects). `docs/img/` holds the plots, drawn by `scripts/bench_all.py` from the runs and `docs/data/` |

## Reproduce

Linux or WSL2 with an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/). Full setup, step by step: [docs/SETUP.md](docs/SETUP.md).

```bash
git clone https://github.com/cooldip00/tiny-llm.git && cd tiny-llm
uv sync                                          # exact versions from uv.lock (PyTorch 2.14.1, CUDA 13.0)
uv run pytest -q                                 # CUDA tests skip without a GPU

# Train the 30M model
uv run python scripts/download_data.py           # TinyStories
uv run python scripts/train_tokenizer.py         # BPE, 8,192 tokens
uv run python scripts/tokenize_data.py           # data/train.bin, data/val.bin
uv run python -m tinyllm.train --preset main --max_steps 7600

# Benchmark TinyLlama (weights download to the Hugging Face cache)
uv run python scripts/tinyllama_smoke.py
uv run python scripts/bench_generate.py --model tinyllama --engine kv-graph
uv run python scripts/bench_serving.py --model tinyllama
uv run python scripts/bench_quant.py --model tinyllama

# Serve TinyLlama over HTTP, then load-test it from a second terminal
uv run python -m tinyllm.server --model tinyllama --cuda-graph
uv run python scripts/load_test.py --rates 1,2,4 --power-mode performance

# Serve the 30M model on CPU with Docker (no GPU needed; the weights download from the Hugging Face Hub)
docker build -t tinyllm --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
docker run --rm -p 7860:7860 tinyllm
curl localhost:7860/v1/completions -H 'content-type: application/json' -d '{"prompt": "Once upon a time", "max_tokens": 60}'
# In a browser: localhost:7860 is the demo, localhost:7860/status the status page (live charts, SLOs, error budget)
# Request logs to a private HF dataset (-e LOG_DATASET=you/logs, needs HF_TOKEN), then a usage report from them
uv run python scripts/analyze_logs.py --dataset you/logs --days 7

# Route between two CPU servers (uses the checkpoint from training above); localhost:8000 is then the demo, /status the status page
uv run python -m tinyllm.server --device cpu --ckpt checkpoints/main/best.pt --trust-router --port 8001 &
uv run python -m tinyllm.server --device cpu --ckpt checkpoints/main/best.pt --trust-router --port 8002 &
uv run python -m tinyllm.router --replica r1=http://127.0.0.1:8001 --replica r2=http://127.0.0.1:8002
# Deploy to Spaces: scripts/push_space.py --role server|router|replica (dry run by default; see docs/SERVING.md)

# Every results table: rerun all benchmarks (~1.5–2 h), then render docs/RESULTS.md and the README tables
uv run python scripts/bench_all.py --run --power-mode performance
uv run python scripts/bench_all.py               # re-render from the saved JSON, no GPU
```

CI runs the CPU test suite on every push, and builds the Docker image and checks that it serves. The image is CPU-only (PyTorch code paths, no Triton or CUDA graphs), so it is much slower than the numbers above. GPU benchmark numbers come from the laptop. The benchmark scripts save each result with the commit, library versions, and the GPU clocks and throttling seen during the run.

## Roadmap

- CUDA graphs for the paged engine (the continuous engine has them)
- Attention ablation (MHA vs GQA vs MLA vs a Gated DeltaNet hybrid) at a fixed compute budget ([docs/RESEARCH.md](docs/RESEARCH.md))

## License

[MIT](LICENSE)
