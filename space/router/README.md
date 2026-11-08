---
title: tiny-llm
emoji: 📖
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
license: mit
short_description: A 30M-parameter story model, served by a from-scratch engine
pinned: false
---

# tiny-llm

A 30M-parameter language model trained from scratch on TinyStories, served by the project's own inference server (continuous batching, streaming, load shedding). This Space is a small router written for the project, in front of two private replicas on HF's free 2-vCPU CPU tier. It is a learning demo of routing, not a way around paying for hardware.

- **Demo:** this page. Type the start of a story and it carries on.
- **Status:** `/status` shows live latency, throughput, the SLOs (time to first token p95 ≤ 2 s, 99% availability) and each replica's state.
- **API:** `POST /v1/completions` (OpenAI-style), e.g.
  `curl https://cooldip00-tiny-llm.hf.space/v1/completions -H 'content-type: application/json' -d '{"prompt": "Once upon a time", "max_tokens": 60}'`
- **Code, design notes and benchmarks:** https://github.com/cooldip00/tiny-llm (this Space runs `Dockerfile.router` from the repo; the commit the replicas run is shown on `/stats`).
- **Model:** https://huggingface.co/CoolDip00/tiny-llm-tinystories-30m

## Limits

It is a small model on shared CPUs: stories are simple and sometimes odd. Each visitor may send 20 requests per minute (bursts of 5); when both replicas are full the answer is 429 at once instead of making everyone wait. The Spaces sleep after 48 hours without visitors, and nothing keeps them awake: the first visit after that gets a "waking up" message, and a retry a minute or so later works.

## What is logged

Request metadata only: timings, token counts, status and finish reason. **Never the prompt or the generated text.** Visitors are counted by a salted hash of the IP address; the IP itself is never stored. The logs go to a private dataset and are kept for 30 days. Details: [docs/SERVING.md](https://github.com/cooldip00/tiny-llm/blob/main/docs/SERVING.md#privacy-rules).
