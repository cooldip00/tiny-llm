# Prior Art & Positioning

Related projects we've studied: what they did, what we adopt, and how we differ.

---

## PA-1 — joe zhuo: LLM inference engine in C++/CUDA (LinkedIn, seen 2026-11-03)
**Link:** llm-inference-engine-jz.vercel.app (from the post; not independently verified)

### What he built
- **Inference only.** No training. Runs an existing open model (TinyLlama 1.1B).
- Whole runtime in **C++ and CUDA**, with his own kernels (matmul, attention, norms, ...).
- Serving features: paged KV cache, continuous batching, CUDA Graphs.
- Claimed result: **1,402 tok/s at batch 16** on a laptop RTX 5060, **3.4× HuggingFace**.
- About 5 weeks, starting from "could barely sum a vector on a GPU".

### His method (the most valuable part; adopted)
1. **Measure the hardware ceiling first** (he measured 323.6 GB/s) and judge every kernel against it.
2. **Keep slow kernels as reference answers.** A fast one replaces a slow one only after matching its output.
3. **Every number is reproducible** from a script in the repo.

→ Adopted in D-015 and the "Bandwidth ceiling method" and "Reference-answer rule" sections of [INFERENCE.md](INFERENCE.md).

### Comparison
| | PA-1 | tiny-llm (us) |
|---|---|---|
| Scope | Depth: inference systems only | Breadth: train → serve → research |
| Model | Pretrained TinyLlama 1.1B | Our own 14–30M from scratch, **plus** TinyLlama 1.1B for inference (D-015) |
| Level | C++/CUDA, all kernels hand-written | PyTorch, plus two timeboxed Triton kernels (D-016) |
| Role signal | GPU / performance engineer | ML engineer or researcher who understands the full stack |
| Novelty | Engineering speed | Research question: attention-efficiency ablation |
| Baseline | HF Transformers `generate` (weak serving baseline) | HF **and** vLLM / llama.cpp (stronger baselines) |

### Lessons that changed our plan
- **A 30M model gives misleading inference numbers.** About 60 MB of bf16 weights means decode is limited by kernel-launch overhead, not bandwidth. Hence the TinyLlama track (D-015).
- **Kernel-level work matters**, and Triton gets us some of it cheaply. Hence D-016.

### Options considered (2026-11-03)
| Option | Description | Outcome |
|---|---|---|
| A | Keep the plan + TinyLlama 1.1B inference track | Included in B |
| **B** | A + one or two Triton kernels measured against the bandwidth ceiling | **Chosen** (D-015, D-016) |
| C | Rewrite the engine in C++/CUDA | Rejected: weeks of work, mostly repeats PA-1 |

### Positioning for the writeup
- Don't compete on raw speed with C++/CUDA engines. Our story is **end to end**: trained our own model, served it and a 1.1B model with a hand-built engine, measured against hardware limits and production engines, then ran an attention ablation.
- Report against vLLM/llama.cpp, not only HF. It's a more credible comparison than PA-1's.
- Comparable hardware framing: his RTX 5060 Laptop measured 323.6 GB/s; ours is rated 448 GB/s (measure in Phase 0.5). Compare **% of ceiling**, not raw tok/s.
  - _Update (2026-11-04):_ Phase 0.5 measured **423 GB/s** read bandwidth on ours (see [HARDWARE.md](HARDWARE.md)). Use 423 GB/s as the ceiling, not the rated 448.

### Positioning update (2026-11-04, D-021)
- Training a small GPT from scratch is common (nanoGPT, llama2.c). Our differentiator is the **inference engine** measured against the bandwidth ceiling and against vLLM / llama.cpp.
- **Allocator story:** a paged KV cache (vLLM-style) is a block allocator problem: fixed-size blocks, fragmentation, allocation policy. It's the same family of problem as a PyTorch caching allocator (block splitting and merging to reduce fragmentation), which matches the author's prior experience. Lead the inference section of the writeup with 4.4 + 4.5 and this link.
- The Phase 5 ablation is listed as future work, not as a result (D-021).

---

## PA-2 — MIT CSAIL: Weight-Space Programming (blog post, published 2026-11-02, seen 2026-11-04)
**Links:** blog https://rikiyat.github.io/blog/weight-space-programming/ · logs https://github.com/RikiyaT/lm_understand-logs · edited model https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming · press: The Register, 2026-11-03.
**Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab. A blog post, not a peer-reviewed paper (no arXiv link as of 2026-11-04).

### What they built
- A frontier LLM (OpenAI's GPT-6 Astra) edits another network's weights **directly, with no gradient training**. Several experiments: writing small game-policy networks from scratch (e.g. Breakout), editing control networks, and editing a real LM.
- **LM experiment (Qwen3.5-0.8B):** Astra got a Python kernel over the weights (no GPU) and a budget of forward passes on practice problems (GSM8K / MATH training sets). It traced a repetition-loop behaviour to one direction written by the MLPs of layers 15 and 19, and subtracted it from 64 output columns in each: **131,072 weights, 0.016% of the model.**

| Qwen3.5-0.8B | MATH-500 | GSM8K | IFEval | Hit token limit (MATH-500) |
|---|---|---|---|---|
| Base | 7.9 | 13.8 | 21.0 | 93.0% |
| Edited, 10 forward passes | 12.3 | 25.7 | 49.0 | 87.2% |
| Edited, 25 forward passes | 14.1 | 25.1 | 39.9 | 86.4% |
| Edited, 50 forward passes | **48.1** | **67.1** | **62.3** | 33.1% |

### Honest reading
- Most of the "6×" gain comes from **removing loops**: the base model hit the 32K-token limit on 93% of problems. It's an obstacle removed, not new reasoning.
- On Gemma 3 270M (no loops), Astra **could not improve it** (6.8 → 7.0). Their words: editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked."
- Only one editor tested (GPT-6 Astra). No Claude, Gemini or open-model editors, so whether a frontier editor is required is **unknown**.
- Not reported: cost, tokens, hardware for the forward passes, variance or seeds, results on the six other benchmarks, an ablation isolating the loop fix. IFEval isn't monotonic in the budget (49.0 at 10 passes vs 39.9 at 25).
- Related known technique: removing a behaviour by subtracting one direction from the weights (e.g. refusal-direction ablation, 2024).

### Method worth adopting
- Same discipline as PA-1: a fixed budget of evaluations, every interaction logged and published, before/after numbers on standard benchmarks.

### Comparison / relevance to us
- Not part of the current scope (D-021). It's the basis for PLAN 7.2: replicate with a different editor (e.g. Claude) on Qwen3.5-0.8B, scaled down for 8 GB (~100 problems, 4K-token cap).
- Relevant to Phase 7 because tiny, targeted weight edits without retraining could sidestep catastrophic forgetting. That claim is **untested** in their post.

### Lessons that changed our plan
- None for Phases 4–6. Added to PARKING_LOT and PLAN Phase 7 (D-021).

---

## Template for future entries
```
## PA-n — <who/what> (<source>, seen <date>)
### What they built
### Method worth adopting
### Comparison
### Lessons that changed our plan (link D-xxx)
```
