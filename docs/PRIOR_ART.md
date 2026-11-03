# Prior Art & Positioning

Related projects we've studied: what they did, what we adopt, and how we differ.
Use this in Phase 6 to position the writeup and resume bullets.

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
- **Kernel-level skill is a strong resume signal**, which we can get cheaply through Triton. Hence D-016.

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

---

## Template for future entries
```
## PA-n — <who/what> (<source>, seen <date>)
### What they built
### Method worth adopting
### Comparison
### Lessons that changed our plan (link D-xxx)
```
