# Research: Efficient Attention Ablation

> **Status: deferred (D-021, 2026-11-04).** Future work after v1; the design below is unchanged and still the plan. The Phase 6 writeup lists this as future work and makes no claims from it.

## Question
At a fixed training budget, how much **model quality** do we give up for **inference efficiency** (KV-cache memory, decode speed) when we replace standard multi-head attention?

## Hypotheses
- **H1:** GQA (2 KV heads) matches MHA val loss within noise and cuts KV cache 4×.
- **H2:** MLA comes close to MHA quality with an even smaller cache. It may underperform at this tiny scale.
- **H3:** A Gated DeltaNet hybrid (3 GDN : 1 MHA) has a slightly higher val loss but the best decode throughput at long context (8k).

## Fixed budget (freeze in Phase 5.1)
| Item | Value |
|---|---|
| Base config | `smoke` (~14M) or `main` (~30M), decide based on Phase 3 throughput |
| Tokens per run | _TBD_ (target ≤ 1 h per run) |
| Context | 512 for training; long-context decode tested with synthetic lengths |
| Optimizer / LR schedule | Same as Phase 3, LR swept once per variant only if a run diverges |
| Seeds | 2–3 per variant |

Param counts differ slightly across variants. Report them, and adjust width or ffn so they're within ±5% of each other.

## Variants
| ID | Attention | Notes |
|---|---|---|
| A | `mha` | Baseline |
| B | `gqa` | n_kv_heads = 2 |
| C | `mla` | kv_lora_rank and rope_dim _TBD_ |
| D | `gdn_hybrid` | `["gdn","gdn","gdn","mha"] * (L/4)`, using fla |

## Metrics
1. Final val loss (mean ± std across seeds)
2. Loss vs **wall-clock** and loss vs tokens
3. KV-cache or state bytes per token (from `cache_bytes_per_token()`)
4. Decode tok/s at 512 / 2k / 8k context, batch 1 and batch 8
5. Training throughput (tokens/sec)

## Threats to validity (put these in the writeup)
- Tiny scale: rankings may flip at larger models or more data.
- TinyStories is easy and short-context, so it understates the long-range benefits of full attention.
- fla Triton kernels vs PyTorch SDPA kernels: speed differences may reflect kernel maturity, not the method itself.
- Few seeds; small differences are noise.
- Laptop thermal throttling can distort wall-clock timings, so monitor and record GPU temps.

## Results (fill in)
| Variant | Params | Val loss (mean ± std) | Cache B/token | Decode tok/s @512 | @2k | @8k | Train tok/s |
|---|---|---|---|---|---|---|---|
| A mha | | | | | | | |
| B gqa | | | | | | | |
| C mla | | | | | | | |
| D gdn_hybrid | | | | | | | |

## Findings
_Write after 5.5._

---

## Future directions (PLAN Phase 7, deferred by D-021)
Beyond the attention ablation. The long-term question: can a machine **learn**, rather than only imitate its training data, with far less data? (A child hears roughly 6–60M words by age 5; LLMs train on trillions of tokens.)

**Source:** a literature review done on 2026-11-04, saved in `D:\ResumeProject\research-learning-not-imitation\` (report in `reports\`, per-topic notes in `research_notes\`). It's a starting map, not verified work: check prior work and links before relying on any claim.

**Main takeaways from that review:**
- **Cheap checkers, not new neurons.** The biggest gains beyond imitation came where a cheap, reliable checker of results exists: self-play in games (AlphaGo Zero), proof checkers (AlphaProof), test-based code and maths rewards (RL from verifiable rewards). Without such a checker, learning from experience is much harder.
- **New units mostly buy speed, not learning.** KANs, spiking neurons and SSMs / linear attention (Mamba, RWKV, Gated DeltaNet) mainly trade quality for efficiency, which is what Phase 5 measures. They don't by themselves fix sample efficiency.
- **World models help where the world can be simulated**: large sample-efficiency gains in games and robotics (e.g. Dreamer, EfficientZero). Results in open-ended language settings are weaker.
- **Continual learning is unsolved.** Loss of plasticity and catastrophic forgetting are still open, with recent approaches (Titans, test-time training, sparse memory finetuning) as partial answers.
- **Tiny recursive reasoners are worth watching.** TRM (7M params) showed strong ARC puzzle results; critiques point at puzzle-specific tricks (augmentation, voting).

**Candidate experiments:** the 8 listed in [PARKING_LOT.md](../PARKING_LOT.md), with known caveats, plus the Weight-Space Programming replication ([PRIOR_ART.md](PRIOR_ART.md) PA-2).

**How this links to the current ablation:** Phase 5 asks how much quality we give up for efficiency. Phase 7 asks the opposite question: how much ability we get per token of data. The same model code, training loop and benchmark discipline serve both.
