# Research: Efficient Attention Ablation

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
