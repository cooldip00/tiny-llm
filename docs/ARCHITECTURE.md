# Model Architecture

Llama-style decoder-only transformer, written from scratch (DECISIONS D-004, D-005).

## Block
```
x ─► RMSNorm ─► Attention (swappable) ─► + ─► RMSNorm ─► SwiGLU MLP ─► + ─► out
└──────────────── residual ──────────────┘   └──────── residual ─────────┘
```
- Pre-norm with RMSNorm, eps 1e-5
- RoPE on q and k (theta 10000), in the **rotate_half** layout (channel i is paired with i + head_dim/2), the same as HF Llama. TinyLlama weights then work without permuting q/k. cos/sin tables are precomputed up to `max_seq_len` and aren't saved in checkpoints.
- SwiGLU: `down(silu(gate(x)) * up(x))`
- No biases anywhere; input embedding tied to the LM head
- Final RMSNorm before the LM head

## Configs
Vocab 8192, context 512 (TinyStories samples are short).

| Name | d_model | n_layers | n_heads | head_dim | ffn_hidden | Params (approx.) |
|---|---|---|---|---|---|---|
| `smoke` | 384 | 6 | 6 | 64 | 1024 | 13,767,552 |
| `main` | 512 | 8 | 8 | 64 | 1408 | 29,893,120 |

Param formula (tied embeddings, MHA): `V·d + L·(4·d² + 3·d·ffn + 2·d) + d`. The `2·d` per layer and the final `d` are the RMSNorm weights.
With GQA, `4·d²` becomes `2·d² + 2·d·kv_dim`. `ModelConfig.estimate_params()` implements this, and the Phase 2.5 test checks the real model against it.

In code: `get_config("main")`, or with overrides such as `get_config("main", n_kv_heads=2)`. Configs are frozen (immutable) and save to and load from dicts for checkpoints.

## Swappable attention interface
Every attention variant implements the same module contract:

```python
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int): ...
    def forward(self, x, rope, cache=None) -> Tensor: ...
    # cache: per-layer state object (KV cache for softmax attention,
    #        recurrent state for linear attention). None during training.
    def cache_bytes_per_token(self) -> int: ...   # used in Phase 5 metrics
```

Registry: `ATTENTION = {"mha": ..., "gqa": ..., "mla": ..., "gdn": ...}`

Per-layer choice through config, so hybrids work:
```python
layer_types = ["gdn", "gdn", "gdn", "mha"] * 2   # 3:1 linear-to-full hybrid
```

| Variant | Phase | Key idea | Cache/token/layer |
|---|---|---|---|
| `mha` | 2 | Baseline multi-head attention | `2·n_heads·head_dim` |
| `gqa` | 5 | Fewer KV heads, shared across query groups | `2·n_kv_heads·head_dim` |
| `mla` | 5 | Low-rank latent KV (DeepSeek), decoupled RoPE | `kv_lora_rank + rope_dim` |
| `gdn` | 5 | Gated DeltaNet linear attention (fla) | **constant** state, independent of length |

## Weight naming (for HF Llama export, Phase 4.8)
Use the HF Llama names so the export is a straight rename-free dump:
`embed_tokens`, `layers.{i}.input_layernorm`, `self_attn.{q,k,v,o}_proj`,
`post_attention_layernorm`, `mlp.{gate,up,down}_proj`, `norm`, `lm_head` (tied).
Only `mha` and `gqa` export to HF Llama. The other variants are research-only.

## Init
- Linear and embedding weights: normal, std 0.02
- Residual output projections (`o_proj`, `down_proj`): std `0.02 / sqrt(2·n_layers)`

## Planned code layout
```
src/tinyllm/
  config.py        # ModelConfig, named presets
  model.py         # RMSNorm, RoPE, MLP, Block, Transformer
  attention/       # mha.py, gqa.py, mla.py, gdn.py, __init__.py (registry)
  data.py          # dataset loading + text cleanup (D-018), .bin creation, memmap loader
  tokenizer.py     # byte-level BPE 8192: train / load (D-007)
  train.py         # training loop
  generate.py      # sampling + inference engine (Phase 4)
  export.py        # HF Llama export
scripts/           # check_env.py, benchmarks
tests/             # shapes, causality, param count
```
