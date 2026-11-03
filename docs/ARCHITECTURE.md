# Model Architecture

Llama-style decoder-only transformer, written from scratch (DECISIONS D-004, D-005).

## Block
```
x ─► RMSNorm ─► Attention (swappable) ─► + ─► RMSNorm ─► SwiGLU MLP ─► + ─► out
└──────────────── residual ──────────────┘   └──────── residual ─────────┘
```
- Pre-norm with RMSNorm, eps 1e-5
- RoPE on q and k (theta 10000)
- SwiGLU: `down(silu(gate(x)) * up(x))`
- No biases anywhere; input embedding tied to the LM head
- Final RMSNorm before the LM head

## Configs
Vocab 8192, context 512 (TinyStories samples are short).

| Name | d_model | n_layers | n_heads | head_dim | ffn_hidden | Params (approx.) |
|---|---|---|---|---|---|---|
| `smoke` | 384 | 6 | 6 | 64 | 1024 | ~13.8M |
| `main` | 512 | 8 | 8 | 64 | 1408 | ~29.9M |

Param formula (tied embeddings, MHA): `V·d + L·(4·d² + 3·d·ffn) + small norm params`.
The causality and param-count tests in Phase 2.5 check against this.

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
  data.py          # tokenizer training, .bin creation, memmap loader
  train.py         # training loop
  generate.py      # sampling + inference engine (Phase 4)
  export.py        # HF Llama export
scripts/           # check_env.py, benchmarks
tests/             # shapes, causality, param count
```
