# Progress

## Current
- **Phase:** 3 — Training (Phase 2 complete)
- **Step:** 3.3 Eval every N steps, checkpoint save/resume, sample text at each eval
- **Next action:** add `evaluate()` (fixed val batches), `save/load_checkpoint` (model, optimizer, step, rng state, configs) and a sampler to `train.py`, with tests
- **Blockers:** none
- **Open decisions:** none
- **Housekeeping:** delete the old doc copies in `D:\ResumeProject\tiny-llm` (keep `MOVED.md`). My automatic delete was blocked.
- **Reminder:** the W&B/Forge trial ends around 2026-12-02. Downgrade to Free then.

---

## Log
Newest first. One entry per work block: what was done, numbers, next.

### 2026-11-03 — Step 3.2 training loop
- `src/tinyllm/train.py`:
  - `TrainConfig` dataclass; every field is also a CLI flag (`python -m tinyllm.train --preset main --max_steps 100`).
  - `lr_at`: linear warmup, then cosine to `min_lr`.
  - `make_optimizer`: fused AdamW, with decay on 2-D weights only.
  - `train_step`: gradient accumulation (loss ÷ n micro-batches), bf16 autocast on CUDA, clip at 1.0, and returns device tensors so there's no GPU sync per step.
  - `train`: the step loop, logging loss / lr / grad norm / tok/s.
- Hyperparameters recorded as **D-019**: 131,072 tokens/step (32×512 × 8), peak LR 1e-3, warmup 200, cosine to 1e-4, wd 0.1, betas (0.9, 0.95).
  - My first default had 4 accumulation steps, which is only 65,536 tokens/step. The test caught it, and it's fixed to 8.
- Short GPU runs (warmup 10):

| Preset | Steps | Loss | Throughput | ≈ TFLOPS (6·N·tok/s) | 1B tokens would take |
|---|---|---|---|---|---|
| smoke | 40 | 9.07 → 4.59 | ~131K tok/s | 10.8 | ~2.1 h |
| main | 30 | 9.09 → 5.60 | ~74K tok/s | 13.3 (~32% of peak) | **~3.7 h** |

- `tests/test_train.py`:
  - LR schedule values and monotonic decay; optimizer groups (norms not decayed, every param exactly once).
  - **Grad accumulation 2 × B equals one 2B batch** (SGD, params equal to 1e-6).
  - Clipping: the reported norm is pre-clip, and the step size equals the clip value.
  - The CPU loop learns a periodic corpus; seq_len above max raises; CUDA fused + bf16 path.
- 87/87 tests pass.

### 2026-11-03 — Step 3.1 data loader
- `tinyllm.data.get_batch(tokens, batch_size, seq_len, rng, device)`:
  - Returns random windows `x = tokens[s:s+T]`, `y = tokens[s+1:s+T+1]` as int64. Sampling is with replacement, and windows cross story boundaries (EOS marks where a story ends).
  - One vectorised fancy-index gather from the memmap. On CUDA the batch goes through pinned memory with `non_blocking=True`.
  - The seeded `np.random.Generator` makes runs reproducible. Saving `rng.bit_generator.state` in a checkpoint makes resume exact (tested).
- Throughput, train memmap to GPU: 16×512 at 6.0 ms (1.4M tok/s); 32×512 at 4.6 ms (**3.6M tok/s**).
  - Training is expected at ~70K tok/s, so the loader costs ~2% of step time and needs no prefetch thread.
- `tests/test_batch.py`: shapes/dtype, contiguous windows with y shifted by one, same seed gives the same batch, rng-state resume, both ends of the array reachable with no out-of-range starts, too-short input raises, CUDA device, real `train.bin`.
- 79/79 tests pass.

### 2026-11-03 — Step 2.6 one-batch overfit done, so Phase 2 is complete
- `scripts/overfit_batch.py`: one fixed batch of 8 × 256 random windows from `train.bin`, AdamW lr 1e-3 (no weight decay), bf16 autocast, 300 steps. Exits non-zero if the final loss is ≥ 0.05.

| Preset | Params | Start loss | Final loss | < 1.0 at step | < 0.05 at step | Time | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| smoke | 13.8M | 9.067 | **0.0033** | 118 | 153 | 9.7 s | 727 MiB |
| main | 29.9M | ~9.0 | **0.0034** | 184 | 209 | 12.3 s | 1213 MiB |

- Start loss ≈ ln(8192) = 9.011, as expected from the init.
- Phase 2 "Done when" met: all 72 tests pass and the overfit succeeds.

### 2026-11-03 — Step 2.5 whole-model tests
- Added to `tests/test_model.py`:
  - Shapes at T = 1, 7 and 512 (max).
  - Causality through all 6 layers: changing the token at position 20 leaves logits 0–19 identical.
  - `num_params()` == `estimate_params()` for smoke, main, untied, and 3 layers. Exact sizes: smoke 13,767,552, main 29,893,120.
  - Initial loss within 0.1 of ln(8192) = 9.01, so the init is near-uniform as intended.
  - Every parameter gets a non-zero gradient (nothing detached or unused).
  - bf16 autocast on CUDA: bf16 logits, fp32 finite loss ≈ 9.01.
- 72/72 tests pass.

### 2026-11-03 — Step 2.4 block + full model
- `src/tinyllm/model.py`:
  - `Block`: pre-norm, `x + self_attn(input_layernorm(x))`, then `x + mlp(post_attention_layernorm(x))`. The attention class comes from the registry via `cfg.layer_types`.
  - `Transformer`: `embed_tokens` → layers → `norm` → `lm_head`, with `lm_head` tied to `embed_tokens` by default (saves V·d = 4.2M params in main).
    - `forward(idx, targets=None)` returns `(logits, loss)`; the loss is cross-entropy computed in fp32.
    - `num_params()` counts the tied matrix once.
  - Init: Linear/Embedding weights N(0, 0.02); `o_proj` and `down_proj` N(0, 0.02/√(2·n_layers)) (smoke: 0.0058); RMSNorm weights 1.
- The attention import inside `Block.__init__` is lazy because `attention/mha.py` imports `apply_rope` from `model.py` (avoids a circular import).
- `tests/test_model.py`: block and model shapes, tied/untied head, HF Llama names, RoPE tables not in the state_dict, init std per weight group, too-long input rejected.
- 60/60 tests pass.

### 2026-11-03 — Step 2.3 attention interface + `mha`
- `src/tinyllm/attention/base.py`: `Attention` contract (`forward(x, rope, cache=None)`, `cache_bytes_per_token()`), `ATTENTION` registry, `@register` decorator (duplicates raise), `build_attention(cfg, layer_idx)` (unknown names raise `KeyError` listing the registered ones).
- `src/tinyllm/attention/mha.py`: `MHA` with HF names `q/k/v/o_proj`, no biases, RoPE on q and k, `F.scaled_dot_product_attention(is_causal=True)`.
  - Rejects `n_kv_heads != n_heads` (that is `gqa`, Step 4.9a).
  - `cache` raises `NotImplementedError` until the KV cache in Phase 4.
  - Cache cost: main = 2048 bytes per token per layer (bf16), 16 KB per token for 8 layers.
- `tests/test_attention.py`: matches a naive explicit-mask softmax oracle (atol 1e-5); changing tokens 10–15 leaves outputs 0–9 unchanged; registry, HF names, GQA rejection; bf16 autocast on CUDA within 3e-2 of fp32.
- 52/52 tests pass (CUDA test ran, nothing skipped).

### 2026-11-03 — Step 2.2 components
- `src/tinyllm/model.py`:
  - `RMSNorm`: statistic computed in fp32, output in the input dtype.
  - `RotaryEmbedding` + `apply_rope`: rotate_half (HF Llama) layout, precomputed non-persistent cos/sin, `offset` for KV-cached decode.
  - `SwiGLU`: HF names `gate_proj` / `up_proj` / `down_proj`, no biases.
- `tests/test_components.py`, checked against independent references:
  - RMSNorm matches `F.rms_norm`.
  - RoPE matches a complex-number reference, is identity at position 0, preserves norms, and gives the same q·k score for the same relative distance (pos 5/2 vs 105/102).
  - SwiGLU: formula, param count 3·d·ffn, HF names.
- 41/41 tests pass.

### 2026-11-03 — Step 2.1 ModelConfig
- `src/tinyllm/config.py`: frozen `ModelConfig` dataclass.
  - Fields: sizes, `n_kv_heads` (None = MHA), per-layer `layer_types` (None = all `mha`), norm eps, RoPE theta, init std, tied embeddings.
  - Validation: heads divide d_model; KV heads divide heads; even head_dim for RoPE; layer_types length matches n_layers.
  - Also: `estimate_params()`, dict round trip for checkpoints, and `get_config(name, **overrides)`.
- Exact param counts: smoke **13,767,552**, main **29,893,120**. They match the ARCHITECTURE.md estimates.
- `tests/test_config.py`: 29/29 tests pass in total.

### 2026-11-03 — Steps 1.3 + 1.4 done, so Phase 1 is complete
- `tinyllm.data.write_tokens` / `load_tokens`: stream-encode, story + EOS, uint16 written via `.tmp` then renamed; memmap loader.
- `scripts/tokenize_data.py` (wall 879 s, peak RAM 2.7 GB):

| Split | Stories | Tokens (incl. EOS) | File | Avg / median / p99 / max tokens per story | Stories > 512 ctx |
|---|---|---|---|---|---|
| train | 2,118,859 | **465,373,792** | 930.7 MB | 219 / 187 / 641 / 1576 | 3.05% |
| val | 21,981 | **4,677,682** | 9.4 MB | 212 / 185 / 622 / 1079 | 2.60% |

- Checks passed: one EOS per story, the file ends with EOS, max id < vocab, and the first, middle and last story decode exactly from the memmap.
- Tests: 17/17 pass (adds `tests/test_tokens.py`).
- Implications:
  - The 1B-token main run is about **2.15 epochs** of train.
  - Ctx 512 covers about 97% of stories whole. Training samples random 512-token windows across story boundaries (EOS marks the boundary).
- Possible speed-up (not needed now): encoding was single-core, ~15 min. Parallelize only if we ever re-tokenize often.

### 2026-11-03 — Step 1.2 tokenizer trained
- `src/tinyllm/tokenizer.py`: byte-level BPE with a GPT-2-style pre-tokenizer, `add_prefix_space=False` (exact roundtrip), `min_frequency=2`, all 256 bytes in the base alphabet, `<|endoftext|>` = id 0.
- `scripts/train_tokenizer.py` trained on 2,118,859 cleaned train stories in **443 s**, peak RAM **2.7 GB**.
- Saved to `tokenizer/tinystories-bpe8192.json` (554 KB, committed).
- Validation: **4,655,701 tokens**, **4.12 chars/token**, **212 tokens/story**, 0 roundtrip mismatches.
- Train-token estimate: 1.898B chars ÷ 4.12 ≈ **461M tokens**. The exact count comes in 1.3.
- The last merges learned are whole words (`Ġscooters`, `Ġknives`), so the vocabulary is saturated with real words, not junk.
- `tests/test_tokenizer.py`: vocab, EOS id, uint16 fit, exact roundtrip (including non-ASCII via bytes), common words are single tokens. 16/16 tests pass in total.

### 2026-11-03 — Step 1.1b text cleanup (D-018)
- `scripts/inspect_chars.py` showed that the non-ASCII text is mojibake. Of 130,115 non-ASCII train stories, 129,940 are affected. Top sequences: `â€œ` 270,583; `â€™` 195,201; bare `â€` ~270K (lost closing quote); dashes 15K.
- Added `src/tinyllm/data.py` with `clean_text` and `load_clean`, and made `tinyllm` an installed package (hatchling build backend).
- `tests/test_data.py`: 8 passed.
- `scripts/check_clean.py`: train 2,118,859 kept (860 dropped), val 21,981 kept (9 dropped), 0 non-ASCII left, 1,898,184,006 train characters.

### 2026-11-03 — Phase 1.1 done
- Added `datasets` 5.1.0 and `tokenizers` 0.23.3.
- `scripts/download_data.py` downloaded `roneneldan/TinyStories` into the HF cache (2.8 GB on disk).
- Train: 2,119,719 stories, 1.90B characters, average 896 characters, 230 empty stories.
- Val: 21,990 stories, 19.2M characters, none empty.
- 6.1% of train stories contain non-ASCII characters. Check what they are before training the tokenizer.
- Rough estimate: ~4 characters per token gives ~470M train tokens, so the planned 1B-token main run is about 2 epochs.

### 2026-11-03 — Phase 0.5 done, so Phase 0 is complete
- Added `scripts/check_env.py`. Ran it plugged in, in performance mode.
- bf16 GEMM n=8192: **39.7 TFLOPS** (94% of the 42.0 theoretical peak). Sustained 30 s: **38.3 TFLOPS**, 70 → 81 °C, no real throttling.
- Bandwidth: copy 410 GB/s, read 423 GB/s (rated 448).
- Training estimate revised to ~12–19 TFLOPS effective, so the main run takes ~2.6–4 h (was 4–5 h).
- **Next:** Phase 1.

### 2026-11-03 — Phase 0.3, 0.6, 0.7 done; move to `~/llm`
- Docs copied to `~/llm` and checked identical with `diff -r`. D-014 accepted.
- `git init -b main`, `.gitignore` added, local author set. Nothing committed yet.
- uv 0.13.7, Python 3.12.16, `.python-version` pinned to 3.12.
- `pyproject.toml`: torch 2.14.1 from the cu130 index (D-017), numpy, pytest (dev).
- 0.4 verified: `torch 2.14.1+cu130`, CUDA 13.0, `cuda.is_available()=True`, RTX 3070 Ti Laptop, compute capability (8, 6), bf16 supported. Triton 3.8.0 came with torch.

### 2026-11-03 — Phase 0.1–0.2 done
- Free space: C: 38.3 GB, D: 167 GB, so Ubuntu was installed on D:.
- WSL 2.6.1.0. Ubuntu 24.04.5 LTS, user `kuladip`, disk at `D:\WSL\Ubuntu-24.04`.
- `nvidia-smi` in WSL: RTX 3070 Ti Laptop, 8192 MiB, driver 610.91, CUDA 13.3 supported, 140 W power cap.
- RAM visible to WSL: 7.6 GiB (WSL's default of 50% of host RAM).

### 2026-11-03 — W&B account
- Checked W&B pricing on the official page (now CoreWeave Forge): Free = $0, 5 GB/month, public projects allowed.
- Created the account (`cooldip00`, org `cooldip00-personal`). D-013 accepted.

### 2026-11-03 — Plan update: Track B
- Reviewed a LinkedIn post about a C++/CUDA inference engine for TinyLlama 1.1B; adopted its method, not its scope.
- Added D-015 (TinyLlama 1.1B inference track) and D-016 (timeboxed Triton kernels).
- PLAN: new steps 4.9 and 4.10; `gqa` pulled forward into 4.9a; timeline extended.
- INFERENCE.md: bandwidth-ceiling method, reference-answer rule, TinyLlama and Triton results tables.

### 2026-11-03 — Planning
- Discussed feasibility, the inference-at-scale concepts and the research options.
- Chose the framework stack (DECISIONS.md D-001 to D-012).
- Host environment check:
  - GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB, driver 610.91
  - Python 3.12.6 (Windows), nvcc 12.4
  - WSL2: only the `docker-desktop` distro exists, so Ubuntu still needs installing
- Created the project docs.
- **Next:** Phase 0.1.
