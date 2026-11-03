# Progress

## Current
- **Phase:** 1 — Data & tokenizer
- **Step:** 1.2 Train the BPE tokenizer
- **Next action:** train byte-level BPE (vocab 8192) on `load_clean("train")`. The cleanup is done (D-018).
- **Blockers:** none
- **Open decisions:** none
- **Housekeeping:** delete the old doc copies in `D:\ResumeProject\tiny-llm` (keep `MOVED.md`). My automatic delete was blocked.
- **Reminder:** the W&B/Forge trial ends around 2026-12-02. Downgrade to Free then.

---

## Log
Newest first. One entry per work block: what was done, numbers, next.

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
