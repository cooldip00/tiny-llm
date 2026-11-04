# tiny-llm — Session Anchor

## Goal (do not drift)
Build a small Llama-style LLM **from scratch** on a single RTX 3070 Ti Laptop (8 GB), then:
1. Train it (TinyStories first).
2. Build our own inference engine and learn how inference works at scale.
3. Run a small, honest research ablation: **can we swap attention for something more efficient?** _(Deferred per D-021: future work after v1. Don't start Phase 5 or Phase 7 before v1 ships.)_
4. Write it up as a resume-quality project.

## Session start ritual (every session, every context reset)
1. Read [PROGRESS.md](PROGRESS.md) and find the current phase and step.
2. Read that phase in [PLAN.md](PLAN.md), including its "Done when" criteria.
3. Check [DECISIONS.md](DECISIONS.md) before proposing any tool, library or design change.
4. Then work **only** on the current step.

## Scope rules
- New idea that's not in the current step → append it to [PARKING_LOT.md](PARKING_LOT.md), don't do it.
- Changing a settled decision → add a new entry to DECISIONS.md that supersedes it. Never silently switch.
- No high-level frameworks (HF Transformers/Trainer, Lightning). Model and training code is ours.
- A step is done only when its "Done when" criteria pass, with the output shown.

## End of each work block
Update PROGRESS.md: what was done, the measured numbers, the next action, any blockers.

## Environment conventions
- Host: Windows 11, PowerShell. Runtime: WSL2 Ubuntu 24.04, bash. See [docs/HARDWARE.md](docs/HARDWARE.md).
- Python 3.12 via `uv`. PyTorch CUDA wheel.
- **Never** install an NVIDIA Linux driver inside WSL. The Windows driver provides CUDA to WSL.

## Reference docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): model spec and the swappable attention interface
- [docs/INFERENCE.md](docs/INFERENCE.md): serving roadmap and benchmark protocol
- [docs/RESEARCH.md](docs/RESEARCH.md): ablation design and results
- [docs/HARDWARE.md](docs/HARDWARE.md): specs, compute and memory budget math
- [docs/PRIOR_ART.md](docs/PRIOR_ART.md): related projects, what we adopted from them, and how we differ (for the writeup)
- [docs/SETUP.md](docs/SETUP.md): every environment command run, with reasons and versions. Append new installs here
