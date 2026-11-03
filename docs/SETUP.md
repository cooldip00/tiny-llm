# Environment Setup Log

Every command we ran to build the environment, in order, with the reason and the result.
Use it to rebuild on a fresh machine, or to debug "why is X this version?".

Related decisions: D-002 (WSL2), D-006 (uv), D-013 (W&B), D-014 (`~/llm`), D-017 (torch version).

---

## Final versions (2026-11-03)
| Component | Version | Why this one |
|---|---|---|
| Windows | 11 Home 10.0.26300 | Host OS |
| NVIDIA driver (Windows) | 610.91 | Already installed; supports CUDA up to **13.3** |
| WSL | 2.6.1.0 (kernel 6.6.87.2-1) | Already installed; new enough for `--location` |
| Ubuntu | 24.04.5 LTS | Current LTS (support until 2029); best-tested target for PyTorch, vLLM, Triton |
| uv | 0.13.7 | Latest at install time |
| Python | 3.12.16 | Python 3.12 has the widest ML library support; 3.13/3.14 still miss some wheels |
| PyTorch | **2.14.1+cu130** | See [why this PyTorch version](#why-this-pytorch-version) |
| Triton | 3.8.0 | Installed automatically as a torch dependency (needed for `torch.compile`, fla and our kernels) |
| git | 2.43.0 | Ships with Ubuntu 24.04 |

---

## Step 1 — Check free disk space (Windows PowerShell)
```powershell
Get-PSDrive C, D | Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
wsl --version
```
**Why:** WSL installs to C: by default, and we need about 20–30 GB (dataset, TinyLlama, checkpoints, the Python environment).
**Result:** C: had 38.3 GB free (too tight, since Windows needs room too), D: had 167 GB. **So we installed on D:.**

## Step 2 — Install Ubuntu on D: (Windows PowerShell, run by the user)
```powershell
wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04
```
**Why:** Linux is needed for Triton/`torch.compile`, `flash-linear-attention` and vLLM (D-002). `--location` puts the virtual disk on D:.
Prompts for a Linux username (`kuladip`) and password (used for `sudo`; never share it).

Fallback if `--location` isn't supported:
```powershell
wsl --install -d Ubuntu-24.04
wsl --shutdown
wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04
```

## Step 3 — Verify the install and the GPU in WSL
```powershell
wsl -l -v
wsl -d Ubuntu-24.04 -- bash -lc "whoami; lsb_release -ds; nvidia-smi; df -h ~ | tail -1; free -h | head -2"
```
**Why:** confirm the GPU is visible inside Linux. CUDA in WSL comes from the **Windows** driver.
⚠️ **Never install an NVIDIA Linux driver inside WSL.** It breaks GPU passthrough.
**Result:** RTX 3070 Ti Laptop, 8192 MiB, CUDA 13.3, 140 W cap. WSL sees 7.6 GiB of RAM (default 50%).

Confirm where the disk is stored:
```powershell
Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | ForEach-Object { Get-ItemProperty $_.PSPath } | Select-Object DistributionName, BasePath
```
**Result:** `Ubuntu-24.04 → D:\WSL\Ubuntu-24.04` ✅

## Step 4 — Move the project into WSL (D-014)
```bash
mkdir -p ~/llm
cp -r /mnt/d/ResumeProject/tiny-llm/. ~/llm/
diff -r /mnt/d/ResumeProject/tiny-llm ~/llm && echo IDENTICAL
```
**Why:** the Linux filesystem is much faster than `/mnt/d` for Linux tools, and code plus docs live in one git repo.
**Result:** IDENTICAL. The old D: folder is now just a `MOVED.md` pointer.

## Step 5 — git
```bash
cd ~/llm
git init -b main
git config user.name  "Kuladip Joshi"          # repo-local, not global
git config user.email "kuladip00@gmail.com"
```
Plus a `.gitignore` that keeps data, checkpoints, model weights (`*.bin`, `*.safetensors`, `*.pt`, `*.gguf`), `wandb/`, `.venv/` and secrets (`.env`, `.netrc`) out of git.

## Step 6 — Install uv and Python 3.12
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env          # put uv on PATH in the current shell
uv python install 3.12
```
**Why uv:** fast installs, a lockfile, and it manages Python itself, so there's no system-Python conflict (D-006).
**Why 3.12:** the safest version for ML wheels today (torch, triton, fla, vLLM).

## Step 7 — Find which PyTorch builds exist
PyTorch publishes a separate index for each CUDA version. We listed the newest torch for Python 3.12 on each:
```bash
for cu in cu126 cu128 cu129 cu130 cu131 cu132 cu133; do
  v=$(curl -s "https://download.pytorch.org/whl/$cu/torch/" \
      | grep -oE "torch-[0-9.]+\+$cu-cp312-cp312-manylinux[^\"#]*x86_64\.whl" \
      | sed 's/-cp312.*//' | sort -uV | tail -1)
  echo "$cu: ${v:-none}"
done
```
**Result (2026-11-03):**
```
cu126: torch-2.14.1+cu126
cu128: torch-2.11.0+cu128
cu129: torch-2.13.0+cu129
cu130: torch-2.14.1+cu130
cu131: none
cu132: torch-2.15.0+cu132
cu133: none
```

### Why this PyTorch version
Two numbers matter: the **torch version** and the **CUDA version it's built for**.

1. **The CUDA build must be ≤ what the driver supports.** The driver supports 13.3, so cu126 through cu132 all work.
2. **torch 2.14.1 over 2.15.0:**
   - 2.14.1 is a **patch release**: the early bugs of 2.14.0 are already fixed.
   - 2.15.0 is the newest major release. Ecosystem libraries we need later (`flash-linear-attention`, vLLM, bitsandbytes-style tools) often lag the newest torch by weeks.
   - We gain nothing from 2.15 for this project.
3. **cu130 over cu126 for 2.14.1:** both have the same torch, but cu130 has newer CUDA libraries (cuBLAS, cuDNN), and Ampere (sm_86) is fully supported in CUDA 13.
4. **Not cu128 or cu129:** they only have older torch versions.

Rule: **pin an exact version** so results are reproducible. Change it only with a new DECISIONS.md entry.

## Step 8 — Project config and install
`pyproject.toml` (key parts):
```toml
requires-python = ">=3.12,<3.13"
dependencies = ["torch==2.14.1", "numpy"]

[dependency-groups]
dev = ["pytest"]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true            # only packages that ask for this index use it

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```
**Why `explicit = true`:** only torch comes from the PyTorch index; everything else comes from PyPI. This avoids pulling odd versions of other packages from the PyTorch index.

```bash
cd ~/llm
uv python pin 3.12        # writes .python-version
uv sync                   # creates .venv, installs everything, writes uv.lock
```
Takes about 10 minutes: torch plus the NVIDIA CUDA libraries are several GB.

## Step 9 — Verify PyTorch and CUDA
```bash
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())'
```
**Result:**
```
2.14.1+cu130 13.0 True NVIDIA GeForce RTX 3070 Ti Laptop GPU (8, 6) True
```
✅ The GPU is usable, compute capability 8.6 (Ampere), bf16 supported.

## Step 10 — W&B account (browser)
- Signed up at wandb.ai (now CoreWeave Forge) as **Professional** (the Academic plan requires a university affiliation).
- Username `cooldip00`, org `cooldip00-personal`, use case "Train and fine-tune models".
- The trial ends around **2026-12-02**. Then downgrade to Free (D-013).
- `wandb login` happens later, inside WSL. **The API key never goes into the repo or the chat.**

## Step 11 — Benchmark the GPU (Step 0.5)
```bash
cd ~/llm
uv run python scripts/check_env.py        # ~1 min; --sustain 0 skips the 30 s load test
```
**Why:** replace the guessed TFLOPS in the plan with real numbers, and get the bandwidth ceiling the inference work is measured against.
Run it **plugged in, in Nitro performance mode**. On battery the GPU is power-limited and the numbers drop.
**Result:** bf16 39.7 TFLOPS (n=8192), 38.3 sustained, read bandwidth 423 GB/s. Full table in [HARDWARE.md](HARDWARE.md).

## Step 12 — Data libraries and TinyStories (Step 1.1)
```bash
cd ~/llm
uv add datasets tokenizers                   # datasets 5.1.0, tokenizers 0.23.3 (pyarrow 26.0.0 comes with them)
uv run python scripts/download_data.py       # downloads into ~/.cache/huggingface, prints stats
```
**Why these libraries:** `datasets` downloads the data and caches it as Arrow files, which are memory-mapped so they're RAM-friendly (D-009). `tokenizers` is a fast Rust BPE trainer (D-007). Neither one is a model framework; the model and training code stays ours.
**Why no version pin:** unlike torch, these don't affect numerical results. `uv.lock` still records the exact versions.
**Result:** 2,119,719 train / 21,990 val stories. 2.8 GB in the HF cache, outside the repo.
The "unauthenticated requests" warning is harmless. A HF token only raises the rate limits.

## Step 13 — Make `tinyllm` an installed package, and add tests (Step 1.1b)
Added to `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tinyllm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
```bash
uv sync               # now also installs tinyllm itself, in editable mode
uv run pytest -q      # run all tests
```
**Why:** with a build backend, `uv sync` installs our `src/tinyllm` package in *editable* mode. Scripts and tests can then `import tinyllm` from anywhere, with no `sys.path` hacks, and code changes apply immediately without reinstalling.
**Why hatchling:** the most common, minimal backend that works with a `src/` layout.
**Result:** `+ tinyllm==0.1.0 (from file:///home/kuladip/llm)`, 8 tests passed.

## Step 14 — Build the training data (Phase 1)
Run once, in order. Each step reads the output of the one before:
```bash
cd ~/llm
uv run python scripts/train_tokenizer.py   # ~7.5 min, peak RAM 2.7 GB → tokenizer/tinystories-bpe8192.json (committed)
uv run python scripts/tokenize_data.py     # ~15 min, peak RAM 2.7 GB → data/train.bin (931 MB), data/val.bin (9 MB)
uv run pytest -q                           # 17 tests
```
**Why the tokenizer file is committed but `data/` isn't:** the tokenizer is small (554 KB) and defines what each token ID means, so it must never silently change. The `.bin` files are large and can be rebuilt from the tokenizer plus the dataset.

## Step 15 — W&B logging (Step 3.4)
```bash
cd ~/llm
uv add wandb                 # wandb 0.31.0
uv run wandb login           # once; paste the API key at the prompt (input may not echo)
```
**Getting the key (W&B is now CoreWeave Forge):** forge.coreweave.com → profile icon → **User settings** → **API keys** → **Create new key**. The key is shown only once; if it's lost, delete it and make a new one.
**Check it worked:** `~/.netrc` gets a `machine api.forge.coreweave.com` entry (permissions 600), and `wandb.Api().viewer.username` returns `cooldip00`. Done 2026-11-03.
**Why:** training metrics and samples go to W&B (D-013). `wandb login` stores the key in `~/.netrc` inside WSL. **Never put the key in the repo, the docs or the chat.**
Logger choices on the training CLI:
- `--logger wandb` (the default) with `--wandb_mode online` (the default). Needs the login above.
- `--wandb_mode offline` writes runs to `~/llm/wandb/` (git-ignored). Upload them later with `uv run wandb sync wandb/offline-run-*`.
- `--logger none` logs to the terminal only. The tests use this, so they never touch the network.

A resumed run (`--resume`) reuses the W&B run ID stored in `ckpt.pt`, so the charts continue in the same run.

## Step 16 — C compiler for `torch.compile` / Triton (before Step 3.7)
```bash
sudo apt update && sudo apt install -y build-essential    # run it yourself; type the sudo password only in the terminal
gcc --version                                             # gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```
**Why:** `torch.compile` generates Triton GPU kernels, and Triton compiles a small C launcher for each one. Without `gcc`, it fails with `RuntimeError: Failed to find C compiler`. The Phase 4.10 Triton kernels need it too.
**Result:** training on the main preset runs +57% faster (D-020). Done 2026-11-03.

---

## Notes
- **Running WSL commands from Windows:** `wsl -d Ubuntu-24.04 -- bash -lc "cd ~/llm && <command>"`.
  Complex bash (with `$`, quotes or `${...}`) gets mangled by PowerShell, so put it in a `.sh` file and run `wsl -d Ubuntu-24.04 -- bash /path/script.sh`.
- **Default distro:** `wsl --set-default Ubuntu-24.04` (optional) makes a plain `wsl` open Ubuntu.
- **WSL RAM:** defaults to 50% of host RAM (7.6 GiB). To raise it, add this to `%UserProfile%\.wslconfig`:
  ```ini
  [wsl2]
  memory=10GB
  ```
  then run `wsl --shutdown`.
- **VS Code:** from an Ubuntu terminal run `code ~/llm` (installs the WSL extension on first use).

## Troubleshooting

### "\\wsl.localhost\Ubuntu-24.04 is not accessible" / `ERROR_SHARING_VIOLATION` (2026-11-03)
**Symptom:** Explorer can't open the Ubuntu folder; `wsl -d Ubuntu-24.04` fails with
`Failed to attach disk ... ext4.vhdx ... being used by another process`.
**Cause:** Windows had mounted `ext4.vhdx` as a host disk (it showed up in `Get-Disk` as "File Backed Virtual"), so WSL couldn't attach it. That happens if the `.vhdx` gets double-clicked or opened.
**Diagnose:**
```powershell
Get-Disk | Where-Object { $_.BusType -eq 'File Backed Virtual' } | Select-Object Number, Location
```
**Fix (needs an admin PowerShell):**
```powershell
wsl --shutdown
Dismount-DiskImage -ImagePath 'D:\WSL\Ubuntu-24.04\ext4.vhdx'    # expect Attached : False
```
Or use Disk Management → right-click the disk → **Detach VHD**.
⚠️ Never click "Initialize" or "Format" on that disk. It would erase Ubuntu.
**Prevention:** never open files inside `D:\WSL\`. Browse through Explorer → Linux → Ubuntu-24.04, with an Ubuntu terminal open.

## Rebuild from scratch (quick path)
```powershell
wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04
```
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone <repo-url> ~/llm && cd ~/llm
uv sync                      # uses uv.lock → exact same versions
uv run python scripts/check_env.py
```
