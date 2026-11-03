"""Step 2.6: overfit one fixed batch. Proves the model + loss + optimizer can learn.

A healthy model memorises a single batch: loss falls from ~ln(8192) = 9.0 to ~0.
If it plateaus, something is broken (mask, RoPE, init, loss shift, ...).

    uv run python scripts/overfit_batch.py [--preset smoke] [--steps 300]
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from tinyllm.config import get_config
from tinyllm.data import load_tokens
from tinyllm.model import Transformer

p = argparse.ArgumentParser()
p.add_argument("--preset", default="smoke")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--seq", type=int, default=256)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--target", type=float, default=0.05, help="pass if final loss is below this")
args = p.parse_args()

torch.manual_seed(0)
assert torch.cuda.is_available(), "needs the GPU"
dev = "cuda"

# One fixed batch of random windows from train.bin. y is x shifted by one token.
tokens = load_tokens("train")
rng = np.random.default_rng(0)
starts = rng.integers(0, len(tokens) - args.seq - 1, size=args.batch)
chunk = np.stack([tokens[s : s + args.seq + 1] for s in starts]).astype(np.int64)
batch = torch.from_numpy(chunk).to(dev)
x, y = batch[:, :-1], batch[:, 1:]

cfg = get_config(args.preset)
model = Transformer(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
print(f"preset={args.preset} params={model.num_params():,} batch={args.batch}x{args.seq} "
      f"lr={args.lr} steps={args.steps}")
print(f"ln(vocab) = {math.log(cfg.vocab_size):.3f}")

first_below = {}
t0 = time.time()
for step in range(1, args.steps + 1):
    with torch.autocast(dev, dtype=torch.bfloat16):
        _, loss = model(x, targets=y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    lv = loss.item()
    for thr in (1.0, 0.1, args.target):
        if lv < thr and thr not in first_below:
            first_below[thr] = step
    if step == 1 or step % 25 == 0:
        print(f"step {step:4d}  loss {lv:.4f}")
elapsed = time.time() - t0

print(f"\nfinal loss {lv:.4f} in {elapsed:.1f} s; "
      f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
for thr, s in sorted(first_below.items(), reverse=True):
    print(f"  first below {thr}: step {s}")
ok = lv < args.target
print("PASS" if ok else f"FAIL: final loss {lv:.4f} >= {args.target}")
sys.exit(0 if ok else 1)
