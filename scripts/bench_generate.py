"""Step 4.1: benchmark naive generation (no KV cache). Baseline for every later Phase 4 step.

Naive = each new token runs a full forward over the whole context (sliding window of
max_seq_len), so step t costs O(t^2) attention and O(t) matmul work. Greedy only.

Protocol (docs/INFERENCE.md): prompts 32/128/512 tokens, generate 256, batch 1/8/32,
3 warmup runs, torch.cuda.synchronize around timers, median of 5 timed runs.
  prefill tok/s = B * prompt_len / TTFT     (TTFT = first forward + first argmax)
  decode  tok/s = B * (new_tokens - 1) / time after the first token
  peak VRAM     = torch.cuda.max_memory_allocated (weights + activations)

Prompts are real validation text (fixed offsets), so the outputs can serve as the
greedy-identical reference for 4.2+. The timed loop is checked against
tinyllm.generate.sample(temperature=0) before any timing.

    uv run python scripts/bench_generate.py [--engine naive|kv] [--ckpt checkpoints/main/best.pt]

Step 4.3 adds --engine kv (KV cache: one prefill, then one token per forward). It skips
rows where prompt + new tokens exceed max_seq_len, since a cache can't slide the window.

Step 4.11 adds --engine kv-graph (each decode step one CUDA-graph replay, tinyllm.cuda_graph)
and --dtype bf16 (weights stored bf16, D-023; output runs/bench/<engine>_bf16.json).

Step 4.9d adds --model tinyllama (TinyLlama 1.1B, bf16 as published; output
runs/bench/tinyllama/<engine>.json) and --engine hf (HF Transformers generate, the baseline,
D-015). TinyLlama prompts are the same validation text, re-tokenized with its own tokenizer.
Every row also reports the bandwidth ceiling B * 423 GB/s / stored weight bytes and the % of it.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from tinyllm.config import ModelConfig
from tinyllm.cuda_graph import GraphDecoder
from tinyllm.data import DATA_DIR, TOKEN_DTYPE, load_tokens
from tinyllm.generate import generate, sample
from tinyllm.kv_cache import KVCache
from tinyllm.model import RotaryEmbedding, Transformer
from tinyllm.provenance import Provenance
from tinyllm.train import load_checkpoint

BANDWIDTH = 423e9           # bytes/s, Phase 0.5 measurement (docs/HARDWARE.md)
TINYLLAMA_VAL = DATA_DIR / "validation_tinyllama.bin"   # git-ignored, like data/*.bin

_VAL: np.ndarray | None = None     # prompt source set by use_model(); None = our validation split


def load_model(path: Path, device: str) -> Transformer:
    ckpt = load_checkpoint(path, device)
    model = Transformer(ModelConfig.from_dict(ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def tinyllama_val_tokens() -> np.ndarray:
    """Our validation split as text (decoded story by story), re-tokenized with TinyLlama's
    tokenizer, stories joined by </s> as ours are by EOS. Built once, then cached in data/."""
    if not TINYLLAMA_VAL.exists():
        from tinyllm import hf_llama as hl
        from tinyllm.tokenizer import load_tokenizer
        val = load_tokens("validation").astype(np.int64)
        stories = [s[s != 0].tolist() for s in np.split(val, np.flatnonzero(val == 0))]   # 0 = our EOS
        texts = [t for t in load_tokenizer().decode_batch(stories) if t]
        tok = hl.HFTokenizer(hl.download(local_only=True))
        enc = tok.tok.encode_batch(texts, add_special_tokens=False)
        ids = np.concatenate([np.asarray(e.ids + [tok.eos_id]) for e in enc])
        assert ids.max() < np.iinfo(TOKEN_DTYPE).max
        tmp = TINYLLAMA_VAL.with_suffix(".tmp")
        ids.astype(TOKEN_DTYPE).tofile(tmp)
        tmp.replace(TINYLLAMA_VAL)
        print(f"wrote {TINYLLAMA_VAL}: {len(texts)} stories, {len(ids)} TinyLlama tokens")
    return np.memmap(TINYLLAMA_VAL, dtype=TOKEN_DTYPE, mode="r")


def val_tokens() -> np.ndarray:
    return load_tokens("validation") if _VAL is None else _VAL


def use_model(spec: str, device: str, dtype: str = "fp32") -> Transformer:
    """'tinyllama' (bf16 weights as published) or a checkpoint path. Also points make_prompts()
    at validation text in that model's tokenizer."""
    global _VAL
    if spec == "tinyllama":
        from tinyllm import hf_llama as hl
        _VAL = tinyllama_val_tokens()
        return hl.load_from_dir(hl.download(local_only=True), device)
    _VAL = None
    model = load_model(Path(spec), device)
    if dtype == "bf16":
        model.to(torch.bfloat16)
        cfg = model.cfg       # keep the RoPE tables fp32, as hf_llama.load_from_dir does
        model.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta).to(device)
    return model


def stored_weight_bytes(model) -> int:
    """Parameters + persistent buffers as stored (Int8Linear: int8 weights + fp32 scales)."""
    bufs = [b for m in model.modules() for n, b in m.named_buffers(recurse=False)
            if n not in m._non_persistent_buffers_set]
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + bufs)


def make_prompts(batch: int, prompt_len: int, device: str) -> torch.Tensor:
    """`batch` windows of validation tokens at fixed, evenly spaced offsets."""
    val = val_tokens()
    starts = np.linspace(0, len(val) - prompt_len - 1, batch, dtype=np.int64)
    rows = np.stack([val[s:s + prompt_len].astype(np.int64) for s in starts])
    return torch.from_numpy(rows).to(device)


@torch.no_grad()
def naive_greedy(model: Transformer, prompt: torch.Tensor, new_tokens: int):
    """Same computation as sample(temperature=0), with a timer after the first token."""
    max_len = model.cfg.max_seq_len
    idx = prompt
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_first = t0
    for i in range(new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx[:, -max_len:])
        nxt = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if i == 0:
            torch.cuda.synchronize()
            t_first = time.perf_counter()
    torch.cuda.synchronize()
    return idx, t_first - t0, time.perf_counter() - t_first


@torch.no_grad()
def kv_greedy(model: Transformer, prompt: torch.Tensor, new_tokens: int):
    """Same computation as generate(temperature=0), timed the same way as naive_greedy."""
    cache = KVCache(model.cfg.n_layers, model.cfg.max_seq_len)
    out = [prompt]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_first = t0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(prompt, cache=cache)
    for i in range(new_tokens):
        nxt = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
        out.append(nxt)
        if i == 0:
            torch.cuda.synchronize()
            t_first = time.perf_counter()
        if i + 1 == new_tokens:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(nxt, cache=cache)
    torch.cuda.synchronize()
    return torch.cat(out, dim=1), t_first - t0, time.perf_counter() - t_first


_DECODERS: dict = {}


@torch.no_grad()
def kv_graph_greedy(model: Transformer, prompt: torch.Tensor, new_tokens: int):
    """Same computation as generate(temperature=0, cuda_graph=dec), timed like kv_greedy.
    One GraphDecoder per batch size is kept across runs, so the warm-up runs capture the
    graphs and the timed runs only replay them."""
    key = (id(model), prompt.shape[0])
    if key not in _DECODERS:
        # One live decoder at a time: each holds a static KV cache of max_seq_len positions
        # (TinyLlama B=32: 1.4 GiB), and stale ones pushed an 8 GB card out of memory (4.9d).
        _DECODERS.clear()
        torch.cuda.empty_cache()
        _DECODERS[key] = GraphDecoder(model, prompt.shape[0])
    dec = _DECODERS[key]
    out = [prompt]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_first = t0
    logits = dec.prefill(prompt)
    for i in range(new_tokens):
        nxt = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
        out.append(nxt)
        if i == 0:
            torch.cuda.synchronize()
            t_first = time.perf_counter()
        if i + 1 == new_tokens:
            break
        logits = dec.step(nxt)
    torch.cuda.synchronize()
    return torch.cat(out, dim=1), t_first - t0, time.perf_counter() - t_first


def load_hf_baseline(device: str):
    """HF LlamaForCausalLM on the same TinyLlama files: bf16, SDPA attention (benchmark only, D-015)."""
    from transformers import LlamaForCausalLM
    from tinyllm import hf_llama as hl
    hf = LlamaForCausalLM.from_pretrained(
        hl.download(local_only=True), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    hf.generation_config.max_length = None      # max_new_tokens decides; silences a per-call warning
    return hf


@torch.no_grad()
def hf_greedy(hf, prompt: torch.Tensor, new_tokens: int):
    """HF generate(), greedy, never stopping early (min_new_tokens), default (dynamic) KV cache.
    TTFT is a separate 1-token generate(); decode time = the full call's time - TTFT."""
    kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
              attention_mask=torch.ones_like(prompt), pad_token_id=hf.config.eos_token_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hf.generate(prompt, max_new_tokens=1, min_new_tokens=1, **kw)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = hf.generate(prompt, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **kw)
    torch.cuda.synchronize()
    return out, ttft, time.perf_counter() - t0 - ttft


ENGINES = {"naive": naive_greedy, "kv": kv_greedy, "kv-graph": kv_graph_greedy, "hf": hf_greedy}


def kv_vs_naive(model, device) -> str:
    """bf16 reference check (INFERENCE.md rule): greedy tokens, then logits, against naive."""
    prompts = make_prompts(8, 128, device)
    kv = kv_greedy(model, prompts, 256)[0][:, 128:]
    naive = naive_greedy(model, prompts, 256)[0][:, 128:]
    same = (kv == naive)
    first_diff = [int((~row).nonzero()[0]) if (~row).any() else None for row in same]
    # Logits on naive's own tokens, so a single flipped token doesn't cascade.
    seq = naive_greedy(model, prompts, 256)[0]
    cache = KVCache(model.cfg.n_layers, model.cfg.max_seq_len)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(seq)[0][:, 127:-1].float()
        cached = [model(seq[:, :128], cache=cache)[0][:, -1:]]
        cached += [model(seq[:, t:t + 1], cache=cache)[0] for t in range(128, seq.shape[1] - 1)]
    cached = torch.cat(cached, dim=1).float()
    err = (cached - full).abs().max().item()
    argmax_agree = (cached.argmax(-1) == full.argmax(-1)).float().mean().item()
    return (f"kv vs naive (8x128 prompts, 256 tokens, bf16): tokens identical {same.all(1).sum()}/8 rows, "
            f"{same.float().mean():.2%} of tokens; first divergence per row {first_diff}; "
            f"teacher-forced max |logit diff| {err:.3f} (logit scale {full.abs().max():.1f}), "
            f"argmax agreement {argmax_agree:.2%}")


def graph_vs_kv(model, device) -> str:
    """Greedy tokens of kv-graph vs eager kv and naive; then, teacher-forced on eager kv's own
    tokens, the logit gap and how close to a tie every top-1 flip is (eager's top-2 margin)."""
    prompts = make_prompts(8, 128, device)
    graph = kv_graph_greedy(model, prompts, 256)[0]
    seq = kv_greedy(model, prompts, 256)[0]
    naive = naive_greedy(model, prompts, 256)[0]

    def first(a, b):
        same = a[:, 128:] == b[:, 128:]
        return int(same.all(1).sum()), [int((~r).nonzero()[0]) if (~r).any() else None for r in same]
    cache = KVCache(model.cfg.n_layers, model.cfg.max_seq_len)
    dec = GraphDecoder(model, 8)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = [model(seq[:, :128], cache=cache)[0][:, -1:]]
        eager += [model(seq[:, t:t + 1], cache=cache)[0] for t in range(128, seq.shape[1] - 1)]
    g = [dec.prefill(seq[:, :128])[:, -1:].float()]
    g += [dec.step(seq[:, t:t + 1]).float() for t in range(128, seq.shape[1] - 1)]
    eager, g = torch.cat(eager, 1).float(), torch.cat(g, 1)
    top2 = eager.topk(2, -1).values
    margin = top2[..., 0] - top2[..., 1]
    flips = g.argmax(-1) != eager.argmax(-1)
    n_kv, d_kv = first(graph, seq)
    n_nv, d_nv = first(graph, naive)
    return (f"kv-graph vs eager kv (8x128 prompts, 256 tokens): identical {n_kv}/8 rows, first divergence "
            f"{d_kv}; vs naive: {n_nv}/8, {d_nv}. Teacher-forced: max |logit diff| "
            f"{(g - eager).abs().max():.3f} (scale {eager.abs().max():.1f}), argmax agreement "
            f"{1 - flips.float().mean():.2%}, {int(flips.sum())} flips, eager margin at flips max "
            f"{margin[flips].max() if flips.any() else 0:.3f} (median margin {margin.median():.3f})")


def bench(model, engine, batch, prompt_len, new_tokens, warmup, repeats, device) -> dict:
    run = ENGINES[engine]
    max_len = model.cfg.max_seq_len if hasattr(model, "cfg") else model.config.max_position_embeddings
    if engine != "naive" and prompt_len + new_tokens > max_len:
        return {"batch": batch, "prompt_len": prompt_len, "new_tokens": new_tokens,
                "skipped": "prompt + new exceeds max_seq_len (no sliding window with a cache)"}
    prompt = make_prompts(batch, prompt_len, device)
    # Drop cached blocks left by earlier rows / checks: on WSL an over-reserved 8 GB card does not
    # OOM, the driver spills to host memory and the row runs ~10x slower (4.9d, B=32 P=512).
    torch.cuda.empty_cache()
    for _ in range(warmup):
        run(model, prompt, new_tokens)
    torch.cuda.reset_peak_memory_stats()
    ttfts, decodes = [], []
    for _ in range(repeats):
        _, ttft, dec = run(model, prompt, new_tokens)
        ttfts.append(ttft)
        decodes.append(dec)
    ttft, dec = statistics.median(ttfts), statistics.median(decodes)
    ceiling = batch * BANDWIDTH / stored_weight_bytes(model)     # weight reads only, KV reads ignored
    decode_tok_s = batch * (new_tokens - 1) / dec
    return {
        "batch": batch, "prompt_len": prompt_len, "new_tokens": new_tokens,
        "ttft_ms": ttft * 1e3,
        "prefill_tok_s": batch * prompt_len / ttft,
        "decode_tok_s": decode_tok_s,
        "ms_per_step": dec / (new_tokens - 1) * 1e3,
        "ceiling_tok_s": ceiling,
        "pct_of_ceiling": 100 * decode_tok_s / ceiling,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "window_slides": prompt_len + new_tokens > max_len,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/main/best.pt")
    p.add_argument("--model", default=None, help="'tinyllama' instead of --ckpt (4.9d)")
    p.add_argument("--prompt-lens", default="32,128,512")
    p.add_argument("--batch-sizes", default=None,
                   help="default 1,8,32 (naive on TinyLlama: 1; a full forward per token at 1.1B)")
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--engine", choices=sorted(ENGINES), default="naive")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                   help="weight storage; both run under bf16 autocast. bf16 is D-023's inference layout")
    p.add_argument("--out", default=None,
                   help="default: runs/bench/<engine>[_bf16].json, runs/bench/tinyllama/<engine>.json")
    args = p.parse_args()
    tiny = args.model == "tinyllama"
    if args.model not in (None, "tinyllama"):
        p.error("--model takes 'tinyllama'; use --ckpt for a checkpoint")
    if args.engine == "hf" and not tiny:
        p.error("--engine hf needs --model tinyllama")
    if tiny:
        args.dtype = "bf16"   # as published; the fp32 copy (4.4 GB) is not an inference layout (D-023)
    args.batch_sizes = args.batch_sizes or ("1" if tiny and args.engine == "naive" else "1,8,32")
    args.out = args.out or (f"runs/bench/tinyllama/{args.engine}.json" if tiny else
                            f"runs/bench/{args.engine}{'_bf16' if args.dtype == 'bf16' else ''}.json")
    device = "cuda"

    model = use_model("tinyllama" if tiny else args.ckpt, device, args.dtype)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | "
          f"{n_params / 1e6:.1f}M params | max_seq_len {model.cfg.max_seq_len} | "
          f"{stored_weight_bytes(model) / 2**20:.0f} MiB weights, ceiling "
          f"{BANDWIDTH / stored_weight_bytes(model):.0f} tok/s per row")

    # Correctness gate: the timed loop must reproduce the project's sampler exactly.
    check = make_prompts(2, 32, device)
    ours, _, _ = naive_greedy(model, check, 64)
    ref = sample(model, check, 64, temperature=0)
    assert torch.equal(ours, ref), "timed loop diverges from sample(temperature=0)"
    print("check: timed loop == sample(temperature=0) on 2x32 prompt, 64 tokens: OK")
    if args.engine == "kv":
        kv_check = kv_greedy(model, check, 64)[0]
        assert torch.equal(kv_check, generate(model, check, 64, temperature=0)), \
            "timed kv loop diverges from generate(temperature=0)"
        print("check: timed kv loop == generate(temperature=0): OK")
        print("check:", kv_vs_naive(model, device))
    if args.engine == "kv-graph":
        g_check = kv_graph_greedy(model, check, 64)[0]
        assert torch.equal(g_check, generate(model, check, 64, temperature=0, cuda_graph=True)), \
            "timed kv-graph loop diverges from generate(temperature=0, cuda_graph=True)"
        print("check: timed kv-graph loop == generate(temperature=0, cuda_graph=True): OK")
        print("check:", graph_vs_kv(model, device))
    if args.engine == "hf":
        # Not asserted: HF's kernels differ, so bf16 near-ties may flip (as in 4.9c). Teacher-forced
        # on our greedy tokens, every top-1 flip should sit at a near-tie of our logits.
        prompts = make_prompts(8, 128, device)
        ours = kv_greedy(model, prompts, 128)[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = model(ours)[0][:, 127:-1].float()
        top2 = ref.topk(2, -1).values
        margin, ref_arg = top2[..., 0] - top2[..., 1], ref.argmax(-1)
        del model, ref
        torch.cuda.empty_cache()
        model = load_hf_baseline(device)
        theirs = hf_greedy(model, prompts, 128)[0]
        with torch.no_grad():
            flips = model(ours).logits[:, 127:-1].float().argmax(-1) != ref_arg
        same = ours[:, 128:] == theirs[:, 128:]
        first = [int((~r).nonzero()[0]) if (~r).any() else None for r in same]
        print(f"check: HF generate vs our kv greedy (8x128 prompts, 128 tokens): identical "
              f"{int(same.all(1).sum())}/8 rows, {same.float().mean():.2%} of tokens, first divergence {first}; "
              f"teacher-forced argmax agreement {1 - flips.float().mean():.2%}, {int(flips.sum())} flips, "
              f"our margin at flips max {margin[flips].max() if flips.any() else 0:.3f} "
              f"(median margin {margin.median():.3f})")

    rows = []
    prov = Provenance(weights=f"{args.dtype} + bf16 autocast", script="bench_generate",
                      engine=args.engine, model="tinyllama" if tiny else args.ckpt).start()
    for b in map(int, args.batch_sizes.split(",")):
        for plen in map(int, args.prompt_lens.split(",")):
            r = bench(model, args.engine, b, plen, args.new_tokens, args.warmup, args.repeats, device)
            rows.append(r)
            if "skipped" in r:
                print(f"B={b:>2} P={plen:>3} | skipped: {r['skipped']}", flush=True)
                continue
            print(f"B={b:>2} P={plen:>3} | TTFT {r['ttft_ms']:7.2f} ms | "
                  f"prefill {r['prefill_tok_s']:9.0f} tok/s | decode {r['decode_tok_s']:7.0f} tok/s "
                  f"({r['ms_per_step']:.2f} ms/step, {r['pct_of_ceiling']:4.1f}% of ceiling) | "
                  f"peak {r['peak_vram_mib']:6.0f} MiB"
                  + (" | window slides" if r["window_slides"] else ""), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"engine": args.engine, "dtype": args.dtype,
                               "model": "tinyllama" if tiny else args.ckpt, "bandwidth_bytes_s": BANDWIDTH,
                               "weight_bytes": stored_weight_bytes(model), "gpu": torch.cuda.get_device_name(),
                               "torch": torch.__version__, "provenance": prov.finish(), "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
