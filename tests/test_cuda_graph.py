"""Step 4.11: CUDA-graph decode (docs/INFERENCE.md). CUDA only.

- Graph replay is bit-identical to the same static step run eagerly.
- Against today's eager generate() (no mask, exact key count, another SDPA kernel) only
  bf16-close: the logit gap is bounded and printed. Greedy tokens must match on the tiny
  model (well-separated logits); on TinyLlama they may differ only at near-ties.
"""

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.cuda_graph import GraphDecoder, default_buckets
from tinyllm.engine import prefill_padded
from tinyllm.generate import generate
from tinyllm.kv_cache import KVCache
from tinyllm.model import Transformer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CFG = ModelConfig(vocab_size=128, max_seq_len=96, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                  ffn_hidden=160)
BUCKETS = (16, 32, 64, 96)          # small, so short runs cross several boundaries


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(CFG).eval()
    with torch.no_grad():           # larger weights: well-separated greedy tokens (see test_kv_cache)
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return m.cuda()


def prompt(b=2, t=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, CFG.vocab_size, (b, t), generator=g).cuda()


@torch.no_grad()
def eager_stepwise(model, ids, n_prefill, pad_lens=None):
    """Teacher-forced logits from the eager KVCache: prefill, then one token per step."""
    cache = KVCache(model.cfg.n_layers, ids.shape[1], pad_lens=pad_lens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = [model(ids[:, :n_prefill], cache=cache)[0][:, -1:]]
        for t in range(n_prefill, ids.shape[1] - 1):
            out.append(model(ids[:, t:t + 1], cache=cache)[0])
    return torch.cat(out, 1).float()


@torch.no_grad()
def graph_stepwise(dec, ids, n_prefill, pad_lens=None):
    out = [dec.prefill(ids[:, :n_prefill], pad_lens)[:, -1:].float()]
    for t in range(n_prefill, ids.shape[1] - 1):
        out.append(dec.step(ids[:, t:t + 1]).float())       # .float() copies out of the static buffer
    return torch.cat(out, 1)


def bf16_gap(ref, x):
    err, scale = (x - ref).abs().max().item(), ref.abs().max().item()
    print(f"logit gap vs eager: max abs {err:.4g} (scale {scale:.3g})")
    return err, scale


def test_default_buckets():
    assert default_buckets(512) == (128, 256, 512)
    assert default_buckets(2048) == (128, 256, 512, 1024, 2048)
    assert default_buckets(300) == (128, 256, 300)
    assert default_buckets(100) == (100,)


@torch.no_grad()
def test_replay_is_bit_identical_to_static_eager_step(model):
    ids = prompt(b=3)
    graph = GraphDecoder(model, 3, buckets=BUCKETS)
    eager = GraphDecoder(model, 3, buckets=BUCKETS, use_graph=False)
    assert torch.equal(graph.prefill(ids), eager.prefill(ids))
    tok = ids[:, -1:]
    for _ in range(CFG.max_seq_len - ids.shape[1]):
        a, b = graph.step(tok), eager.step(tok)
        assert torch.equal(a, b)
        tok = b[:, -1].argmax(-1, keepdim=True)
    assert sorted(graph.graphs) == list(BUCKETS)               # every bucket captured once
    assert eager.graphs == {}


def test_greedy_matches_eager_generate_across_buckets(model):
    ids = prompt()
    new = CFG.max_seq_len - ids.shape[1]                       # to the last position: all buckets
    want = generate(model, ids, new, temperature=0)
    dec = GraphDecoder(model, 2, buckets=BUCKETS)
    assert torch.equal(generate(model, ids, new, temperature=0, cuda_graph=dec), want)
    ref, got = eager_stepwise(model, want, ids.shape[1]), graph_stepwise(dec, want, ids.shape[1])
    err, scale = bf16_gap(ref, got)
    assert err <= 0.02 * scale


def test_cuda_graph_true_uses_default_buckets(model):
    ids = prompt(seed=3)
    assert torch.equal(generate(model, ids, 40, temperature=0, cuda_graph=True),
                       generate(model, ids, 40, temperature=0))


def test_padded_batch_matches_eager_padded(model):
    prompts = [prompt(1, n, seed=n)[0].tolist() for n in (5, 12, 9)]
    new = 60
    logits, cache, pad = prefill_padded(model, prompts, 12 + new)
    want = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(new):
            nxt = logits.float().argmax(-1, keepdim=True)
            want.append(nxt)
            logits = model(nxt, cache=cache)[0][:, -1]
    want = torch.cat(want, 1)

    pad_t = torch.tensor(pad, device="cuda")
    idx = torch.tensor([[0] * n + p for n, p in zip(pad, prompts)], device="cuda")
    dec = GraphDecoder(model, 3, buckets=BUCKETS)
    logits, got = dec.prefill(idx, pad_t)[:, -1], []
    for _ in range(new):
        nxt = logits.float().argmax(-1, keepdim=True)
        got.append(nxt)
        logits = dec.step(nxt)[:, -1]
    assert torch.equal(torch.cat(got, 1), want)

    seq = torch.cat([idx, want], 1)
    err, scale = bf16_gap(eager_stepwise(model, seq, 12, pad_t), graph_stepwise(dec, seq, 12, pad_t))
    assert err <= 0.02 * scale


def test_eos_and_decoder_reuse(model):
    dec = GraphDecoder(model, 2, buckets=BUCKETS)
    ids = prompt(seed=5)
    free = generate(model, ids, 50, temperature=0)
    eos = free[0, ids.shape[1] + 7].item()                     # a token row 0 emits at step 8
    want = generate(model, ids, 50, temperature=0, eos_id=eos)
    assert torch.equal(generate(model, ids, 50, temperature=0, eos_id=eos, cuda_graph=dec), want)
    captured = set(dec.graphs)
    other = prompt(seed=6)                                     # same decoder, a new request
    assert torch.equal(generate(model, other, 30, temperature=0, cuda_graph=dec),
                       generate(model, other, 30, temperature=0))
    assert set(dec.graphs) >= captured


def test_sampling_matches_eager_with_same_generator(model):
    ids = prompt(seed=7)
    kw = dict(temperature=0.8, top_k=20)
    want = generate(model, ids, 40, generator=torch.Generator("cuda").manual_seed(0), **kw)
    got = generate(model, ids, 40, generator=torch.Generator("cuda").manual_seed(0), cuda_graph=True, **kw)
    assert torch.equal(got, want)


@torch.no_grad()
def test_full_cache_and_wrong_batch_raise(model):
    dec = GraphDecoder(model, 2, buckets=BUCKETS)
    with pytest.raises(ValueError, match="batch"):
        dec.prefill(prompt(b=3))
    dec.prefill(prompt(t=CFG.max_seq_len - 1))
    dec.step(prompt(t=1))
    with pytest.raises(ValueError, match="full"):
        dec.step(prompt(t=1))
    with pytest.raises(ValueError, match="bucket"):
        GraphDecoder(model, 1, buckets=(16, 32))


# --- the real TinyLlama ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tinyllama():
    from tinyllm import hf_llama as hl
    try:
        path = hl.download(local_only=True)
    except Exception:
        pytest.skip("TinyLlama not in the HF cache (run scripts/tinyllama_smoke.py once)")
    return hl.load_from_dir(path), hl.HFTokenizer(path)


def test_tinyllama_graph_greedy_matches_eager_up_to_near_ties(tinyllama):
    """At 1.1B the masked-window SDPA kernel shifts some logits by a bf16 ulp, enough to flip a
    near-tie (measured: "haiku" step 0, eager margin 0.125 = one ulp at |logit| 16-32, median
    margin 1.19). So: replay == static eager exactly; vs eager generate, every top-1 flip on the
    same tokens is a near-tie, and free-running greedy agrees until such a flip."""
    model, tok = tinyllama
    dec = GraphDecoder(model, 1, max_len=512)
    static = GraphDecoder(model, 1, max_len=512, use_graph=False)
    for q in ["What is a KV cache?", "Write a haiku about the sea.", "Name three prime numbers."]:
        ids = torch.tensor([tok.encode(tok.chat_prompt(q))], device="cuda")
        n = ids.shape[1]
        want = generate(model, ids, 64, temperature=0)
        got = generate(model, ids, 64, temperature=0, cuda_graph=dec)
        ref, g = eager_stepwise(model, want, n)[0], graph_stepwise(dec, want, n)
        assert torch.equal(g, graph_stepwise(static, want, n))
        err, scale = bf16_gap(ref, g[0])
        assert err <= 0.02 * scale
        top2 = ref.topk(2, -1).values
        flips = (g[0].argmax(-1) != ref.argmax(-1)).nonzero().flatten()
        assert ((top2[flips, 0] - top2[flips, 1]) <= 0.25).all(), q      # <= 2 bf16 ulps
        diff = (got[0] != want[0]).nonzero().flatten()
        print(f"{q!r}: {len(flips)} flips, greedy first diff "
              f"{diff[0].item() - n if len(diff) else None}")
        if len(diff):
            assert diff[0].item() - n in flips.tolist()
