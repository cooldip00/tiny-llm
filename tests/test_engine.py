"""Step 4.4: static and continuous batching must reproduce each request's solo generate()."""

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.engine import Request, prefill_padded, run_continuous, run_static
from tinyllm.generate import generate
from tinyllm.kv_cache import KVCache, SlotKVCache
from tinyllm.model import Transformer

CFG = ModelConfig(vocab_size=128, max_seq_len=48, d_model=64, n_layers=3, n_heads=4, ffn_hidden=160)
TOL = dict(atol=1e-4, rtol=1e-4)


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(CFG).eval()
    with torch.no_grad():                         # well-separated logits (see test_kv_cache)
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return m


def make_requests(n=7, seed=1, eos_id=None):
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for _ in range(n):
        t = int(torch.randint(2, 14, (1,), generator=g))
        new = int(torch.randint(1, 16, (1,), generator=g))
        reqs.append(Request(torch.randint(0, CFG.vocab_size, (t,), generator=g).tolist(), new, eos_id))
    return reqs


def solo(model, r: Request) -> list[int]:
    out = generate(model, torch.tensor([r.prompt]), r.max_new_tokens, temperature=0.0, eos_id=r.eos_id)
    return out[0, len(r.prompt):].tolist()


@torch.no_grad()
def test_padded_prefill_matches_solo(model):
    reqs = make_requests()
    logits, cache, pad = prefill_padded(model, [r.prompt for r in reqs], max_len=CFG.max_seq_len)
    assert cache.pos == max(len(r.prompt) for r in reqs)
    for j, r in enumerate(reqs):
        ref = model(torch.tensor([r.prompt]))[0][0, -1]
        torch.testing.assert_close(logits[j], ref, **TOL)
        assert pad[j] == cache.pos - len(r.prompt)


@torch.no_grad()
def test_slot_decode_matches_solo(model):
    """Two sequences join at different times; every decode logit equals the full forward."""
    a = torch.randint(0, CFG.vocab_size, (20,), generator=torch.Generator().manual_seed(2)).tolist()
    b = torch.randint(0, CFG.vocab_size, (20,), generator=torch.Generator().manual_seed(3)).tolist()
    slots = SlotKVCache(CFG.n_layers, 3, CFG.n_kv_heads, CFG.head_dim, CFG.max_seq_len,
                        torch.float32, "cpu")
    seqs = {0: a[:9], 2: None}                                # slot 1 stays free throughout
    _, pre, pad = prefill_padded(model, [seqs[0]], max_len=9)
    slots.admit([0], pre, pad)
    for step in range(10):
        if step == 3:                                          # b joins mid-flight in slot 2
            seqs[2] = b[:4]
            _, pre, pad = prefill_padded(model, [seqs[2]], max_len=4)
            slots.admit([2], pre, pad)
        src = {0: a, 2: b}
        nxt = torch.zeros(3, 1, dtype=torch.long)
        for s, seq in seqs.items():
            if seq is not None:
                nxt[s, 0] = src[s][len(seq)]
        logits = model(nxt, cache=slots)[0][:, -1]
        for s, seq in seqs.items():
            if seq is not None:
                seq.append(src[s][len(seq)])
                torch.testing.assert_close(logits[s], model(torch.tensor([seq]))[0][0, -1], **TOL)
    assert slots.lens == [19, 0, 11]


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_static_matches_solo(model, batch_size):
    reqs = make_requests()
    results, stats = run_static(model, reqs, batch_size)
    assert [r.tokens for r in results] == [solo(model, r) for r in reqs]
    assert stats["useful_tokens"] == sum(r.max_new_tokens for r in reqs)


@pytest.mark.parametrize("n_slots", [1, 3, 10])
def test_continuous_matches_solo(model, n_slots):
    reqs = make_requests()
    results, stats = run_continuous(model, reqs, n_slots)
    assert [r.tokens for r in results] == [solo(model, r) for r in reqs]
    assert stats["useful_tokens"] == sum(r.max_new_tokens for r in reqs)


def test_eos_frees_slot_for_next_request(model):
    base = make_requests(n=3, seed=4)
    for r in base:
        r.max_new_tokens = 15
    eos = solo(model, base[0])[2]                              # request 0 hits EOS by token 3
    reqs = [Request(r.prompt, r.max_new_tokens, eos) for r in base]
    want = [solo(model, r) for r in reqs]
    assert len(want[0]) <= 3 and want[0][-1] == eos
    results, _ = run_continuous(model, reqs, n_slots=2)
    assert [r.tokens for r in results] == want
    if len(want[1]) > len(want[0]):                            # request 2 took slot 0 while 1 still ran
        assert results[2].ttft_s < results[1].latency_s
    static, _ = run_static(model, reqs, batch_size=2)
    assert [r.tokens for r in static] == want


def test_utilisation_stats(model):
    reqs = [Request([1, 2, 3], 10), Request([4, 5], 2)]
    _, s = run_static(model, reqs, batch_size=2)
    assert s["decode_steps"] == 9                              # the longest row sets the pace
    assert s["row_util"] == pytest.approx((9 + 1) / 18)        # the short row is live for 1 step
    _, c = run_continuous(model, reqs, n_slots=1)              # one slot, always busy
    assert c["row_util"] == 1.0 and c["decode_steps"] == 9 + 1
    assert 0 < c["kv_used_peak"] <= 1


@pytest.mark.parametrize("kv", ["slots", "paged"])
def test_requests_that_only_fit_alone_are_not_batched(model, kv):
    # Each fits max_len 48 alone (40 + 5, 2 + 40); padded together they'd need 40 + 40.
    # (Found in 4.5: a preempted request returns with a longer context.)
    reqs = [Request(list(range(1, 41)), 5), Request([3, 4], 40)]
    results, _ = run_continuous(model, reqs, n_slots=2, kv=kv, block_size=4)
    assert [r.tokens for r in results] == [solo(model, r) for r in reqs]


def test_too_long_raises(model):
    long = [Request([1] * 40, 10)]
    with pytest.raises(ValueError, match="max_seq_len"):
        run_static(model, long, batch_size=1)
    with pytest.raises(ValueError, match="slot max_len"):
        run_continuous(model, long, n_slots=1)


def test_slot_cache_rejects_multi_token_step(model):
    slots = SlotKVCache(CFG.n_layers, 2, CFG.n_kv_heads, CFG.head_dim, 8, torch.float32, "cpu")
    with pytest.raises(ValueError, match="one token"):
        model(torch.zeros(2, 3, dtype=torch.long), cache=slots)
    with pytest.raises(ValueError, match="> slot max_len"):
        cache = KVCache(CFG.n_layers, 16)
        model(torch.zeros(1, 9, dtype=torch.long), cache=cache)
        slots.admit([0], cache, [0])
