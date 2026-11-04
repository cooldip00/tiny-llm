"""Step 4.5: paged KV cache: allocator, block-table decode, and engine runs equal to solo generate()."""

import pytest
import torch

from tinyllm.engine import Request, prefill_padded, run_continuous
from tinyllm.paged_kv import BlockAllocator, OutOfBlocks, PagedKVCache

from test_engine import CFG, TOL, make_requests, model, solo  # noqa: F401  (model is a fixture)


# -- allocator ------------------------------------------------------------
def test_allocator_alloc_free_reuse():
    a = BlockAllocator(4, block_size=16)
    got = [a.alloc() for _ in range(4)]
    assert got == [0, 1, 2, 3] and a.num_free == 0 and a.peak_used == 4
    with pytest.raises(OutOfBlocks):
        a.alloc()
    a.free([2, 0])
    assert a.num_free == 2
    assert {a.alloc(), a.alloc()} == {0, 2}                    # freed blocks are reused
    assert a.peak_used == 4


def test_allocator_rejects_double_free():
    a = BlockAllocator(2, block_size=16)
    b = a.alloc()
    a.free([b])
    with pytest.raises(ValueError, match="twice"):
        a.free([b])
    with pytest.raises(ValueError, match="twice"):
        a.free([1])                                            # never allocated


# -- cache ----------------------------------------------------------------
def paged(n_slots=3, n_blocks=40, block_size=4, max_len=CFG.max_seq_len):
    return PagedKVCache(CFG.n_layers, n_slots, CFG.n_kv_heads, CFG.head_dim, n_blocks,
                        block_size, max_len, torch.float32, "cpu")


@torch.no_grad()
def test_paged_decode_matches_solo(model):
    """Padded joint prefill (pad is not stored), a late joiner, block boundaries crossed."""
    g = torch.Generator().manual_seed(5)
    src = {s: torch.randint(0, CFG.vocab_size, (30,), generator=g).tolist() for s in (0, 1, 2)}
    cache = paged()
    seqs = {0: src[0][:9], 1: src[1][:3], 2: None}             # 0 and 1 share one padded prefill
    _, pre, pad = prefill_padded(model, [seqs[0], seqs[1]], max_len=9)
    cache.admit([0, 1], pre, pad)
    assert cache.lens[:2] == [9, 3] and cache.first[:2] == [0, 6]
    assert [len(t) for t in cache.tables[:2]] == [3, 1]        # 9 tokens → 3 blocks of 4; 3 → 1
    for step in range(12):
        if step == 5:
            seqs[2] = src[2][:6]
            _, pre, pad = prefill_padded(model, [seqs[2]], max_len=6)
            cache.admit([2], pre, pad)
        nxt = torch.zeros(3, 1, dtype=torch.long)
        for s, seq in seqs.items():
            if seq is not None:
                nxt[s, 0] = src[s][len(seq)]
        assert cache.reserve_step() == []
        logits = model(nxt, cache=cache)[0][:, -1]
        for s, seq in seqs.items():
            if seq is not None:
                seq.append(src[s][len(seq)])
                torch.testing.assert_close(logits[s], model(torch.tensor([seq]))[0][0, -1], **TOL)
    assert cache.lens == [21, 15, 13]
    assert [len(t) for t in cache.tables] == [6, 4, 4]         # ceil(n / 4) blocks each
    used = cache.alloc.num_used
    cache.release(1)
    assert cache.alloc.num_used == used - 4 and cache.tables[1] == []


def test_reserve_step_preempts_newest():
    cache = paged(n_slots=2, n_blocks=2, block_size=4)
    for s, n in ((0, 4), (1, 4)):                              # each fills exactly one block
        pre = type("Pre", (), {})()                            # minimal stand-in for a KVCache
        pre.pos = n
        pre.k = [torch.zeros(1, CFG.n_kv_heads, n, CFG.head_dim) for _ in range(CFG.n_layers)]
        pre.v = pre.k
        cache.admit([s], pre, [0])
    assert cache.alloc.num_free == 0
    assert cache.reserve_step() == [1]                         # both need a block; 1 is newer
    assert cache.active == [True, False] and cache.alloc.num_free == 1


# -- engine ---------------------------------------------------------------
@pytest.mark.parametrize("n_slots,block_size", [(1, 4), (3, 4), (3, 16), (10, 8)])
def test_paged_engine_matches_solo(model, n_slots, block_size):
    reqs = make_requests()
    results, stats = run_continuous(model, reqs, n_slots, kv="paged", block_size=block_size)
    assert [r.tokens for r in results] == [solo(model, r) for r in reqs]
    assert stats["preemptions"] == 0 and stats["kv_allocated_at_end"] == 0


def test_tiny_budget_preempts_and_still_matches(model):
    reqs = make_requests(n=8, seed=7)
    for r in reqs:
        r.max_new_tokens = 15
    # 12 blocks of 4 = 48 tokens: room for about two sequences at a time, wanted by up to 4.
    results, stats = run_continuous(model, reqs, n_slots=4, kv="paged", block_size=4, kv_blocks=12)
    assert stats["preemptions"] > 0
    assert [r.tokens for r in results] == [solo(model, r) for r in reqs]
    assert stats["kv_allocated_at_end"] == 0                   # every block returned


def test_paged_same_memory_holds_more_sequences(model):
    """The 4.5 claim at test scale: with the KV memory of 2 slots, paging runs more at once."""
    reqs = [Request([1, 2, 3, 4], 6) for _ in range(6)]        # 10 tokens each, max_len 48
    budget_blocks = 2 * CFG.max_seq_len // 4                   # = 2 slots × 48 positions
    _, s = run_continuous(model, reqs, n_slots=2)
    _, p = run_continuous(model, reqs, n_slots=6, kv="paged", block_size=4, kv_blocks=budget_blocks)
    assert s["peak_active"] == 2 and p["peak_active"] == 6
    assert p["kv_reserved_mib"] == s["kv_reserved_mib"]
    assert p["decode_steps"] < s["decode_steps"]


def test_request_too_big_for_budget_raises(model):
    with pytest.raises(ValueError, match="does not fit"):
        run_continuous(model, [Request(list(range(20)), 5)], n_slots=1, kv="paged",
                       block_size=4, kv_blocks=3)


def test_unknown_kv_kind_raises(model):
    with pytest.raises(ValueError, match="kv must be"):
        run_continuous(model, make_requests(n=1), n_slots=1, kv="nope")
