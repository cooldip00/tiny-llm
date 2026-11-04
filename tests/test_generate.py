"""Step 4.2: sampling (temperature, top-k, top-p) on hand-made logits."""

import pytest
import torch

from tinyllm.config import ModelConfig
from tinyllm.generate import sample, sample_next
from tinyllm.model import Transformer

TINY = ModelConfig(vocab_size=64, max_seq_len=32, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64)
# One row with known probabilities, shuffled so "most likely" is not just index 0.
PROBS = torch.tensor([[0.05, 0.5, 0.15, 0.3]])  # sorted: 1 (0.5), 3 (0.3), 2 (0.15), 0 (0.05)
LOGITS = PROBS.log()


def draws(n=4000, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    logits = LOGITS.expand(n, -1)
    return sample_next(logits, generator=g, **kw).squeeze(-1)


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    assert sample_next(logits, temperature=0).tolist() == [[1], [0]]
    assert sample_next(logits, temperature=0, top_k=2, top_p=0.1).tolist() == [[1], [0]]


def test_shape_is_batch_by_one():
    assert sample_next(torch.randn(5, 11)).shape == (5, 1)


def test_top_p_keeps_smallest_nucleus():
    # mass before each sorted token: 0, 0.5, 0.8, 0.95 → top_p=0.7 keeps tokens 1 and 3
    out = draws(top_p=0.7)
    assert set(out.tolist()) == {1, 3}
    frac = (out == 1).float().mean().item()
    assert frac == pytest.approx(0.5 / 0.8, abs=0.03)    # renormalised: 0.625 / 0.375


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    assert set(draws(top_p=0.49).tolist()) == {1}        # token 1 alone covers 0.49
    assert set(draws(top_p=0.51).tolist()) == {1, 3}     # 0.5 < 0.51, so the next token joins


def test_tiny_top_p_equals_greedy():
    assert set(draws(top_p=1e-6).tolist()) == {1}


def test_top_p_one_is_unfiltered():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    logits = LOGITS.expand(200, -1)
    assert torch.equal(sample_next(logits, top_p=1.0, generator=g1),
                       sample_next(logits, generator=g2))


def test_top_k_then_top_p():
    # top_k=3 drops token 0 → probs renormalise to 0.526, 0.316, 0.158; top_p=0.8 keeps 1 and 3
    assert set(draws(top_k=3, top_p=0.8).tolist()) == {1, 3}
    assert set(draws(top_k=3).tolist()) == {1, 2, 3}


def test_temperature_applies_before_top_p():
    # high temperature flattens the distribution, so the 0.7 nucleus needs 3 tokens
    assert set(draws(temperature=10.0, top_p=0.7).tolist()) == {1, 2, 3}


def test_rows_are_filtered_independently():
    logits = torch.stack([LOGITS[0], LOGITS[0].flip(-1)])  # row 2's best token is index 2
    g = torch.Generator().manual_seed(0)
    out = sample_next(logits.repeat(500, 1), top_p=0.5, generator=g).view(500, 2)
    assert set(out[:, 0].tolist()) == {1} and set(out[:, 1].tolist()) == {2}


def test_sample_with_tiny_top_p_equals_greedy():
    with torch.random.fork_rng():                        # don't disturb other tests' global RNG
        torch.manual_seed(0)
        model = Transformer(TINY)
    prompt = torch.tensor([[5, 6]])
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample(model, prompt, 8, temperature=1.0, top_p=1e-6, generator=g),
                       sample(model, prompt, 8, temperature=0))


@pytest.mark.parametrize("kw", [{"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}])
def test_invalid_arguments_raise(kw):
    with pytest.raises(ValueError):
        sample_next(LOGITS, **kw)
