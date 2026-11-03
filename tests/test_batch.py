import numpy as np
import pytest
import torch

from tinyllm.data import TOKEN_DTYPE, bin_path, get_batch, load_tokens

# Small fake corpus: token i has value i, so every window is easy to check.
TOKENS = np.arange(1000, dtype=TOKEN_DTYPE)


def test_shapes_and_dtype():
    x, y = get_batch(TOKENS, 4, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == torch.int64


def test_windows_are_contiguous_and_y_is_shifted():
    x, y = get_batch(TOKENS, 8, 32, np.random.default_rng(0))
    for row_x, row_y in zip(x, y):
        s = row_x[0].item()
        assert torch.equal(row_x, torch.arange(s, s + 32))
        assert torch.equal(row_y, torch.arange(s + 1, s + 33))


def test_same_seed_same_batch_and_state_resumes():
    a = get_batch(TOKENS, 4, 16, np.random.default_rng(42))
    b = get_batch(TOKENS, 4, 16, np.random.default_rng(42))
    assert torch.equal(a[0], b[0])

    rng = np.random.default_rng(7)
    get_batch(TOKENS, 4, 16, rng)
    saved = rng.bit_generator.state            # what a checkpoint stores
    expected = get_batch(TOKENS, 4, 16, rng)
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = saved
    assert torch.equal(get_batch(TOKENS, 4, 16, rng2)[0], expected[0])


def test_covers_the_ends_without_going_out_of_range():
    # T = len-1 leaves exactly one valid start (0): the whole array.
    x, y = get_batch(TOKENS, 3, 999, np.random.default_rng(0))
    assert x[:, 0].eq(0).all() and y[:, -1].eq(999).all()
    # Many small windows: the last token is reachable as a target.
    _, y = get_batch(TOKENS, 20_000, 8, np.random.default_rng(0))
    assert y.max().item() == 999 and y.min().item() >= 1


def test_too_short_raises():
    with pytest.raises(ValueError):
        get_batch(TOKENS[:10], 1, 10, np.random.default_rng(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_device():
    x, y = get_batch(TOKENS, 4, 16, np.random.default_rng(0), device="cuda")
    assert x.is_cuda and y.is_cuda
    assert torch.equal(y[:, :-1], x[:, 1:])


@pytest.mark.skipif(not bin_path("train").exists(), reason="run scripts/tokenize_data.py first")
def test_real_train_memmap():
    x, y = get_batch(load_tokens("train"), 4, 512, np.random.default_rng(0))
    assert x.shape == (4, 512) and int(x.max()) < 8192
    assert torch.equal(y[:, :-1], x[:, 1:])
