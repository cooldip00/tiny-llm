import math

import pytest
import torch

from tinyllm.attention.mha import MHA
from tinyllm.config import get_config
from tinyllm.model import Block, RotaryEmbedding, Transformer

torch.manual_seed(0)
CFG = get_config("smoke")


@pytest.fixture(scope="module")
def model():
    return Transformer(CFG)


def test_block_shape_and_attention_from_registry():
    block = Block(CFG, 0)
    assert isinstance(block.self_attn, MHA)
    rope = RotaryEmbedding(CFG.head_dim, CFG.max_seq_len)(8)
    x = torch.randn(2, 8, CFG.d_model)
    assert block(x, rope).shape == x.shape


def test_forward_shapes(model):
    idx = torch.randint(0, CFG.vocab_size, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, CFG.vocab_size) and loss is None
    _, loss = model(idx, targets=idx)
    assert loss.shape == () and loss.dtype == torch.float32


def test_embeddings_tied(model):
    assert model.lm_head.weight is model.embed_tokens.weight


def test_untied_has_separate_head():
    m = Transformer(get_config("smoke", tie_embeddings=False))
    assert m.lm_head.weight is not m.embed_tokens.weight


def test_hf_llama_param_names(model):
    names = set(dict(model.named_parameters()))  # tied lm_head is listed once, as embed_tokens
    layer0 = {f"layers.0.{n}" for n in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")}
    assert {"embed_tokens.weight", "norm.weight"} | layer0 <= names


def test_rope_tables_not_in_state_dict(model):
    assert not any("cos" in k or "sin" in k for k in model.state_dict())


def test_init_std(model):
    resid = 0.02 / math.sqrt(2 * CFG.n_layers)
    w = model.layers[0]
    assert model.embed_tokens.weight.std().item() == pytest.approx(0.02, rel=0.02)
    assert w.self_attn.q_proj.weight.std().item() == pytest.approx(0.02, rel=0.02)
    assert w.mlp.gate_proj.weight.std().item() == pytest.approx(0.02, rel=0.02)
    assert w.self_attn.o_proj.weight.std().item() == pytest.approx(resid, rel=0.02)
    assert w.mlp.down_proj.weight.std().item() == pytest.approx(resid, rel=0.02)
    assert torch.all(w.input_layernorm.weight == 1)


def test_rejects_sequence_longer_than_max(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, CFG.max_seq_len + 1, dtype=torch.long))


# ---- Step 2.5: whole-model properties ----

@pytest.mark.parametrize("B,T", [(1, 1), (3, 7), (1, CFG.max_seq_len)])
def test_shapes_edge_lengths(model, B, T):
    logits, _ = model(torch.randint(0, CFG.vocab_size, (B, T)))
    assert logits.shape == (B, T, CFG.vocab_size)


def test_causal_through_all_layers(model):
    idx = torch.randint(0, CFG.vocab_size, (2, 32))
    idx2 = idx.clone()
    idx2[:, 20] = (idx2[:, 20] + 1) % CFG.vocab_size  # change one token at position 20
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)
    torch.testing.assert_close(a[:, :20], b[:, :20])  # earlier positions see nothing
    assert not torch.allclose(a[:, 20:], b[:, 20:])   # position 20 onward does change


@pytest.mark.parametrize("name,overrides", [
    ("smoke", {}), ("main", {}), ("smoke", {"tie_embeddings": False}), ("smoke", {"n_layers": 3}),
])
def test_param_count_matches_formula(name, overrides):
    cfg = get_config(name, **overrides)
    assert Transformer(cfg).num_params() == cfg.estimate_params()


def test_exact_preset_sizes():
    assert Transformer(get_config("smoke")).num_params() == 13_767_552
    assert Transformer(get_config("main")).num_params() == 29_893_120


def test_initial_loss_near_uniform():
    # A well-initialised model is almost uniform over the vocab: loss ≈ ln(V) = 9.01, plus a small
    # excess from the spread of the initial logits. A logit is RMSNorm(h) (|.|² ≈ d) · an embedding
    # row (std init_std), so its std is σ = init_std·√d (0.39 here), and E[loss] ≈ ln V + σ²/2 = 9.09.
    # Measured over 20 seeds: mean 9.088, std 0.026. Seeded, so it no longer depends on test order.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (4, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, loss = m(idx[:, :-1], targets=idx[:, 1:])
    sigma = CFG.init_std * math.sqrt(CFG.d_model)
    assert loss.item() == pytest.approx(math.log(CFG.vocab_size) + sigma**2 / 2, abs=0.08)  # ~3 std


def test_every_parameter_gets_a_gradient():
    m = Transformer(CFG)
    idx = torch.randint(0, CFG.vocab_size, (2, 16))
    _, loss = m(idx[:, :-1], targets=idx[:, 1:])
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_autocast_cuda():
    m = Transformer(CFG).cuda()
    idx = torch.randint(0, CFG.vocab_size, (2, 128), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = m(idx[:, :-1], targets=idx[:, 1:])
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
    assert torch.isfinite(loss) and loss.item() == pytest.approx(math.log(CFG.vocab_size), abs=0.1)
