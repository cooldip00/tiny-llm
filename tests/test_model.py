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
