import json

import pytest

from tinyllm.config import ModelConfig, get_config


def test_presets_match_architecture_doc():
    smoke, main = get_config("smoke"), get_config("main")
    assert (smoke.d_model, smoke.n_layers, smoke.n_heads, smoke.head_dim, smoke.ffn_hidden) == (384, 6, 6, 64, 1024)
    assert (main.d_model, main.n_layers, main.n_heads, main.head_dim, main.ffn_hidden) == (512, 8, 8, 64, 1408)
    assert main.vocab_size == 8192 and main.max_seq_len == 512


def test_param_estimates():
    # Hand-computed from V·d + L·(4·d² + 3·d·ffn + 2·d) + d
    assert get_config("smoke").estimate_params() == 13_767_552  # ~13.8M
    assert get_config("main").estimate_params() == 29_893_120   # ~29.9M


def test_flops_per_token():
    # 6·(params − RMSNorm weights) + 12·L·d·T, hand-computed at T = 512
    main, smoke = get_config("main"), get_config("smoke")
    assert main.flops_per_token(512) == 6 * (29_893_120 - 8_704) + 12 * 8 * 512 * 512 == 204_472_320
    assert smoke.flops_per_token(512) == 6 * (13_767_552 - 4_992) + 12 * 6 * 384 * 512 == 96_731_136
    assert main.flops_per_token(1024) - main.flops_per_token(512) == 12 * 8 * 512 * 512  # attention term grows with T
    assert get_config("main", n_kv_heads=2).flops_per_token(512) < main.flops_per_token(512)


def test_defaults_fill_in():
    cfg = get_config("main")
    assert cfg.n_kv_heads == cfg.n_heads
    assert cfg.layer_types == ("mha",) * 8


def test_gqa_override_reduces_params():
    gqa = get_config("main", n_kv_heads=2)
    assert gqa.n_kv_heads == 2
    assert gqa.estimate_params() < get_config("main").estimate_params()


def test_n_layers_override_rederives_layer_types():
    assert get_config("main", n_layers=4).layer_types == ("mha",) * 4


def test_hybrid_layer_types():
    cfg = get_config("main", layer_types=["gdn", "gdn", "gdn", "mha"] * 2)
    assert cfg.layer_types[3] == "mha" and len(cfg.layer_types) == 8


@pytest.mark.parametrize("bad", [
    dict(d_model=500, n_heads=8),          # not divisible
    dict(n_heads=8, n_kv_heads=3),         # GQA groups uneven
    dict(n_layers=8, layer_types=["mha"]),  # wrong length
    dict(d_model=24, n_heads=8),           # head_dim 3 is odd, breaks RoPE
])
def test_invalid_configs_raise(bad):
    with pytest.raises(ValueError):
        ModelConfig(**bad)


def test_json_roundtrip():
    cfg = get_config("main", n_kv_heads=4)
    assert ModelConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_frozen():
    with pytest.raises(AttributeError):
        get_config("main").d_model = 1024
