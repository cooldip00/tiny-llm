import pytest
import torch
import torch.nn.functional as F

from tinyllm.config import get_config
from tinyllm.model import RMSNorm, RotaryEmbedding, SwiGLU, apply_rope

torch.manual_seed(0)


# ---------- RMSNorm ----------

def test_rmsnorm_matches_pytorch_reference():
    norm = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(2, 10, 64)
    ref = F.rms_norm(x, (64,), weight=norm.weight, eps=1e-5)
    torch.testing.assert_close(norm(x), ref)


def test_rmsnorm_output_has_unit_rms():
    y = RMSNorm(64)(torch.randn(4, 64) * 50)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(4), atol=1e-4, rtol=0)


def test_rmsnorm_keeps_bf16_dtype():
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    assert RMSNorm(64).to(torch.bfloat16)(x).dtype == torch.bfloat16


# ---------- RoPE ----------

D, T = 64, 32


@pytest.fixture(scope="module")
def rope():
    return RotaryEmbedding(D, max_seq_len=128)


def test_rope_position_zero_is_identity(rope):
    x = torch.randn(1, 2, 1, D)
    cos, sin = rope(1)
    torch.testing.assert_close(apply_rope(x, cos, sin), x)


def test_rope_preserves_norm(rope):
    x = torch.randn(2, 4, T, D)
    cos, sin = rope(T)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rope_score_depends_only_on_relative_position(rope):
    q, k = torch.randn(D), torch.randn(D)

    def score(m: int, n: int) -> torch.Tensor:
        cq, sq = rope(1, offset=m)
        ck, sk = rope(1, offset=n)
        qr = apply_rope(q.view(1, 1, 1, D), cq, sq)
        kr = apply_rope(k.view(1, 1, 1, D), ck, sk)
        return (qr * kr).sum()

    torch.testing.assert_close(score(5, 2), score(105, 102), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(score(9, 9), score(0, 0), atol=1e-4, rtol=1e-4)


def test_rope_matches_complex_number_reference(rope):
    # rotate_half layout = complex multiply of z_i = x_i + j·x_{i+D/2} by e^{j·pos·freq_i}
    x = torch.randn(1, 1, T, D)
    cos, sin = rope(T)
    freqs = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
    angles = torch.outer(torch.arange(T).float(), freqs)
    z = torch.complex(x[..., : D // 2], x[..., D // 2:]) * torch.polar(torch.ones_like(angles), angles)
    ref = torch.cat([z.real, z.imag], dim=-1)
    torch.testing.assert_close(apply_rope(x, cos, sin), ref, atol=1e-5, rtol=1e-5)


def test_rope_offset_equals_slice_of_full_table(rope):
    cos_full, sin_full = rope(T)
    cos_off, sin_off = rope(4, offset=10)
    torch.testing.assert_close(cos_off, cos_full[10:14])
    torch.testing.assert_close(sin_off, sin_full[10:14])


def test_rope_rejects_positions_past_max(rope):
    with pytest.raises(ValueError):
        rope(10, offset=125)


def test_rope_buffers_not_in_state_dict(rope):
    assert len(rope.state_dict()) == 0


# ---------- SwiGLU ----------

def test_swiglu_formula_shape_and_params():
    cfg = get_config("smoke")
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    y = mlp(x)
    assert y.shape == x.shape
    ref = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(y, ref)
    assert sum(p.numel() for p in mlp.parameters()) == 3 * cfg.d_model * cfg.ffn_hidden
    assert all(m.bias is None for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))


def test_hf_parameter_names():
    names = set(dict(SwiGLU(get_config("smoke")).named_parameters()))
    assert names == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
