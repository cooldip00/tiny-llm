"""Step 4.9b: safetensors I/O, HF Llama config + weight loading, TinyLlama (when cached)."""

import json
import struct

import pytest
import torch

from tinyllm import hf_llama as hl
from tinyllm.config import ModelConfig
from tinyllm.generate import generate
from tinyllm.model import Transformer
from tinyllm.safetensors_io import load_file, read_header, save_file

# TinyLlama-1.1B-Chat-v1.0 config.json at the pinned revision (D-026).
TINYLLAMA_CFG = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "bos_token_id": 1, "eos_token_id": 2,
    "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 5632,
    "max_position_embeddings": 2048, "model_type": "llama", "num_attention_heads": 32,
    "num_hidden_layers": 22, "num_key_value_heads": 4, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
    "rope_scaling": None, "rope_theta": 10000.0, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    "transformers_version": "4.35.0", "use_cache": True, "vocab_size": 32000,
}
SMALL = ModelConfig(vocab_size=96, max_seq_len=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=96)


# --- safetensors ---------------------------------------------------------------------------------

def test_safetensors_roundtrip_all_dtypes(tmp_path):
    g = torch.Generator().manual_seed(0)
    ts = {"bf16": torch.randn(3, 5, generator=g).bfloat16(), "f16": torch.randn(7, generator=g).half(),
          "f32": torch.randn(2, 2, 2, generator=g), "i64": torch.arange(-3, 9).view(3, 4),
          "u8": torch.arange(10, dtype=torch.uint8), "bool": torch.tensor([True, False, True]),
          "scalar": torch.tensor(2.5), "empty": torch.zeros(0, 4)}
    save_file(ts, tmp_path / "a.safetensors", {"format": "pt"})
    back = load_file(tmp_path / "a.safetensors")
    assert back.keys() == ts.keys()
    for k in ts:
        assert back[k].dtype == ts[k].dtype and torch.equal(back[k], ts[k]), k
    header, start = read_header(tmp_path / "a.safetensors")
    assert "__metadata__" not in header and start % 8 == 0          # header padded to 8 bytes


def test_reads_a_hand_written_file(tmp_path):
    # Built byte by byte from the format spec, independent of save_file.
    data = struct.pack("<2e", 1.5, -2.0) + struct.pack("<3i", 7, 8, 9)   # F16 [2], then I32 [3]
    header = json.dumps({"__metadata__": {"x": "y"},
                         "h": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
                         "i": {"dtype": "I32", "shape": [3], "data_offsets": [4, 16]}}).encode()
    header += b" " * (-len(header) % 8)
    p = tmp_path / "hand.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + data)
    t = load_file(p)
    assert t["h"].tolist() == [1.5, -2.0] and t["h"].dtype == torch.float16
    assert t["i"].tolist() == [7, 8, 9] and t["i"].dtype == torch.int32
    assert load_file(p, dtype=torch.bfloat16)["i"].dtype == torch.int32  # integers are never cast


def test_bad_byte_count_raises(tmp_path):
    header = json.dumps({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}).encode()
    header += b" " * (-len(header) % 8)
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    with pytest.raises(ValueError, match="bytes for shape"):
        load_file(p)


# --- config ----------------------------------------------------------------------------------------

def test_tinyllama_config_maps():
    c = hl.config_from_hf(TINYLLAMA_CFG)
    assert (c.vocab_size, c.max_seq_len, c.d_model, c.n_layers, c.n_heads, c.n_kv_heads, c.ffn_hidden) == \
        (32000, 2048, 2048, 22, 32, 4, 5632)
    assert c.head_dim == 64 and c.layer_types == ("gqa",) * 22 and not c.tie_embeddings
    assert c.estimate_params() == 1_100_048_384                      # TinyLlama's published 1.1B


def test_rope_theta_from_transformers_v5_layout():
    v5 = {k: v for k, v in TINYLLAMA_CFG.items() if k not in ("rope_theta", "rope_scaling")}
    v5["rope_parameters"] = {"rope_type": "default", "rope_theta": 500000.0}
    assert hl.config_from_hf(v5).rope_theta == 500000.0
    assert hl.config_from_hf(TINYLLAMA_CFG).rope_theta == 10000.0       # pre-v5 top-level key


@pytest.mark.parametrize("field,value", [
    ("rope_scaling", {"type": "linear", "factor": 2.0}), ("hidden_act", "gelu"), ("attention_bias", True),
    ("model_type", "mistral"), ("pretraining_tp", 2), ("head_dim", 128),
    ("rope_parameters", {"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0}),
])
def test_unsupported_config_raises(field, value):
    with pytest.raises(ValueError, match="unsupported"):
        hl.config_from_hf({**TINYLLAMA_CFG, field: value})


# --- weights: our model -> HF layout -> our model ---------------------------------------------------

@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
def test_hf_dir_roundtrip_gives_identical_logits(tmp_path, tie):
    import dataclasses
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = Transformer(dataclasses.replace(SMALL, tie_embeddings=tie)).eval()
    hl.save_as_hf(m, tmp_path)
    names = set(read_header(tmp_path / "model.safetensors")[0])
    assert "model.layers.0.self_attn.k_proj.weight" in names and "model.norm.weight" in names
    assert ("lm_head.weight" in names) != tie                        # tied checkpoints omit the head
    back = hl.load_from_dir(tmp_path, device="cpu", dtype=torch.float32)
    assert back.cfg == m.cfg
    assert (back.lm_head.weight is back.embed_tokens.weight) == tie
    idx = torch.randint(0, SMALL.vocab_size, (2, 9), generator=torch.Generator().manual_seed(1))
    assert torch.equal(back(idx)[0], m(idx)[0])
    assert torch.equal(generate(back, idx, 8, temperature=0), generate(m, idx, 8, temperature=0))


def test_missing_or_extra_weights_raise(tmp_path):
    m = Transformer(SMALL)
    hl.save_as_hf(m, tmp_path)
    sd = load_file(tmp_path / "model.safetensors")
    save_file({k: v for k, v in sd.items() if "layers.1.mlp.up_proj" not in k} | {"model.extra": torch.ones(1)},
              tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing.*up_proj.*unexpected.*extra"):
        hl.load_from_dir(tmp_path, device="cpu")


def test_loads_in_bf16_on_requested_device(tmp_path):
    hl.save_as_hf(Transformer(SMALL), tmp_path)
    m = hl.load_from_dir(tmp_path, device="cpu", dtype=torch.bfloat16)
    assert {p.dtype for p in m.parameters()} == {torch.bfloat16}
    assert m.rope.cos.device.type == "cpu" and m.rope.cos.dtype == torch.float32   # rebuilt, not meta


# --- the real TinyLlama (skipped unless already downloaded) -----------------------------------------

@pytest.fixture(scope="module")
def tinyllama_dir():
    try:
        return hl.download(local_only=True)
    except Exception:
        pytest.skip("TinyLlama not in the HF cache (run scripts/tinyllama_smoke.py once)")


def test_tinyllama_header_matches_our_model(tinyllama_dir):
    header, _ = read_header(tinyllama_dir / "model.safetensors")
    cfg = hl.config_from_hf(json.loads((tinyllama_dir / "config.json").read_text()))
    with torch.device("meta"):
        ours = {k: tuple(v.shape) for k, v in Transformer(cfg).state_dict().items()}
    theirs = {hl.hf_to_ours(k): tuple(v["shape"]) for k, v in header.items()}
    assert theirs == ours and {v["dtype"] for v in header.values()} == {"BF16"}


def test_tinyllama_tokenizer(tinyllama_dir):
    tok = hl.HFTokenizer(tinyllama_dir)
    assert (tok.bos_id, tok.eos_id) == (1, 2)
    ids = tok.encode("Hello world")
    assert ids[0] == 1 and tok.decode(ids) == "Hello world"
    # Our chat prompt is the template TinyLlama's tokenizer_config ships (Zephyr format).
    assert "<|user|>" in tok.chat_template and "<|assistant|>" in tok.chat_template
