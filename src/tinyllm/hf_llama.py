"""Load a pretrained Hugging Face Llama checkpoint into our Transformer (step 4.9b, D-015 / D-026).

Only our code runs the model. Hugging Face pieces used here are file formats and downloads:
`huggingface_hub` fetches the files, our `safetensors_io` reads the weights, and `tokenizers`
reads `tokenizer.json`. HF Transformers is used only as the logits reference in tests (4.9c).

Our parameter names already follow HF Llama (embed_tokens, layers.N.self_attn.q_proj, ...,
input_layernorm, mlp.gate_proj, norm, lm_head), so the mapping just drops HF's "model." prefix.
RoPE uses the same rotate_half layout as HF, so q/k weights need no permutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tinyllm.config import ModelConfig
from tinyllm.model import RotaryEmbedding, Transformer
from tinyllm.safetensors_io import load_file

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_REV = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"   # pinned (D-026)
FILES = ["config.json", "generation_config.json", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def config_from_hf(hf: dict) -> ModelConfig:
    """HF LlamaConfig dict -> ModelConfig. Raises on any feature our model doesn't implement."""
    unsupported = []
    if hf.get("model_type") != "llama":
        unsupported.append(f"model_type={hf.get('model_type')!r}")
    if hf.get("hidden_act", "silu") != "silu":
        unsupported.append(f"hidden_act={hf['hidden_act']!r}")
    if hf.get("rope_scaling"):
        unsupported.append(f"rope_scaling={hf['rope_scaling']!r}")
    # transformers >= 5 writes {"rope_parameters": {"rope_theta": ..., "rope_type": ...}} instead of
    # a top-level rope_theta; scaled variants (linear, llama3, yarn, ...) are a rope_type there.
    rp = hf.get("rope_parameters") or {}
    if rp.get("rope_type", "default") != "default":
        unsupported.append(f"rope_parameters={rp!r}")
    rope_theta = rp.get("rope_theta", hf.get("rope_theta", 10000.0))
    if hf.get("attention_bias") or hf.get("mlp_bias"):
        unsupported.append("bias in attention / MLP")
    if hf.get("pretraining_tp", 1) != 1:
        unsupported.append(f"pretraining_tp={hf['pretraining_tp']}")
    hd = hf.get("head_dim")
    if hd is not None and hd != hf["hidden_size"] // hf["num_attention_heads"]:
        unsupported.append(f"head_dim={hd} != hidden_size / num_attention_heads")
    if unsupported:
        raise ValueError("unsupported Llama config: " + ", ".join(unsupported))
    return ModelConfig(
        vocab_size=hf["vocab_size"],
        max_seq_len=hf["max_position_embeddings"],
        d_model=hf["hidden_size"],
        n_layers=hf["num_hidden_layers"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
        ffn_hidden=hf["intermediate_size"],
        norm_eps=hf["rms_norm_eps"],
        rope_theta=rope_theta,
        init_std=hf.get("initializer_range", 0.02),
        tie_embeddings=hf.get("tie_word_embeddings", False),
    )


def hf_to_ours(name: str) -> str:
    return name.removeprefix("model.")


def ours_to_hf(name: str) -> str:
    return name if name == "lm_head.weight" else "model." + name


def download(repo: str = TINYLLAMA, revision: str = TINYLLAMA_REV, local_only: bool = False) -> Path:
    """Snapshot directory in the HF cache (~/.cache/huggingface). Weights never go into the repo."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision, allow_patterns=FILES, local_files_only=local_only))


def load_from_dir(path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Transformer:
    """Build our Transformer from <path>/config.json and <path>/model.safetensors, strict.

    The model is created on the meta device, so no fp32 copy is ever allocated (TinyLlama: 4.4 GB);
    each weight is copied once from the memory-mapped file to `device` in `dtype` (D-023: bf16).
    """
    path = Path(path)
    cfg = config_from_hf(json.loads((path / "config.json").read_text()))
    state = {hf_to_ours(k): v for k, v in load_file(path / "model.safetensors", device, dtype).items()}
    # Rotary inv_freq is a buffer in some HF exports; ours are recomputed and never stored.
    state = {k: v for k, v in state.items() if not k.endswith("rotary_emb.inv_freq")}
    if cfg.tie_embeddings:
        state.pop("lm_head.weight", None)        # tied: the embedding is the head
    with torch.device("meta"):
        model = Transformer(cfg)
    expected = {k: v for k, v in model.state_dict().items() if not (cfg.tie_embeddings and k == "lm_head.weight")}
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    bad_shape = [k for k in expected.keys() & state.keys() if tuple(expected[k].shape) != tuple(state[k].shape)]
    if missing or unexpected or bad_shape:
        raise ValueError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}, "
                         f"shape {bad_shape[:5]}")
    model.load_state_dict(state, strict=False, assign=True)   # strictness checked above (tie aware)
    if cfg.tie_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    # Non-persistent buffers were created on meta: rebuild the RoPE tables on the real device.
    model.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta).to(device)
    return model.eval()


def load_hf_llama(repo: str = TINYLLAMA, revision: str = TINYLLAMA_REV, device: str = "cuda",
                  dtype: torch.dtype = torch.bfloat16, local_only: bool = False) -> Transformer:
    return load_from_dir(download(repo, revision, local_only), device, dtype)


def save_as_hf(model: Transformer, path: str | Path) -> None:
    """Write our model as an HF-style Llama directory (config.json + model.safetensors). Used by tests and 4.8b."""
    from tinyllm.safetensors_io import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    c = model.cfg
    hf = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_act": "silu",
          "vocab_size": c.vocab_size, "max_position_embeddings": c.max_seq_len, "hidden_size": c.d_model,
          "num_hidden_layers": c.n_layers, "num_attention_heads": c.n_heads,
          "num_key_value_heads": c.n_kv_heads, "intermediate_size": c.ffn_hidden, "rms_norm_eps": c.norm_eps,
          "rope_theta": c.rope_theta, "initializer_range": c.init_std, "tie_word_embeddings": c.tie_embeddings,
          "attention_bias": False, "rope_scaling": None}
    (path / "config.json").write_text(json.dumps(hf, indent=2))
    sd = model.state_dict()
    if c.tie_embeddings:
        sd.pop("lm_head.weight")
    save_file({ours_to_hf(k): v for k, v in sd.items()}, path / "model.safetensors", {"format": "pt"})


# --- tokenizer -------------------------------------------------------------------------------------

class HFTokenizer:
    """tokenizer.json via the `tokenizers` library. encode() adds BOS (<s>) like the Llama tokenizer."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        path = Path(path)
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        cfg = json.loads((path / "tokenizer_config.json").read_text()) if (path / "tokenizer_config.json").exists() else {}
        self.chat_template = cfg.get("chat_template")
        self.bos_id = self.tok.token_to_id("<s>")
        self.eos_id = self.tok.token_to_id("</s>")

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return ([self.bos_id] if bos else []) + ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    @staticmethod
    def chat_prompt(user: str, system: str | None = None) -> str:
        """TinyLlama-Chat v1.0 (Zephyr) template, with the assistant turn opened for generation."""
        sys = f"<|system|>\n{system}</s>\n" if system else ""
        return f"{sys}<|user|>\n{user}</s>\n<|assistant|>\n"
