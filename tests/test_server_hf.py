"""6.5b: the server's --hf-model loader (a 4.8b export, from a folder or the Hub), on CPU.

The Hub download is replaced by a fake snapshot_download, so no test touches the network.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import huggingface_hub
import pytest
import torch

from tinyllm.hf_llama import config_from_hf
from tinyllm.model import norm_backend
from tinyllm.server import HF_MODEL_FILES, load_variant, main, resolve_hf_model
from tinyllm.tokenizer import EOS, load_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import make_tiny_hf_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return make_tiny_hf_model.make(tmp_path_factory.mktemp("tiny") / "tiny-story", seed=0)


def args(ref, **kw):
    return Namespace(hf_model=ref, revision=kw.get("revision"), offline=kw.get("offline", False),
                     device="cpu", ckpt=None, model=None)


def test_folder_loads_with_its_weights_and_our_tokenizer(tiny_dir):
    model, tok, name, version = load_variant(args(str(tiny_dir)))
    assert set(HF_MODEL_FILES) <= {p.name for p in tiny_dir.iterdir()}
    assert name == "tiny-story" and version.startswith("tiny-story@") and "+" in version
    assert model.cfg == config_from_hf(json.loads((tiny_dir / "config.json").read_text()))
    assert next(model.parameters()).dtype == torch.float32
    assert norm_backend(model) == "torch"                      # what /stats reports on a CPU Space
    # Same weights as the model that was saved (same seed), so the same logits.
    from tinyllm.config import ModelConfig
    from tinyllm.model import Transformer
    with torch.random.fork_rng():
        torch.manual_seed(0)
        ref = Transformer(ModelConfig(**make_tiny_hf_model.TINY)).eval()
    x = torch.tensor([[5, 17, 99, 3, 42, 8]])
    with torch.no_grad():
        assert torch.equal(model(x)[0], ref(x)[0])
    # Our TinyStories BPE: no BOS, EOS <|endoftext|>, decode(encode(text)) == text.
    text = "Once upon a time, a little dog found a red ball."
    assert tok.encode(text) == load_tokenizer().encode(text).ids
    assert tok.decode(tok.encode(text)) == text
    assert tok.eos_id == load_tokenizer().token_to_id(EOS) == 0


def test_repo_id_goes_through_the_hf_cache_with_the_revision(tiny_dir, monkeypatch):
    calls = []

    def fake(repo, revision=None, allow_patterns=None, local_files_only=False):
        calls.append((repo, revision, allow_patterns, local_files_only))
        return str(tiny_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    _, _, name, version = load_variant(args("someone/tiny-story-30m", revision="abc123", offline=True))
    assert calls == [("someone/tiny-story-30m", "abc123", HF_MODEL_FILES, True)]
    assert name == "tiny-story-30m" and version.startswith("tiny-story-30m@")


def test_revision_with_a_local_folder_is_an_error(tiny_dir):
    with pytest.raises(ValueError, match="--revision"):
        resolve_hf_model(str(tiny_dir), revision="main")


def test_a_folder_with_another_tokenizer_is_refused(tiny_dir, tmp_path):
    other = tmp_path / "llama-like"
    other.mkdir()
    for f in HF_MODEL_FILES:
        (other / f).write_bytes((tiny_dir / f).read_bytes())
    tc = json.loads((other / "tokenizer_config.json").read_text())
    (other / "tokenizer_config.json").write_text(json.dumps({**tc, "bos_token": "<s>", "eos_token": "</s>"}))
    with pytest.raises(ValueError, match="--model tinyllama"):
        load_variant(args(str(other)))


def test_eos_mismatch_between_config_and_tokenizer_is_refused(tiny_dir, tmp_path):
    bad = tmp_path / "bad-eos"
    bad.mkdir()
    for f in HF_MODEL_FILES:
        (bad / f).write_bytes((tiny_dir / f).read_bytes())
    cfg = json.loads((bad / "config.json").read_text())
    (bad / "config.json").write_text(json.dumps({**cfg, "eos_token_id": 2}))
    with pytest.raises(ValueError, match="EOS id"):
        load_variant(args(str(bad)))


def test_hf_model_and_ckpt_are_mutually_exclusive(tiny_dir):
    with pytest.raises(SystemExit):
        main(["--hf-model", str(tiny_dir), "--ckpt", "x.pt"])
