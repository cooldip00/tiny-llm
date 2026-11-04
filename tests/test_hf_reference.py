"""Step 4.9c: our model vs HF Transformers LlamaForCausalLM (test-only reference, D-015).

CPU part: random tiny Llamas in fp32 against HF's *eager* attention (an independent implementation
of the same math), in both directions of the weight conversion. GPU part: the real TinyLlama, skipped
unless it is in the HF cache and CUDA is available.
"""

import dataclasses

import pytest
import torch

pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from tinyllm import hf_llama as hl  # noqa: E402
from tinyllm.config import ModelConfig  # noqa: E402
from tinyllm.generate import generate  # noqa: E402
from tinyllm.logit_compare import hf_stepwise, logit_stats, ours_stepwise  # noqa: E402
from tinyllm.model import Transformer  # noqa: E402

# Non-default theta and eps, so a silently dropped config field shows up as a logit mismatch.
SMALL = ModelConfig(vocab_size=96, max_seq_len=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                    ffn_hidden=96, rope_theta=500.0, norm_eps=1e-6)
# fp32 vs HF eager: logits reach ~12 and differ by ~2e-5 (op-order rounding, ~2e-6 relative).
# A real mapping bug is far larger: the wrong rope_theta alone gives a max diff of ~14.
FP32 = dict(atol=1e-4, rtol=1e-4)
TIE = pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])


def _scramble(model: torch.nn.Module, seed: int) -> None:
    """Large random weights (norm gains too), so logits are far from uniform and argmax is sharp."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.ndim == 2 else 1.0) + (1.0 if p.ndim == 1 else 0.0))


def hf_tiny(tie: bool) -> LlamaForCausalLM:
    c = SMALL
    cfg = LlamaConfig(vocab_size=c.vocab_size, hidden_size=c.d_model, intermediate_size=c.ffn_hidden,
                      num_hidden_layers=c.n_layers, num_attention_heads=c.n_heads, num_key_value_heads=c.n_kv_heads,
                      max_position_embeddings=c.max_seq_len, rope_theta=c.rope_theta, rms_norm_eps=c.norm_eps,
                      tie_word_embeddings=tie, bos_token_id=None, eos_token_id=None, pad_token_id=None,
                      attn_implementation="eager")
    m = LlamaForCausalLM(cfg).eval()
    _scramble(m, 0)
    return m


def ids(B=2, T=20, seed=1):
    return torch.randint(0, SMALL.vocab_size, (B, T), generator=torch.Generator().manual_seed(seed))


@TIE
def test_hf_checkpoint_loads_with_matching_logits(tmp_path, tie):
    hf = hf_tiny(tie)
    hf.save_pretrained(tmp_path)                       # transformers 5 layout (rope_parameters, dtype)
    ours = hl.load_from_dir(tmp_path, device="cpu", dtype=torch.float32)
    assert ours.cfg == dataclasses.replace(SMALL, tie_embeddings=tie, init_std=hf.config.initializer_range)
    x = ids()
    torch.testing.assert_close(ours(x)[0], hf(x).logits, **FP32)


@TIE
def test_our_export_loads_in_hf_with_matching_logits(tmp_path, tie):
    ours = Transformer(dataclasses.replace(SMALL, tie_embeddings=tie)).eval()
    _scramble(ours, 2)
    hl.save_as_hf(ours, tmp_path)
    hf = LlamaForCausalLM.from_pretrained(tmp_path, dtype=torch.float32, attn_implementation="eager").eval()
    assert hf.config.num_key_value_heads == SMALL.n_kv_heads and hf.config.tie_word_embeddings == tie
    x = ids(seed=3)
    with torch.no_grad():
        torch.testing.assert_close(hf(x).logits, ours(x)[0], **FP32)


def test_kv_cached_decode_matches_hf(tmp_path):
    hf = hf_tiny(False)
    hf.save_pretrained(tmp_path)
    ours = hl.load_from_dir(tmp_path, device="cpu", dtype=torch.float32)
    x = ids(T=24, seed=4)
    with torch.no_grad():
        full = hf(x).logits.float()
    # Ours prefill 8 + 16 single-token steps, against HF's full forward and HF's own cached decode.
    torch.testing.assert_close(ours_stepwise(ours, x, 8), full, **FP32)
    torch.testing.assert_close(hf_stepwise(hf, x, 8), full, **FP32)


def test_greedy_generation_matches_hf(tmp_path):
    hf = hf_tiny(False)
    hf.save_pretrained(tmp_path)
    ours = hl.load_from_dir(tmp_path, device="cpu", dtype=torch.float32)
    x = ids(T=8, seed=5)
    theirs = hf.generate(x, max_new_tokens=16, do_sample=False)
    assert torch.equal(generate(ours, x, 16, temperature=0), theirs)


# --- the real TinyLlama on GPU ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tinyllama_dir():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        return hl.download(local_only=True)
    except Exception:
        pytest.skip("TinyLlama not in the HF cache (run scripts/tinyllama_smoke.py once)")


def test_tinyllama_tokenizer_vs_hf(tinyllama_dir):
    """We encode with the shipped tokenizer.json as-is. transformers 5's AutoTokenizer applies the
    `"legacy": false` flag from tokenizer_config.json on top, which differs in exactly two places:
      - text right after a special token gets no "▁" (tokenizer.json's normalizer prepends one to
        every segment, so "</s>\\n" is ["</s>", "▁", "<0x0A>"] for us);
      - one leading space of the input is absorbed into the prefix " one" -> ["▁one"] (ours: ["▁", "▁one"]).
    Its apply_chat_template also adds no BOS (the template has none); we add BOS like plain encode.
    Ordinary text is identical. 4.9c's logit checks feed identical ids to both models.
    """
    from transformers import AutoTokenizer
    ref = AutoTokenizer.from_pretrained(tinyllama_dir)
    tok = hl.HFTokenizer(tinyllama_dir)
    for s in ["Hello world", "naïve café, 東京 🙂", "1234567890 + 0.5", "a  b\n  c\t d", "x\n\ny"]:
        assert tok.encode(s) == ref(s).input_ids, s
    assert tok.encode(" one space") == [1, 29871, 697, 2913] and ref(" one space").input_ids == [1, 697, 2913]
    space = tok.tok.token_to_id("▁")
    for user, system in [("What is 2 + 2?", None), ("Hi!", "You are a pirate.")]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        want = ref.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        want = list(want["input_ids"] if hasattr(want, "keys") else want)
        assert ref.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) == tok.chat_prompt(user, system)
        # Same string; tokens differ only by our BOS and one "▁" after each "</s>".
        expect = [tok.bos_id]
        for i in want:
            expect += [i, space] if i == tok.eos_id else [i]
        assert tok.encode(tok.chat_prompt(user, system)) == expect


def test_tinyllama_bf16_logits_as_close_to_fp32_as_hf_bf16(tinyllama_dir):
    """Done-when for 4.9c: ours in bf16 is no further from the fp32 reference than HF's own bf16."""
    tok = hl.HFTokenizer(tinyllama_dir)
    prompt = torch.tensor([tok.encode(tok.chat_prompt("Name three primary colors."))], device="cuda")
    n = prompt.shape[1]

    hf = LlamaForCausalLM.from_pretrained(tinyllama_dir, dtype=torch.float32, attn_implementation="eager")
    hf = hf.to("cuda").eval()
    seq = hf.generate(prompt, max_new_tokens=24, do_sample=False)          # the teacher-forced sequence
    with torch.no_grad():
        truth = hf(seq).logits.float().cpu()
    del hf
    torch.cuda.empty_cache()

    hf = LlamaForCausalLM.from_pretrained(tinyllama_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    hf = hf.to("cuda").eval()
    theirs = hf_stepwise(hf, seq, n)
    del hf
    torch.cuda.empty_cache()
    ours = ours_stepwise(hl.load_from_dir(tinyllama_dir), seq, n)

    for name, sl in [("prefill", slice(0, n)), ("decode", slice(n, None))]:
        ref, o_l, h_l = truth[0, sl], ours[0, sl], theirs[0, sl]
        o, h = logit_stats(ref, o_l), logit_stats(ref, h_l)
        assert o["kl_mean"] <= 1.5 * h["kl_mean"] + 1e-4, (name, o, h)
        assert o["max_abs"] <= 1.5 * h["max_abs"] + 0.05, (name, o, h)
        # Top-1 vs fp32 is not ~100% even for HF bf16: near-ties flip under bf16 rounding (4.9c run:
        # 3 / 264 decode positions, fp32 top-2 margins 0.002-0.019 vs a median of 1.7, identical for HF).
        # So: no more flips than HF + 1, and every flip of ours must be at a near-tie.
        assert o["top1"] * o["positions"] >= h["top1"] * h["positions"] - 1, (name, o, h)
        top2 = ref.topk(2, -1).values
        flips = (top2[:, 0] - top2[:, 1])[o_l.argmax(-1) != ref.argmax(-1)]
        assert (flips < 0.1).all(), (name, flips)
