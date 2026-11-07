"""6.5b: served text keeps the space between the prompt and the completion (PARKING_LOT 2026-11-06).

A SentencePiece decoder (TinyLlama) drops the leading space at the start of a decode, so the
server decodes the completion behind the prompt's last token.
"""

import pytest

from tinyllm.server import _Detokenizer, decode_completion

PIECES = ["</s>", "▁Once", "▁upon", "▁a", "▁time", ",", "▁Max", "▁was", "▁happy", "."]


class MetaspaceTokenizer:
    """SentencePiece-style: '▁' marks a space; decode drops the first leading space."""
    eos_id = 0

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        text = "".join(PIECES[i] for i in ids if i != 0).replace("▁", " ")
        return text[1:] if text.startswith(" ") else text


def ids(*pieces):
    return [PIECES.index(p) for p in pieces]


PROMPT, COMP = ids("▁Once", "▁upon", "▁a", "▁time", ","), ids("▁Max", "▁was", "▁happy", ".")


def test_plain_decode_drops_the_space_decode_completion_keeps_it():
    tok = MetaspaceTokenizer()
    assert tok.decode(COMP) == "Max was happy."                 # the bug: "time,Max was happy."
    assert decode_completion(tok, COMP, PROMPT[-1:]) == " Max was happy."
    assert tok.decode(PROMPT) + decode_completion(tok, COMP, PROMPT[-1:]) == tok.decode(PROMPT + COMP)


def test_streamed_pieces_join_to_the_same_text():
    tok = MetaspaceTokenizer()
    d = _Detokenizer(tok, PROMPT[-1:])
    streamed = "".join(d.add(t) for t in COMP) + d.flush()
    assert streamed == " Max was happy."


def test_no_context_and_start_of_text_behave_as_before():
    tok = MetaspaceTokenizer()
    assert decode_completion(tok, COMP, []) == "Max was happy."
    assert decode_completion(tok, COMP, [0]) == "Max was happy."      # only EOS / BOS before it: text start


def test_falls_back_when_the_context_is_not_a_prefix():
    class Merging:                         # the context's text changes once more tokens follow it
        def decode(self, ids):
            return "X" if ids == [1] else "Y" + "".join(map(str, ids))
    assert decode_completion(Merging(), [2, 3], [1]) == "Y23"


def test_tinyllama_tokenizer_if_cached():
    hub = pytest.importorskip("huggingface_hub")
    try:
        path = hub.snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_files_only=True,
                                     allow_patterns=["tokenizer.json", "tokenizer_config.json"])
    except Exception:
        pytest.skip("TinyLlama tokenizer not in the HF cache")
    from tinyllm.hf_llama import HFTokenizer
    tok = HFTokenizer(path)
    prompt, full = tok.encode("Once upon a time, a dog named Max.", bos=True), \
        tok.encode("Once upon a time, a dog named Max. Max was happy.", bos=True)
    assert full[:len(prompt)] == prompt
    comp = full[len(prompt):]
    assert tok.decode(comp) == "Max was happy."
    assert decode_completion(tok, comp, prompt[-1:]) == " Max was happy."
