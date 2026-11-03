"""Byte-level BPE tokenizer, vocab 8192 (D-007).

Byte-level means every string can be encoded (unknown characters fall back to
raw bytes), and decode(encode(text)) == text exactly.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 8192
EOS = "<|endoftext|>"  # separates stories in the token stream; id 0
TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "tokenizer" / "tinystories-bpe8192.json"


def train_tokenizer(texts: Iterable[str] | Iterator[list[str]], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    # GPT-2-style split into words/punctuation/spaces before BPE, no prefix space
    # (so the roundtrip is exact).
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes always present
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Tokenizer:
    return Tokenizer.from_file(str(path))
