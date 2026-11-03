"""Step 1.2: train the byte-level BPE tokenizer on cleaned TinyStories train text.

Run:  uv run python scripts/train_tokenizer.py
Output: tokenizer/tinystories-bpe8192.json (small, committed to git)
"""

import time

from tinyllm.data import load_clean
from tinyllm.tokenizer import EOS, TOKENIZER_PATH, VOCAB_SIZE, load_tokenizer, train_tokenizer


def batches(ds, size: int = 10_000):
    for i in range(0, len(ds), size):
        yield ds[i: i + size]["text"]


def main() -> None:
    train = load_clean("train")
    t0 = time.perf_counter()
    tok = train_tokenizer(batches(train))
    print(f"\ntrained in {time.perf_counter() - t0:.0f}s")

    TOKENIZER_PATH.parent.mkdir(exist_ok=True)
    tok.save(str(TOKENIZER_PATH))
    tok = load_tokenizer()  # check the saved file, not the in-memory object
    print(f"saved {TOKENIZER_PATH}  ({TOKENIZER_PATH.stat().st_size / 1e3:.0f} KB)")

    vocab = tok.get_vocab_size()
    assert vocab == VOCAB_SIZE, vocab
    assert tok.token_to_id(EOS) == 0
    print(f"vocab {vocab}, {EOS} id {tok.token_to_id(EOS)}")

    # Compression and exact roundtrip on the whole validation split.
    val = load_clean("validation")["text"]
    encs = tok.encode_batch(val)
    n_tokens = sum(len(e.ids) for e in encs)
    n_chars = sum(len(t) for t in val)
    bad = sum(tok.decode(e.ids) != t for e, t in zip(encs, val))
    print(f"val: {n_tokens:,} tokens, {n_chars / n_tokens:.2f} chars/token, "
          f"{n_tokens / len(val):.0f} tokens/story, roundtrip mismatches {bad}")
    assert bad == 0

    sample = val[0][:120]
    ids = tok.encode(sample).ids
    print(f"\nsample: {sample!r}")
    print("tokens:", [tok.id_to_token(i) for i in ids][:30])
    print("last merges learned:", [tok.id_to_token(i) for i in range(VOCAB_SIZE - 10, VOCAB_SIZE)])


if __name__ == "__main__":
    main()
