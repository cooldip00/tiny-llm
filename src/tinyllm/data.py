"""Dataset loading and text cleanup (D-008, D-009, D-018).

Every step that reads TinyStories (tokenizer training, tokenization) goes
through `load_clean`, so they all see exactly the same text.
"""

from datasets import Dataset, load_dataset

DATASET = "roneneldan/TinyStories"

# D-018: TinyStories has UTF-8 punctuation that was decoded as Windows-1252
# ("mojibake"), e.g. “ -> "â€œ". Map it back to plain ASCII.
# Order matters: the 3-character sequences must run before the bare "â€".
_REPLACEMENTS: list[tuple[str, str]] = [
    ("â€œ", '"'),    # “ left double quote
    ("â€\x9d", '"'), # ” right double quote (when the 3rd byte survived)
    ("â€™", "'"),    # ’ apostrophe / right single quote
    ("â€˜", "'"),    # ‘ left single quote
    ("â€“", "-"),    # – en dash
    ("â€”", "-"),    # — em dash
    ("â€¦", "..."),  # … ellipsis
    ("â€", '"'),     # ” right double quote whose 3rd byte (0x9D) was lost
    ("Â\xa0", " "),  # no-break space
    ("Â\xad", ""),   # soft hyphen
    # Real (correctly encoded) typographic punctuation -> ASCII.
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ("–", "-"), ("—", "-"), ("…", "..."), ("\xa0", " "), ("\xad", ""),
]


def clean_text(text: str) -> str | None:
    """Return cleaned ASCII text, or None if the story should be dropped.

    Dropped: empty stories, and stories that still contain non-ASCII
    characters after the replacements (rare leftovers such as 'Ã©').
    """
    if not text.isascii():
        for bad, good in _REPLACEMENTS:
            if bad in text:
                text = text.replace(bad, good)
    text = text.strip()
    if not text or not text.isascii():
        return None
    return text


def _clean_batch(batch: dict) -> dict:
    return {"text": [clean_text(t) or "" for t in batch["text"]]}


def _keep_batch(batch: dict) -> list[bool]:
    return [t != "" for t in batch["text"]]


def load_clean(split: str, num_proc: int = 4) -> Dataset:
    """Load a TinyStories split ('train' or 'validation') with D-018 cleanup applied.

    `datasets` caches the result, so only the first call does the work.
    """
    ds = load_dataset(DATASET, split=split)
    # With split=... (and no streaming) this is always a Dataset. The check narrows
    # the declared Union return type for type checkers, and fails loudly otherwise.
    assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds).__name__}"
    ds = ds.map(_clean_batch, batched=True, num_proc=num_proc, desc=f"clean {split}")
    return ds.filter(_keep_batch, batched=True, num_proc=num_proc, desc=f"drop {split}")
