"""Dataset loading and text cleanup (D-008, D-009, D-018).

Every step that reads TinyStories (tokenizer training, tokenization) goes
through `load_clean`, so they all see exactly the same text.
"""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset

DATASET = "roneneldan/TinyStories"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # git-ignored
TOKEN_DTYPE = np.uint16  # vocab 8192 < 65536 (D-009)

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


def bin_path(split: str) -> Path:
    """'train' -> data/train.bin, 'validation' -> data/val.bin"""
    return DATA_DIR / f"{'val' if split == 'validation' else split}.bin"


def write_tokens(split: str, tok, eos_id: int, batch_size: int = 10_000) -> tuple[int, int]:
    """Encode a cleaned split into one flat uint16 file: story EOS story EOS ...

    Streams batch by batch, so RAM use stays small. Writes to a .tmp file and
    renames at the end, so a crash never leaves a half-written .bin behind.
    Returns (n_stories, n_tokens).
    """
    ds = load_clean(split)
    out = bin_path(split)
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".tmp")
    n_tokens = 0
    with open(tmp, "wb") as f:
        for i in range(0, len(ds), batch_size):
            encs = tok.encode_batch(ds[i: i + batch_size]["text"])  # Rust, multi-threaded
            ids = np.fromiter(
                (t for e in encs for t in (*e.ids, eos_id)), dtype=np.int64
            )
            assert ids.max() < np.iinfo(TOKEN_DTYPE).max
            ids.astype(TOKEN_DTYPE).tofile(f)
            n_tokens += len(ids)
    tmp.replace(out)
    return len(ds), n_tokens


def load_tokens(split: str) -> np.memmap:
    """Memory-map a .bin file: no RAM cost, the OS pages it in on demand."""
    return np.memmap(bin_path(split), dtype=TOKEN_DTYPE, mode="r")


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random training windows: x = tokens[s : s+T], y = tokens[s+1 : s+T+1].

    Windows start anywhere and cross story boundaries; EOS tells the model where a
    story ends. Sampling with replacement from a seeded `rng` makes runs reproducible,
    and saving `rng.bit_generator.state` in a checkpoint makes resume exact.
    On CUDA the batch goes through pinned memory so the copy can overlap with compute.
    """
    if len(tokens) < seq_len + 1:
        raise ValueError(f"need at least {seq_len + 1} tokens, got {len(tokens)}")
    starts = rng.integers(0, len(tokens) - seq_len, size=batch_size)    # last valid start: len-T-1
    idx = starts[:, None] + np.arange(seq_len + 1)                       # (B, T+1)
    chunk = torch.from_numpy(tokens[idx].astype(np.int64))               # gather, then widen for embedding
    if torch.device(device).type == "cuda":
        chunk = chunk.pin_memory().to(device, non_blocking=True)
    return chunk[:, :-1], chunk[:, 1:]
