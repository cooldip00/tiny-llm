import numpy as np
import pytest

from tinyllm.data import bin_path, load_tokens
from tinyllm.tokenizer import EOS, TOKENIZER_PATH, load_tokenizer

pytestmark = pytest.mark.skipif(
    not (bin_path("validation").exists() and TOKENIZER_PATH.exists()),
    reason="run scripts/train_tokenizer.py and scripts/tokenize_data.py first",
)


def test_val_bin_loads_and_decodes():
    tok = load_tokenizer()
    eos = tok.token_to_id(EOS)
    arr = load_tokens("validation")
    assert isinstance(arr, np.memmap) and arr.dtype == np.uint16
    assert arr[-1] == eos
    assert int(arr.max()) < tok.get_vocab_size()
    first_end = int(np.argmax(arr == eos))
    story = tok.decode(arr[:first_end].tolist())
    assert story and story.isascii() and EOS not in story
