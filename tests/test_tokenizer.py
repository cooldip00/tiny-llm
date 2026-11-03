import pytest

from tinyllm.tokenizer import EOS, TOKENIZER_PATH, VOCAB_SIZE, load_tokenizer

pytestmark = pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="run scripts/train_tokenizer.py first")


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_vocab_and_eos(tok):
    assert tok.get_vocab_size() == VOCAB_SIZE
    assert tok.token_to_id(EOS) == 0
    assert VOCAB_SIZE <= 2**16  # token ids fit in uint16 (D-009)


@pytest.mark.parametrize("text", [
    'Once upon a time, Lily said, "Hi!"\n\nThe end.',
    "  leading and trailing spaces  ",
    "numbers 12345 and symbols #@$%^&*() {}[]",
    "tabs\tand\r\nwindows newlines",
    "non-ASCII still roundtrips via bytes: café, 日本, 🙂",
    "",
])
def test_roundtrip_exact(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_common_words_are_single_tokens(tok):
    for word in [" the", " and", " Lily", " said", " happy"]:
        assert len(tok.encode(word).ids) == 1, word
