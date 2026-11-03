from tinyllm.data import clean_text


def test_ascii_unchanged():
    s = 'Lily said, "Hi!" It\'s a sunny day.\n\nThe end.'
    assert clean_text(s) == s


def test_mojibake_quotes():
    # Real example from the train split: the closing quote lost its last byte.
    s = "a voice shouted, â€œBe careful not to touch it!â€ Annie jumped."
    assert clean_text(s) == 'a voice shouted, "Be careful not to touch it!" Annie jumped.'


def test_mojibake_apostrophe_dash_ellipsis():
    s = "Tom didnâ€™t know â€“ he waitedâ€¦ â€˜okâ€™"
    assert clean_text(s) == "Tom didn't know - he waited... 'ok'"


def test_mojibake_closing_quote_before_punctuation():
    assert clean_text("He said â€œyesâ€.") == 'He said "yes".'


def test_real_typographic_punctuation():
    assert clean_text("“Hi,” she said — it’s fine…") == '"Hi," she said - it\'s fine...'


def test_drop_empty_and_whitespace():
    assert clean_text("") is None
    assert clean_text("  \n ") is None


def test_drop_leftover_non_ascii():
    assert clean_text("She ate at the cafÃ© today.") is None
    assert clean_text("She ate at the café today.") is None


def test_strips_outer_whitespace():
    assert clean_text("  Once upon a time.\n") == "Once upon a time."
