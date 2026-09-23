import pytest
from dataset import normalize_code_mixed

def test_normalize_offset_length():
    text, offsets = normalize_code_mixed("h@te 100")
    assert len(text) == len(offsets)

def test_leet_reversal():
    text, offsets = normalize_code_mixed("h@te")
    assert text == "hate"
    assert len(text) == len(offsets)

def test_spacing_collapse():
    text, offsets = normalize_code_mixed("h a t e")
    assert text == "hate"
    assert len(text) == len(offsets)

def test_repeated_char_collapse():
    text, offsets = normalize_code_mixed("haaaate")
    assert text == "haate"
    assert len(text) == len(offsets)

def test_emojis_preserved():
    text, offsets = normalize_code_mixed("hate 😡")
    assert "😡" in text
    assert len(text) == len(offsets)

def test_nfkc_normalization():
    text, offsets = normalize_code_mixed("ｈａｔｅ")  # fullwidth
    assert text == "hate"
    assert len(text) == len(offsets)

def test_numbers_not_leet_reversed():
    text, offsets = normalize_code_mixed("100")
    assert text == "100"
    
def test_round_trip_offset():
    orig = "h@te 100 h a t e"
    text, offsets = normalize_code_mixed(orig)
    # The first "hate" maps back to "h@te"
    # "hate" is at indices 0-3 in normalized
    orig_chars = [orig[offsets[i]] for i in range(4)]
    assert "".join(orig_chars) == "h@te"
