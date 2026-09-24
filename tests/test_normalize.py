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


# --- regressions: ordinary text must survive normalization unchanged --------
@pytest.mark.parametrize("text", [
    "i am a boy",
    "I am a big fan",
    "a b",                       # only two isolated letters
    "covid19 cases rose",
    "mp3 and top10 lists",
    "see you b4 10am, 4k video",
    "1000 people",
    "email me at user@mail.com",
    "thanks @ravi_k",
    "visit www.example.com/aaa",
    "e.g. this",
    "नमस्ते दोस्तों",
    "நீ ஒரு நல்ல மனிதன்",
    "hello!",
])
def test_ordinary_text_unchanged(text):
    out, offsets = normalize_code_mixed(text)
    assert out == text
    assert offsets == list(range(len(text)))


@pytest.mark.parametrize("text,expected", [
    ("h.a.t.e them", "hate them"),
    ("h-a-t-e", "hate"),
    ("you are a b i t c h", "you are a bitch"),
    ("sh1t", "shit"),
    ("1diot", "idiot"),
    ("l0ser", "loser"),
    ("$hit", "shit"),
    ("sh!t", "shit"),
    ("!!!!!", "!!"),
    ("h​ate", "hate"),
    ("h‍ate", "hate"),
    ("“quoted”", '"quoted"'),
])
def test_evasions_are_undone(text, expected):
    out, offsets = normalize_code_mixed(text)
    assert out == expected
    assert len(out) == len(offsets)


def test_offsets_are_monotonic_and_in_range():
    orig = "u r a b i t c h!!!! go to www.x.com h@te"
    out, offsets = normalize_code_mixed(orig)
    assert offsets == sorted(offsets)
    assert all(0 <= o < len(orig) for o in offsets)


def test_to_original_span_maps_back():
    from text_norm import to_original_span
    orig = "you are a b i t c h"
    out, offsets = normalize_code_mixed(orig)
    start = out.index("bitch")
    s, e = to_original_span(start, start + len("bitch"), offsets, orig)
    assert orig[s:e] == "b i t c h"


def test_to_original_span_keeps_indic_vowel_signs():
    from text_norm import to_original_span
    orig = "नमस्ते"
    out, offsets = normalize_code_mixed(orig)
    s, e = to_original_span(0, len(out), offsets, orig)
    assert orig[s:e] == orig


def test_detect_script_kannada_and_urdu():
    from text_norm import detect_script
    assert detect_script("ನೀನು ಒಳ್ಳೆಯವನು") == "native"
    assert detect_script("تم اچھے ہو") == "native"
    assert detect_script("tum acche ho") == "latin"
    assert detect_script("tum अच्छे ho yaar") == "mixed"
