import pytest
from script_augment import ScriptAugmenter

def test_p_augment_1():
    aug = ScriptAugmenter(p_augment=1.0, langs=["ur_roman"])
    res = aug.augment("آ", language="ur_roman", script="native")
    # 'آ' -> 'aa'
    assert res.text == "aa"
    assert res.script == "latin"

def test_p_augment_0():
    aug = ScriptAugmenter(p_augment=0.0, langs=["ur_roman"])
    res = aug.augment("آ", language="ur_roman", script="native")
    assert res.text == "آ"
    assert res.script == "native"

def test_spans_valid_false():
    aug = ScriptAugmenter(p_augment=1.0, langs=["ur_roman"])
    res = aug.augment("آ", language="ur_roman", script="native")
    # 'آ' is len 1, 'aa' is len 2
    assert res.spans_valid is False

def test_english_never_augmented():
    aug = ScriptAugmenter(p_augment=1.0, langs=["hi", "ta"])
    res = aug.augment("hello", language="en", script="latin")
    assert res.text == "hello"
    assert res.script == "latin"

def test_urdu_native_to_roman():
    aug = ScriptAugmenter(p_augment=1.0, langs=["ur_roman"])
    res = aug.augment("ا", language="ur_roman", script="native")
    assert res.text == "a"
    assert ord(res.text[0]) < 128


# --- transliteration must actually change the script (it silently no-opped before) ---
translit = pytest.importorskip("indic_transliteration")


@pytest.mark.parametrize("text,lang", [
    ("नमस्ते दोस्तों", "hi"),
    ("நீ ஒரு நல்ல மனிதன்", "ta"),
    ("నువ్వు మంచివాడివి", "te"),
    ("തെണ്ടി പട്ടി", "ml"),
    ("ನೀನು ಒಳ್ಳೆಯವನು", "kn"),
    ("تم اچھے ہو", "ur"),
])
def test_native_to_latin_produces_plain_ascii(text, lang):
    from script_augment import ScriptAugmenter
    from text_norm import detect_script
    out = ScriptAugmenter(p_augment=1.0).augment(text, lang, "native")
    assert out.script == "latin"
    assert detect_script(out.text) == "latin"
    assert out.text.isascii(), out.text
    assert out.text == out.text.lower()


def test_latin_to_native_hindi_has_no_word_final_virama():
    from script_augment import ScriptAugmenter
    out = ScriptAugmenter(p_augment=1.0).augment("tum bahut acche ho", "hi", "latin")
    assert out.script == "native"
    assert "् " not in out.text + " "
