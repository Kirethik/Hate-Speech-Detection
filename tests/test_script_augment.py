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
