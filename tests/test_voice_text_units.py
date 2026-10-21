from voice.sentences import split_sentences
from voice.service import guess_lang
from voice.tts_client import route
from voice.types import Seg, Word
from voice.voice_ref import any_speech, pick_ref


def W(t, a, b):
    return Word(t, a, b, 0.9)


def test_split_on_punctuation_and_pause():
    words = [W("Hello", 0, .3), W("there.", .35, .7), W("How", .8, 1.0), W("are", 1.05, 1.2),
             W("you", 1.25, 1.4), W("doing", 2.5, 2.8), W("today", 2.85, 3.1)]
    out = split_sentences([Seg(0, 3.1, "x", words)])
    assert [s.text for s in out] == ["Hello there.", "How are you", "doing today"]
    assert out[1].start == 0.8 and out[2].start == 2.5


def test_split_handles_devanagari_danda():
    words = [W("मैं", 0, .3), W("ठीक", .35, .6), W("हूँ।", .65, .9), W("आप", 1.0, 1.2), W("कैसे", 1.25, 1.5)]
    out = split_sentences([Seg(0, 1.5, "x", words)])
    assert len(out) == 2


def test_segments_without_words_pass_through():
    s = Seg(0, 4, "chunk from vad")
    assert split_sentences([s]) == [s]


def test_pick_ref_skips_flagged_and_prefers_long_clean_runs():
    segs = [Seg(0, 3, "a", conf=.9), Seg(3.2, 6, "b", conf=.9), Seg(6.2, 9, "bad", conf=.9),
            Seg(9.2, 11, "c", conf=.9)]
    spans, text = pick_ref(segs, {2}, lo=4, hi=12)
    assert (spans, text) == ([(0, 6)], "a b")


def test_pick_ref_joins_short_clean_segments():
    segs = [Seg(0, 2.5, "a", conf=.9), Seg(3, 5.5, "bad", conf=.9), Seg(6, 8.5, "c", conf=.8)]
    spans, text = pick_ref(segs, {1}, lo=4, hi=12)
    assert spans == [(0, 2.5), (6, 8.5)] and text == "a c"


def test_pick_ref_none_then_any_speech():
    segs = [Seg(0, 1.5, "a"), Seg(5, 9, "flagged words here")]
    assert pick_ref(segs, {1}, lo=4) is None
    assert any_speech(segs, lo=3)[0] == [(5, 9)]


def test_tts_routing():
    assert route("ta", {"indicf5", "chatterbox"}) == ("indicf5", "ta")
    assert route("hi", {"indicf5", "chatterbox"}) == ("chatterbox", "hi")
    assert route("hi", {"indicf5"}) == ("indicf5", "hi")
    assert route("ur", {"chatterbox"}) == ("chatterbox", "hi")
    assert route("en", {"indicf5"}) is None
    assert route("en", {"dummy"}) == ("dummy", "en")


def test_guess_lang():
    assert guess_lang("வணக்கம் நண்பா") == "ta"
    assert guess_lang("നമസ്കാരം") == "ml"
    assert guess_lang("నమస్కారం") == "te"
    assert guess_lang("नमस्ते दोस्त") == "hi"
    assert guess_lang("hello friend") == "en"
