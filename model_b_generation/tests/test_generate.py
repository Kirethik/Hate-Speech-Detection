"""Safety gate and fallbacks in generate.py, with a fake Model B and a fake Model A."""

import pytest

from model_b_generation import gen_metrics, generate
from model_b_generation.generate import (generate_suggestions, mask_spans, masked_rewrite,
                                         reuses_flagged, rewrite_for_speech)

TEXT = "those vermin people should leave our town now"
SPANS = [{"start_char": 6, "end_char": 12, "text": "vermin", "score": 0.9}]


def fake_scorer(bad_words=("vermin", "hate")):
    return lambda texts: [0.95 if any(w in t.lower() for w in bad_words) else 0.05 for t in texts]


@pytest.fixture
def fake_b(monkeypatch):
    calls = {}

    def gen(mb, prompts, k=2, **kw):
        calls["prompts"], calls["kw"] = prompts, kw
        return [list(mb[p.split(" | ")[0]]) for p in prompts]

    import model_b_generation.infer_gen as ig
    monkeypatch.setattr(ig, "generate_candidates", gen)
    return calls


def test_mask_spans_removes_flagged_words():
    assert mask_spans(TEXT, SPANS) == "those people should leave our town now"
    assert mask_spans("x, y", []) == "x, y"


def test_reuses_flagged():
    assert reuses_flagged("they are Vermin", SPANS)
    assert not reuses_flagged("they are neighbours", SPANS)


def test_masked_rewrite_needs_scorer_and_enough_words():
    assert masked_rewrite(TEXT, SPANS, None, 0.5) is None
    assert masked_rewrite(TEXT, SPANS, fake_scorer(), 0.5)[0].startswith("those people")
    short = [{"start_char": 0, "end_char": 36, "text": TEXT[:36]}]
    assert masked_rewrite(TEXT, short, fake_scorer(), 0.5) is None
    assert masked_rewrite(TEXT, SPANS, fake_scorer(("people",)), 0.5) is None  # still flagged


def test_speech_rewrite_prefers_safe_model_b(fake_b):
    mb = {"rewrite": [TEXT, "those vermin must go", "I would rather they moved elsewhere"]}
    r = rewrite_for_speech(TEXT, "en", "unknown", SPANS, mb, fake_scorer(), 0.5, budget_ms=800)
    assert r["method"] == "model_b" and r["text"] == "I would rather they moved elsewhere"
    assert fake_b["prompts"] == ["rewrite | English | unknown: " + TEXT]
    assert fake_b["kw"]["max_time"] == pytest.approx(0.8)


def test_speech_rewrite_falls_back_to_mask_then_none(fake_b):
    mb = {"rewrite": [TEXT, "vermin vermin"]}  # copy + reuses flagged word
    r = rewrite_for_speech(TEXT, "en", None, SPANS, mb, fake_scorer(), 0.5)
    assert r["method"] == "masked"
    assert rewrite_for_speech("vermin all", "en", None,
                              [{"start_char": 0, "end_char": 6, "text": "vermin"}],
                              mb, fake_scorer(), 0.5) is None


def test_no_scorer_means_nothing_generated_is_trusted(fake_b):
    mb = {"rewrite": ["a fine sentence"], "respond": ["a fine reply"]}
    out = generate_suggestions(TEXT, "ta", "religion", mb, None, spans=SPANS)
    assert out["fallback"] and out["rewrite"] == []
    assert out["respond"][0]["template"] and out["respond"][0]["language"] == "ta"


def test_text_mode_returns_both_tasks(fake_b):
    mb = {"rewrite": ["they should rethink this", "hate them"],
          "respond": ["everyone deserves respect", "everyone deserves respect"]}
    out = generate_suggestions(TEXT, "en", "religion", mb, fake_scorer(), spans=SPANS)
    assert [r["text"] for r in out["rewrite"]] == ["they should rethink this"]
    assert [r["text"] for r in out["respond"]] == ["everyone deserves respect"]
    assert not out["fallback"]


def test_unknown_language_templates_fall_back_to_english():
    out = generate_suggestions("x", "kn", None)
    assert out["respond"][0]["language"] == "en"


def test_every_supported_language_has_templates():
    from model_b_generation.prompts import LANGUAGE_NAMES
    missing = set(LANGUAGE_NAMES) - set(generate.FALLBACK_RESPONSES) - {"ur", "kn"}
    assert not missing


def test_metrics_basics():
    assert gen_metrics.chrf("same text", "same text") == pytest.approx(100.0)
    assert gen_metrics.chrf("abc", "xyz") == 0.0
    assert gen_metrics.copy_rate(["Hello there!", "new"], ["hello there", "old"]) == 0.5
    assert gen_metrics.distinct_2(["a b a b"]) == pytest.approx(2 / 3)
    rep = gen_metrics.report(["x y"], ["x y"], ["z"], ["rewrite|en"], lambda t: [0.1], 0.5)
    assert rep["safety_rate"] == 1.0 and rep["by_group"]["rewrite|en"]["n"] == 1


def test_prompt_format():
    from model_b_generation.prompts import build_prompt
    assert build_prompt("respond", "ur_roman", "  x ", None) == "respond | Roman Urdu | unknown: x"
    with pytest.raises(ValueError):
        build_prompt("translate", "en", "x")
