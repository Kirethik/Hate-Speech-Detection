import soundfile as sf

from tests.voice_fakes import SR, FakeLlm, FakeTts, fake_registry, make_audio
from voice.pipeline import JobOptions, process


def _src(tmp_path):
    p = tmp_path / "in.wav"
    sf.write(p, make_audio(), SR)
    return p


def test_end_to_end_replaces_only_the_hateful_sentence(tmp_path):
    reg = fake_registry(tmp_path)
    events = []
    res = process(reg, _src(tmp_path), tmp_path / "out", JobOptions(),
                  lambda *a: events.append(a))
    ae = res["audio_edit"]
    assert [t["flagged"] for t in ae["transcript"]] == [False, True, False]
    assert len(ae["edits"]) == 1
    e = ae["edits"][0]
    assert e["method"] == "tts" and e["replacement_text"].startswith("I am worried")
    assert e["severity"] == "hate" and e["target"] == "nationality_migrant"
    assert 2.9 < e["start"] < 3.2 and e["new_start"] is not None
    assert res["detection"]["is_hate"] is True
    assert (tmp_path / "out" / "cleaned.mp3").exists() and (tmp_path / "out" / "original.mp3").exists()
    y, sr = sf.read(tmp_path / "out" / "cleaned.wav")
    assert sr == SR and abs(len(y) / sr - ae["cleaned_duration_s"]) < 0.01
    assert ae["voice_source"] == "auto"
    assert events[-1][0] == "done"
    # the reference clip came from clean speech and carried its transcript
    tts = reg._objs["tts"]
    assert "Good morning" in tts.calls[0][3] and "plague" not in tts.calls[0][3]


def test_clean_audio_is_untouched(tmp_path):
    class NoHate:
        def score(self, texts):
            return [0.01] * len(texts)

    reg = fake_registry(tmp_path, detector=NoHate())
    res = process(reg, _src(tmp_path), tmp_path / "out", JobOptions())
    assert res["audio_edit"]["edits"] == []
    assert reg._objs["llm"].calls == 0  # gate kept the LLM idle


def test_unsafe_replacement_is_muted(tmp_path):
    reg = fake_registry(tmp_path, llm=FakeLlm(replacement="they are still vermin"))
    res = process(reg, _src(tmp_path), tmp_path / "out", JobOptions())
    e = res["audio_edit"]["edits"][0]
    assert e["method"] == "muted" and e["replacement_text"] == ""


def test_tts_down_mutes(tmp_path):
    reg = fake_registry(tmp_path, tts=FakeTts(up=False))
    res = process(reg, _src(tmp_path), tmp_path / "out", JobOptions())
    assert res["audio_edit"]["edits"][0]["method"] == "muted"


def test_llm_down_falls_back_to_confident_classifier(tmp_path):
    class DeadLlm:
        def judge(self, *a, **k):
            return None

    reg = fake_registry(tmp_path, llm=DeadLlm())
    res = process(reg, _src(tmp_path), tmp_path / "out", JobOptions())
    e = res["audio_edit"]["edits"][0]
    assert e["method"] == "muted" and e["severity"] == "hate"


def test_too_long_audio_rejected(tmp_path):
    reg = fake_registry(tmp_path)
    reg.s.max_audio_s = 5
    try:
        process(reg, _src(tmp_path), tmp_path / "out", JobOptions())
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "limit" in str(e)
