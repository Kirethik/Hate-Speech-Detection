"""Fake models for voice-pipeline tests (no downloads, runs in seconds)."""

import numpy as np

from voice.registry import Registry
from voice.settings import Settings
from voice.types import Seg, Verdict, Word

SR = 22050
# three "sentences": tone 0-2.5 s, tone 3.0-5.5 s, tone 6.0-8.5 s (silence between)
SENTS = [(0.1, 2.4, "Good morning everyone, thanks for coming today."),
         (3.1, 5.4, "Those people are a plague on this city."),
         (6.1, 8.4, "Let us start the meeting now please.")]


def make_audio(sr=SR):
    y = np.zeros(int(9.0 * sr), dtype=np.float32)
    for i, (a, b, _) in enumerate(SENTS):
        t = np.arange(int((b - a) * sr)) / sr
        y[int(a * sr): int(a * sr) + len(t)] = 0.3 * np.sin(2 * np.pi * (180 + 60 * i) * t)
    return y


def _words(a, b, text):
    ws = text.split()
    step = (b - a) / len(ws)
    return [Word(w, a + k * step, a + (k + 1) * step - 0.02, 0.9) for k, w in enumerate(ws)]


class FakeWhisper:
    name = "fake-whisper"

    def detect_language(self, y16):
        return "en", 0.99

    def transcribe(self, y16, lang=None):
        segs = [Seg(SENTS[0][0], SENTS[2][1], " ".join(s[2] for s in SENTS),
                    sum((_words(*s) for s in SENTS), []), 0.9)]
        return "en", segs


class FakeDetector:
    names = ["fake"]

    def score(self, texts):
        return [0.95 if ("plague" in t or "vermin" in t) else 0.02 for t in texts]


class FakeLlm:
    name = "fake-llm"

    def __init__(self, replacement="I am worried about how our city is changing."):
        self.replacement = replacement
        self.calls = 0

    def judge(self, context, sentence, lang, mode="rewrite", feedback="", tone="calm"):
        self.calls += 1
        if "plague" not in sentence and "vermin" not in sentence:
            return Verdict(False, "none", "none", [], "", "", "", "fine")
        txt = self.replacement
        spans = [w for w in ("a plague", "vermin") if w in sentence]
        return Verdict(True, "hate", "nationality_migrant", spans, txt,
                       "Blaming a whole group is unfair.", txt, "dehumanising")

    def ping(self):
        return True


class FakeTts:
    def __init__(self, up=True):
        self.up = up
        self.calls = []

    def health(self):
        return {"dummy": self.up}

    def synth(self, text, lang, ref_wav, ref_text, up=None):
        self.calls.append((text, lang, len(ref_wav), ref_text))
        sr = 24000
        t = np.arange(int(2.0 * sr)) / sr
        return (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr, "dummy"


def fake_registry(tmp_path, **over):
    s = Settings(data_dir=str(tmp_path / "voice"))
    objs = {"whisper": FakeWhisper(), "indic_asr": None, "detector": FakeDetector(),
            "llm": FakeLlm(), "tts": FakeTts()}
    objs.update(over)
    return Registry(s, overrides=objs)
