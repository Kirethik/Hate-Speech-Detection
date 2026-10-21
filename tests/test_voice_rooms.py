import json
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.voice_fakes import FakeDetector, FakeLlm, FakeTts, fake_registry
from voice.rooms import SR, RoomHub, Segmenter, to_pcm16
from voice.types import Seg, Word


class EnergyVad:
    """Stand-in for silero: speech = RMS above a threshold; end after 0.3 s of silence."""

    def __init__(self):
        self.active, self.quiet = False, 0

    def __call__(self, w):
        loud = float(np.sqrt(np.mean(w ** 2))) > 0.02
        if loud and not self.active:
            self.active, self.quiet = True, 0
            return {"start": 0}
        if self.active:
            self.quiet = 0 if loud else self.quiet + 1
            if self.quiet * 512 >= 0.3 * SR:
                self.active = False
                return {"end": 0}
        return None


class ScriptedWhisper:
    """Returns the next scripted sentence for each utterance."""
    name = "scripted"

    def __init__(self, lines):
        self.lines = list(lines)

    def detect_language(self, y):
        return "en", 0.99

    def transcribe(self, y, lang=None):
        t = self.lines.pop(0)
        ws = t.split()
        d = len(y) / SR
        return "en", [Seg(0, d, t, [Word(w, i * d / len(ws), (i + 1) * d / len(ws), 0.9) for i, w in enumerate(ws)], 0.9)]


def tone(sec, f=220):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def frames(y, n=320):
    b = to_pcm16(y)
    return [b[i:i + 2 * n] for i in range(0, len(b), 2 * n)]


def test_segmenter_splits_on_silence():
    seg = Segmenter(EnergyVad())
    y = np.concatenate([np.zeros(SR // 2, np.float32), tone(1.0), np.zeros(SR // 2, np.float32), tone(0.8),
                        np.zeros(SR // 2, np.float32)])
    utts = []
    for f in frames(y):
        utts += seg.feed(f)
    assert len(utts) == 2
    assert 1.0 <= len(utts[0]) / SR <= 1.7 and 0.8 <= len(utts[1]) / SR <= 1.5


def test_segmenter_caps_long_utterances():
    seg = Segmenter(EnergyVad())
    utts = []
    for f in frames(tone(19.0)):
        utts += seg.feed(f)
    assert len(utts) == 2 and all(len(u) / SR <= 8.1 for u in utts)


def _unpack(b):
    n = struct.unpack(">I", b[:4])[0]
    return json.loads(b[4:4 + n]), b[4 + n:]


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAS_ENGINE", "pretrained")
    from server.app import app

    with TestClient(app) as c:
        reg = fake_registry(tmp_path, whisper=ScriptedWhisper([
            "Good morning everyone, thanks for coming to the meeting today.",
            "Those people are a plague on this city.",
        ]))
        c.app.state.voice.reg = reg
        c.app.state.rooms = RoomHub(c.app.state.voice, vad_factory=EnergyVad)
        yield c, reg


def _recv_until(ws, pred, limit=40):
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            if pred("bytes", m["bytes"]):
                return m["bytes"]
        elif m.get("text"):
            d = json.loads(m["text"])
            if pred("json", d):
                return d
    raise AssertionError("expected message not received")


def test_live_room_cleans_hateful_utterance(app_client):
    c, reg = app_client
    with c.websocket_connect("/ws/rooms/team-42") as a, c.websocket_connect("/ws/rooms/team-42") as b:
        a.send_json({"type": "join", "name": "Asha"})
        ja = a.receive_json()
        b.send_json({"type": "join", "name": "Ben"})
        jb = _recv_until(b, lambda k, d: k == "json" and d["type"] == "joined")
        assert ja["type"] == "joined" and len(jb["members"]) == 2
        # Asha says a clean 3.5 s sentence (becomes her voice reference), then a hateful one
        speech = np.concatenate([np.zeros(SR // 4, np.float32), tone(3.5), np.zeros(SR // 2, np.float32),
                                 tone(2.0, 300), np.zeros(SR // 2, np.float32)])
        for f in frames(speech):
            a.send_bytes(f)
        h1, pcm1 = _unpack(_recv_until(b, lambda k, d: k == "bytes"))
        assert not h1["edited"] and h1["name"] == "Asha" and "Good morning" in h1["text"]
        assert abs(len(pcm1) / 2 / SR - 3.5) < 0.8
        h2, pcm2 = _unpack(_recv_until(b, lambda k, d: k == "bytes"))
        assert h2["edited"] and h2["method"] == "tts" and h2["text"].startswith("I am worried")
        assert abs(len(pcm2) / 2 / SR - 2.0) < 0.3  # FakeTts returns 2 s of audio
        # the speaker is told what happened, and never receives her own audio
        r = _recv_until(a, lambda k, d: k == "json" and d["type"] == "self_result" and d["edited"])
        assert "plague" in r["original_text"]
        # the TTS got Asha's clean sentence as the cloning reference
        assert "Good morning" in reg._objs["tts"].calls[0][3]


def test_room_rejects_bad_code_and_requires_join(app_client):
    c, _ = app_client
    with c.websocket_connect("/ws/rooms/a!") as ws:
        assert ws.receive_json()["type"] == "error"


def test_legacy_endpoints_use_pretrained_models(app_client):
    c, _ = app_client
    d = c.post("/api/analyze/text", json={"text": "those people are vermin", "lang_hint": "en", "tone": "firm"}).json()
    assert d["detection"]["is_hate"] and d["detection"]["is_demo_mode"] is False
    assert d["suggestions"]["rewrite"] and d["analysis_id"]
    assert d["detection"]["rationale"][0]["text"] == "vermin"
    ms = c.get("/api/model-status").json()
    assert ms["model_a"]["is_installed"] and not ms["model_b"]["is_demo_mode"]
    assert c.get("/api/health").json()["engine"] == "pretrained"
    hist = c.get("/api/history").json()
    assert any(h["id"] == d["analysis_id"] for h in hist)


def test_rewrite_studio_endpoint(app_client):
    c, reg = app_client

    class L(FakeLlm):
        def rewrite(self, text, lang, tone="calm", n=2):
            return {"rewrites": ["I see this differently.", "they are vermin"], "responses": ["Let's keep it respectful."]}

    reg._objs["llm"] = L()
    r = c.post("/api/rewrite", json={"text": "you idiots", "tone": "professional", "language": "en"}).json()
    assert [x["text"] for x in r["rewrites"]] == ["I see this differently."]  # toxic candidate dropped
    assert r["responses"] and r["requested_tone"] == "professional" and "removed" in r["warning"]


def test_audio_analysis_endpoint(app_client, tmp_path):
    import io

    import soundfile as sf

    from tests.voice_fakes import FakeWhisper, make_audio
    c, reg = app_client
    reg._objs["whisper"] = FakeWhisper()
    buf = io.BytesIO()
    sf.write(buf, make_audio(), 22050, format="WAV")
    d = c.post("/api/analyze/audio", files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
               data={"lang_hint": "en"}).json()
    assert d["input"]["mode"] == "audio" and d["detection"]["is_hate"]
    rat = d["detection"]["rationale"][0]
    assert rat["text"] == "a plague" and 3.0 <= rat["audio_start"] <= 3.2
    assert len(d["sentences"]) == 3
