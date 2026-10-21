import io
import os

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient


def test_dummy_engine_roundtrip(monkeypatch):
    monkeypatch.setenv("CIVITAS_TTS_ENGINE", "dummy")
    from tts_service.app import app

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, np.float32), 16000, format="WAV")
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["engine"] == "dummy"
        r = c.post("/synthesize", data={"text": "hello there", "lang": "en"},
                   files={"ref_audio": ("r.wav", buf.getvalue(), "audio/wav")})
        assert r.status_code == 200
        y, sr = sf.read(io.BytesIO(r.content))
        assert sr == 24000 and len(y) > 0.5 * sr
        assert c.post("/synthesize", data={"text": " ", "lang": "en"},
                      files={"ref_audio": ("r.wav", buf.getvalue(), "audio/wav")}).status_code == 400
