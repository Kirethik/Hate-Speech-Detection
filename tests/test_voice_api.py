import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.voice_routes import router
from tests.voice_fakes import SR, fake_registry, make_audio
from voice.service import VoiceService


@pytest.fixture
def client(tmp_path):
    reg = fake_registry(tmp_path)
    app = FastAPI()
    app.include_router(router)

    @app.on_event("startup")
    async def _s():
        app.state.voice = VoiceService(reg.s, reg)
        await app.state.voice.jobs.start()

    with TestClient(app) as c:
        yield c


def _wav_bytes(y=None, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, make_audio() if y is None else y, sr, format="WAV")
    return buf.getvalue()


def _wait(client, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.1)
    raise TimeoutError


def test_job_lifecycle(client):
    r = client.post("/api/v1/jobs", files={"file": ("talk.wav", _wav_bytes(), "audio/wav")},
                    data={"mode": "counter", "language": "auto"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    j = _wait(client, jid)
    assert j["status"] == "done", j
    assert j["options"]["mode"] == "counter"
    res = client.get(f"/api/v1/jobs/{jid}/result").json()
    ae = res["audio_edit"]
    assert ae["cleaned_url"] == f"/api/v1/media/{jid}/cleaned.mp3"
    assert len(ae["edits"]) == 1
    m = client.get(ae["cleaned_url"])
    assert m.status_code == 200 and m.headers["content-type"] == "audio/mpeg" and len(m.content) > 1000
    assert client.get(f"/api/v1/media/{jid}/input.wav").status_code == 404
    ev = client.get(f"/api/v1/jobs/{jid}/events")
    assert "done" in ev.text
    lst = client.get("/api/v1/jobs").json()
    assert lst[0]["id"] == jid and lst[0]["summary"]["edits"] == 1


def test_rejects_bad_uploads(client):
    assert client.post("/api/v1/jobs", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/v1/jobs", files={"file": ("x.wav", b"", "audio/wav")}).status_code == 400
    assert client.get("/api/v1/jobs/../../etc").status_code == 404
    assert client.get("/api/v1/jobs/zzzzzzzzzzzz").status_code == 404


def test_failed_job_reports_error(client):
    r = client.post("/api/v1/jobs", files={"file": ("bad.wav", b"not really audio", "audio/wav")})
    j = _wait(client, r.json()["job_id"])
    assert j["status"] == "failed" and j["error"]
    assert client.get(f"/api/v1/jobs/{j['id']}/result").status_code == 409


def test_text_analysis(client):
    r = client.post("/api/v1/analyze/text", json={"text": "Those people are vermin"})
    d = r.json()
    assert d["detection"]["is_hate"] and d["input"]["language"] == "en"
    assert d["suggestions"]["rewrite"] and d["detection"]["rationale"][0]["text"] == "vermin"
    d = client.post("/api/v1/analyze/text", json={"text": "Have a nice day"}).json()
    assert not d["detection"]["is_hate"] and d["suggestions"]["rewrite"] == []
    assert client.post("/api/v1/analyze/text", json={"text": "  "}).status_code == 400


def test_voice_enrollment_used_by_jobs(client):
    t = np.arange(int(5 * SR)) / SR
    y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    p = client.get("/api/v1/voices/prompt", params={"language": "ta"}).json()
    r = client.post("/api/v1/voices/enroll", files={"file": ("v.wav", _wav_bytes(y), "audio/wav")},
                    data={"language": "ta", "text": p["text"]})
    assert r.status_code == 200, r.text
    vid = r.json()["id"]
    assert client.get(f"/api/v1/voices/{vid}").json()["text"] == p["text"]
    r = client.post("/api/v1/jobs", files={"file": ("talk.wav", _wav_bytes(), "audio/wav")},
                    data={"voice_id": vid})
    j = _wait(client, r.json()["job_id"])
    res = client.get(f"/api/v1/jobs/{j['id']}/result").json()
    assert res["audio_edit"]["voice_source"] == "enrolled"
    assert client.delete(f"/api/v1/voices/{vid}").status_code == 200
    assert client.get(f"/api/v1/voices/{vid}").status_code == 404
    short = client.post("/api/v1/voices/enroll", files={"file": ("v.wav", _wav_bytes(y[: SR]), "audio/wav")})
    assert short.status_code == 422
