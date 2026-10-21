"""
Minimal TTS microservice. One engine per process:

    set CIVITAS_TTS_ENGINE=chatterbox   (or indicf5 / omnivoice / dummy)
    python -m uvicorn tts_service.app:app --host 127.0.0.1 --port 9002

POST /synthesize  form: text, lang, ref_text ; file: ref_audio (wav)  ->  audio/wav
GET  /health      200 once the model is loaded
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from tts_service.engines import load

try:  # pick up HF_TOKEN / CIVITAS_TTS_DEVICE from the repo .env
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("civitas.tts")

STATE: dict = {"engine": None, "lock": threading.Lock()}
REF_DIR = Path(tempfile.gettempdir()) / "civitas_tts_refs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    name = os.getenv("CIVITAS_TTS_ENGINE", "dummy")
    STATE["engine"] = load(name)  # blocks startup until ready, so /health means "ready"
    REF_DIR.mkdir(parents=True, exist_ok=True)
    log.info("engine %s ready", name)
    yield


app = FastAPI(title="Civitas TTS", lifespan=lifespan)


@app.get("/health")
def health():
    e = STATE["engine"]
    return {"status": "ok", "engine": e.name, "device": getattr(e, "device", "cpu"), "sample_rate": e.sr,
            "needs_ref_text": e.needs_ref_text}


def _synth(text: str, lang: str, ref_text: str, ref: bytes) -> bytes:
    e = STATE["engine"]
    ref_path = REF_DIR / f"{hashlib.sha1(ref).hexdigest()}.wav"
    if not ref_path.exists():
        ref_path.write_bytes(ref)
    with STATE["lock"]:  # one synthesis at a time per process
        t0 = time.perf_counter()
        y, sr = e.synth(text, lang, str(ref_path), ref_text)
        log.info("%s: %d chars -> %.1fs audio in %.1fs", e.name, len(text), len(y) / sr, time.perf_counter() - t0)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/synthesize")
async def synthesize(text: str = Form(...), lang: str = Form("en"), ref_text: str = Form(""),
                     ref_audio: UploadFile = File(...)):
    text = text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    e = STATE["engine"]
    if e.needs_ref_text and not ref_text.strip():
        raise HTTPException(400, f"{e.name} needs ref_text (exact transcript of the reference audio)")
    ref = await ref_audio.read()
    if not ref:
        raise HTTPException(400, "empty reference audio")
    try:
        wav = await run_in_threadpool(_synth, text, lang, ref_text.strip(), ref)
    except Exception as ex:
        log.exception("synthesis failed")
        raise HTTPException(500, f"synthesis failed: {ex}")
    return Response(wav, media_type="audio/wav")
