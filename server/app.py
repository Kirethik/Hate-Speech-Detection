"""
Civitas AI API.

CIVITAS_ENGINE selects what backs the /api/analyze, /api/rewrite and /api/suggest endpoints:
  pretrained (default)  downloaded pretrained models via the voice pipeline (voice/)
  legacy                the original Model A / Model B code path (Colab-trained weights or demo mode)
The /api/v1 routes (Voice Studio, live rooms) always use the pretrained pipeline.
"""

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from config import API_HOST, API_PORT, CORS_ORIGINS, DB_PATH, HISTORY_LIMIT, MAX_AUDIO_MB
from pipeline.analyze import analyze_audio, analyze_text
from pipeline.registry import ModelRegistry
from server.db import get_analytics_summary, get_history, init_db, save_analysis, save_feedback
from server.models import AnalysisResult, FeedbackRequest, RewriteRequest, SuggestRequest, TextAnalysisRequest
from server.voice_routes import router as voice_router
from src.models.alternative_generator import generate_alternative
from src.models.model_loader import get_system_model_status

_registry: ModelRegistry | None = None


def engine() -> str:
    return os.getenv("CIVITAS_ENGINE", "pretrained").strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry
    init_db(DB_PATH)
    _registry = ModelRegistry()
    # Civitas Voice (pretrained-model pipeline): job queue + lazy model registry + live rooms
    from voice.rooms import RoomHub
    from voice.service import VoiceService
    from voice.settings import get_settings

    app.state.voice = VoiceService(get_settings())
    app.state.rooms = RoomHub(app.state.voice)
    await app.state.voice.jobs.start()
    if os.getenv("CIVITAS_WARMUP", "0") == "1":
        threading.Thread(target=app.state.voice.reg.warmup, daemon=True).start()
    yield
    await app.state.voice.jobs.stop()


app = FastAPI(title="Civitas AI API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # credentials + "*" is rejected by browsers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voice_router)


async def _off_loop(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


# --- Core Analysis Endpoints ---

@app.post("/api/analyze/text", response_model=AnalysisResult)
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_text_endpoint(req: TextAnalysisRequest, request: Request):
    if not req.text.strip():
        raise HTTPException(400, "text cannot be empty")
    if engine() == "pretrained":
        result = await _off_loop(request.app.state.voice.analyze_text, req.text.strip(), req.lang_hint,
                                 "rewrite", req.tone or "calm")
    else:
        result = analyze_text(req.text, lang_hint=req.lang_hint, tone=req.tone or "calm", registry=_registry)
    result["analysis_id"] = save_analysis(DB_PATH, result)
    return result


@app.post("/api/analyze/audio", response_model=AnalysisResult)
async def analyze_audio_endpoint(
    request: Request,
    file: UploadFile = File(...),
    lang_hint: Optional[str] = Form(None),
    tone: Optional[str] = Form("calm"),
):
    if file.size and file.size > MAX_AUDIO_MB * 1e6:
        raise HTTPException(413, "File too large")
    audio_bytes = await file.read()
    if engine() == "pretrained":
        try:
            result = await _off_loop(request.app.state.voice.analyze_audio, audio_bytes,
                                     file.filename or "audio.webm", lang_hint, tone or "calm")
        except ValueError as e:
            raise HTTPException(422, str(e))
    else:
        result = analyze_audio(audio_bytes, lang_hint=lang_hint, tone=tone or "calm", registry=_registry)
    result["analysis_id"] = save_analysis(DB_PATH, result)
    return result


# --- Alternative Generation ---

@app.post("/api/rewrite")
@app.post("/rewrite")
async def rewrite_endpoint(req: RewriteRequest, request: Request):
    if not req.text.strip():
        raise HTTPException(400, "text cannot be empty")
    if engine() == "pretrained":
        return await _off_loop(request.app.state.voice.rewrite, req.text.strip(), req.tone or "calm",
                               req.language)
    alt = generate_alternative(text=req.text, tone=req.tone or "calm", language=req.language or "en",
                               target=req.target)
    return alt.to_dict()


@app.post("/api/suggest")
async def suggest_legacy_endpoint(req: SuggestRequest, request: Request):
    if engine() == "pretrained":
        r = await _off_loop(request.app.state.voice.rewrite, req.text, req.tone or "calm", req.lang)
        return {"rewrite": r["rewrites"], "respond": r["responses"], "fallback": r["safety_status"] != "verified_safe"}
    alt = generate_alternative(text=req.text, tone=req.tone or "calm", language=req.lang or "en")
    return {
        "rewrite": [r.to_dict() for r in alt.rewrites],
        "respond": [r.to_dict() for r in alt.responses],
        "fallback": alt.safety_status != "verified_safe",
    }


# --- Streaming WebSockets ---

@app.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    from pipeline.stream import StreamSession

    await ws.accept()
    session = StreamSession(registry=_registry)
    try:
        while True:
            chunk = await ws.receive_bytes()
            await session.push_chunk(chunk)
            result = await session.next_result()
            if result:
                await ws.send_json({"type": "utterance_result", **result})
    except Exception:
        await ws.close()


@app.websocket("/ws/rooms/{room_id}")
async def room_endpoint(ws: WebSocket, room_id: str):
    await ws.app.state.rooms.serve(ws, room_id)


# --- Feedback & History ---

@app.post("/api/feedback")
@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    fid = save_feedback(DB_PATH, req.model_dump())
    return {"feedback_id": fid, "status": "recorded"}


@app.get("/api/history")
async def history_endpoint(limit: int = HISTORY_LIMIT):
    return get_history(DB_PATH, limit)


# --- Analytics & Model Status ---

@app.get("/api/analytics")
@app.get("/analytics")
async def analytics_endpoint():
    return get_analytics_summary(DB_PATH)


def _model_status(request: Request) -> dict:
    if engine() == "pretrained":
        return request.app.state.voice.model_status()
    return {k: v.to_dict() for k, v in get_system_model_status().items()}


@app.get("/api/model-status")
@app.get("/model-status")
async def model_status_endpoint(request: Request):
    return _model_status(request)


@app.get("/api/health")
@app.get("/health")
async def health_endpoint(request: Request):
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else (
            "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    except Exception:
        device = "cpu"
    vram = _registry.vram_report() if _registry else {}
    return {
        "status": "ok",
        "engine": engine(),
        "mock_mode": engine() != "pretrained" and (_registry.mock_mode() if _registry else True),
        "device": device,
        "vram": vram,
        "models": _model_status(request),
        "version": "2.0.0",
    }


# --- Web app (production) ---
# One origin serves API + UI: set CIVITAS_STATIC_DIR to the built frontend (frontend/dist).
# Mounted last so every /api, /ws and /health route above wins over the static files.
_static = os.getenv("CIVITAS_STATIC_DIR", "")
if _static and os.path.isdir(_static):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static, html=True), name="web")


if __name__ == "__main__":
    uvicorn.run("server.app:app", host=API_HOST, port=API_PORT, reload=True)
