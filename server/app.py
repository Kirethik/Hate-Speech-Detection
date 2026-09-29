
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from config import (
    API_HOST, API_PORT, CORS_ORIGINS, DB_PATH, HISTORY_LIMIT, MAX_AUDIO_MB,
)
from server.models import *
from server.db import init_db, save_analysis, get_history, save_feedback
from pipeline.registry import ModelRegistry
from pipeline.analyze import analyze_text, analyze_audio

_registry: ModelRegistry | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry
    init_db(DB_PATH)
    _registry = ModelRegistry()
    yield

app = FastAPI(title="Civitas AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/api/analyze/text", response_model=AnalysisResult)
async def analyze_text_endpoint(req: TextAnalysisRequest):
    if not req.text.strip():
        raise HTTPException(400, "text cannot be empty")
    result = analyze_text(req.text, lang_hint=req.lang_hint, registry=_registry)
    analysis_id = save_analysis(DB_PATH, result)
    result["analysis_id"] = analysis_id
    return result

@app.post("/api/analyze/audio", response_model=AnalysisResult)
async def analyze_audio_endpoint(file: UploadFile = File(...), lang_hint: str | None = None):
    if file.size and file.size > MAX_AUDIO_MB * 1e6:
        raise HTTPException(413, f"File too large")
    audio_bytes = await file.read()
    result = analyze_audio(audio_bytes, lang_hint=lang_hint, registry=_registry)
    analysis_id = save_analysis(DB_PATH, result)
    result["analysis_id"] = analysis_id
    return result

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

@app.post("/api/suggest")
async def suggest_endpoint(req: SuggestRequest):
    from model_b_generation.generate import generate_suggestions
    # Phase 4 wires Model A/B from the registry; until then templates only
    return generate_suggestions(req.text, req.lang, "unknown")

@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    fid = save_feedback(DB_PATH, req.model_dump())
    return {"feedback_id": fid}

@app.get("/api/history")
async def history_endpoint(limit: int = HISTORY_LIMIT):
    return get_history(DB_PATH, limit)

@app.get("/api/health")
async def health_endpoint():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    vram = _registry.vram_report() if _registry else {}
    return {
        "status": "ok",
        "mock_mode": _registry.mock_mode() if _registry else True,
        "device": device,
        "vram": vram,
        "version": "1.0.0",
    }

if __name__ == "__main__":
    uvicorn.run("server.app:app", host=API_HOST, port=API_PORT, reload=True)
