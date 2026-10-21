"""
/api/v1 routes for Civitas Voice (pretrained-model pipeline).

POST /api/v1/jobs                  upload audio -> {job_id}
GET  /api/v1/jobs                  recent jobs
GET  /api/v1/jobs/{id}             status/progress
GET  /api/v1/jobs/{id}/events      Server-Sent Events progress stream
GET  /api/v1/jobs/{id}/result      full result (transcript, edits, urls)
GET  /api/v1/media/{id}/{file}     original.mp3 / cleaned.mp3 / cleaned.wav
POST /api/v1/analyze/text          text analysis with the same models
GET  /api/v1/voices/prompt         sentence to read for voice enrollment
POST /api/v1/voices/enroll         enroll a voice -> {id}
DELETE /api/v1/voices/{id}
GET  /api/v1/system                which components are loaded / reachable
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from voice.service import ENROLL_PROMPTS, VoiceService

router = APIRouter(prefix="/api/v1", tags=["voice"])
MEDIA_FILES = {"original.mp3", "cleaned.mp3", "cleaned.wav"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".opus", ".flac", ".aac", ".wma", ".amr", ".3gp"}


def svc(request: Request) -> VoiceService:
    v = getattr(request.app.state, "voice", None)
    if v is None:
        raise HTTPException(503, "voice service not started")
    return v


def _jid(jid: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        raise HTTPException(404, "job not found")
    return jid


@router.post("/jobs")
async def create_job(request: Request, file: UploadFile = File(...), mode: str = Form("rewrite"),
                     language: str = Form("auto"), voice_id: str = Form(""),
                     replace_offensive: bool = Form(True)):
    s = svc(request)
    name = file.filename or "audio.webm"
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in AUDIO_EXT and not (file.content_type or "").startswith(("audio/", "video/")):
        raise HTTPException(415, "unsupported file type: upload an audio file")
    data = await file.read()
    if len(data) > s.s.max_upload_mb * 1e6:
        raise HTTPException(413, f"file too large (limit {s.s.max_upload_mb:.0f} MB)")
    if not data:
        raise HTTPException(400, "empty file")
    opts = s.job_options(mode, language, voice_id or None, replace_offensive)
    jid = s.jobs.submit(name if ext else name + ".webm", data, opts)
    return {"job_id": jid, "status": "queued", "position": s.jobs.position(jid)}


@router.get("/jobs")
async def list_jobs(request: Request, limit: int = 50):
    return svc(request).store.list(min(max(limit, 1), 200))


@router.get("/jobs/{jid}")
async def get_job(request: Request, jid: str):
    s = svc(request)
    j = s.store.get(_jid(jid))
    if not j:
        raise HTTPException(404, "job not found")
    j["position"] = s.jobs.position(jid) if j["status"] == "queued" else 0
    return j


@router.get("/jobs/{jid}/result")
async def job_result(request: Request, jid: str):
    j = svc(request).store.get(_jid(jid), with_result=True)
    if not j:
        raise HTTPException(404, "job not found")
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    res = j["result"]
    ae = res.get("audio_edit", {})
    for k in ("original_url", "cleaned_url"):
        if ae.get(k) and not ae[k].startswith("/"):
            ae[k] = f"/api/v1/media/{jid}/{ae[k]}"
    res["job_id"] = jid
    return res


@router.get("/jobs/{jid}/events")
async def job_events(request: Request, jid: str):
    s = svc(request)
    j = s.store.get(_jid(jid))
    if not j:
        raise HTTPException(404, "job not found")
    q = s.jobs.subscribe(jid)

    async def gen():
        try:
            cur = s.store.get(jid)
            yield f"data: {json.dumps({k: cur[k] for k in ('id', 'status', 'stage', 'progress', 'message', 'error')})}\n\n"
            if cur["status"] in ("done", "failed"):
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("status") in ("done", "failed"):
                    return
        finally:
            s.jobs.unsubscribe(jid, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/media/{jid}/{name}")
async def media(request: Request, jid: str, name: str):
    if name not in MEDIA_FILES:
        raise HTTPException(404, "not found")
    p = svc(request).jobs.job_dir(_jid(jid)) / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="audio/mpeg" if name.endswith(".mp3") else "audio/wav",
                        filename=f"civitas-{jid}-{name}")


class TextReq(BaseModel):
    text: str
    language: str | None = "auto"
    mode: str = "rewrite"
    tone: str = "calm"


@router.post("/analyze/text")
async def analyze_text(request: Request, req: TextReq):
    if not req.text.strip():
        raise HTTPException(400, "text cannot be empty")
    if len(req.text) > 5000:
        raise HTTPException(413, "text too long (5000 characters max)")
    s = svc(request)
    return await asyncio.get_running_loop().run_in_executor(
        None, s.analyze_text, req.text.strip(), req.language, req.mode, req.tone)


@router.get("/voices/prompt")
async def voice_prompt(language: str = "en"):
    return {"language": language, "text": ENROLL_PROMPTS.get(language, ENROLL_PROMPTS["en"])}


@router.post("/voices/enroll")
async def enroll(request: Request, file: UploadFile = File(...), language: str = Form("en"),
                 text: str = Form("")):
    s = svc(request)
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, s.enroll, data, file.filename or "voice.webm", language, text or None)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/voices/{vid}")
async def get_voice(request: Request, vid: str):
    v = svc(request).voice(vid)
    if not v:
        raise HTTPException(404, "voice not found")
    return v


@router.delete("/voices/{vid}")
async def delete_voice(request: Request, vid: str):
    if not svc(request).delete_voice(vid):
        raise HTTPException(404, "voice not found")
    return {"deleted": vid}


@router.get("/system")
async def system(request: Request):
    s = svc(request)
    loop = asyncio.get_running_loop()

    def probe():
        out = {"settings": {"whisper_model": s.s.whisper_model, "llm_model": s.s.llm_model,
                            "llm_base_url": s.s.llm_base_url, "tts_routes": s.s.tts_route_map,
                            "detector_device": s.s.detector_device},
               "registry": s.reg.status()}
        try:
            from devices import describe

            out["device"] = describe()
        except Exception as e:  # pragma: no cover
            out["device"] = {"error": str(e)}
        try:
            out["llm_reachable"] = s.reg.llm().ping()
        except Exception:
            out["llm_reachable"] = False
        try:
            out["tts"] = s.reg.tts().health()
        except Exception:
            out["tts"] = {}
        return out

    return await loop.run_in_executor(None, probe)


@router.get("/eval")
async def eval_report(request: Request):
    p = Path(svc(request).s.data_dir) / "eval_latest.json"
    if not p.exists():
        raise HTTPException(404, "no evaluation yet: run scripts/evaluate.py")
    return json.loads(p.read_text(encoding="utf-8"))
