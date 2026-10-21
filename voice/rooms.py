"""
Live rooms: people talk, others hear a moderated version a moment later.

Protocol (WebSocket /ws/rooms/{room_id}):
  client -> server  text  {"type": "join", "name": str, "language": "auto"|code, "mode": "rewrite"|"counter",
                           "voice_id": str|null}
                    binary  16 kHz mono PCM16 little-endian frames (any size, typically 20 ms)
                    text  {"type": "mute", "muted": bool} | {"type": "leave"}
  server -> client  text  {"type": "joined", "member_id", "members": [...], "room": id}
                    text  {"type": "presence", "members": [...]}
                    text  {"type": "caption", ...}         every utterance (cleaned text), to everyone
                    text  {"type": "self_result", ...}     to the speaker: what happened to their utterance
                    text  {"type": "state", "member_id", "state": "speaking"|"processing"|"idle"}
                    binary  [4-byte big-endian header length][JSON header][PCM16 16 kHz audio]
                            cleaned audio for listeners (never sent back to the speaker)

Each utterance waits for the end of the sentence (VAD silence), so listeners hear speech
roughly "utterance length + 1-3 s" later, like a broadcast delay.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

from . import audio_io
from .splice import match_loud, soft_beep, trim_silence

log = logging.getLogger("civitas.rooms")

SR = 16000
WIN = 512                 # silero VAD window at 16 kHz
MAX_UTTER_S = 8.0
MIN_UTTER_S = 0.4
MAX_MEMBERS = 8
REF_MAX_S = 12.0


# ---------------------------------------------------------------------------- VAD segmenter
class SileroStream:
    """Per-speaker streaming VAD (each stream needs its own model state)."""

    def __init__(self, min_silence_ms: int = 450, threshold: float = 0.5):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        self._torch = torch
        self.it = VADIterator(load_silero_vad(), threshold=threshold, sampling_rate=SR,
                              min_silence_duration_ms=min_silence_ms, speech_pad_ms=60)

    def __call__(self, win: np.ndarray):
        return self.it(self._torch.from_numpy(win), return_seconds=False)

    def reset(self):
        self.it.reset_states()


class Segmenter:
    """Turns a stream of PCM16 frames into utterances (float32 arrays at 16 kHz)."""

    def __init__(self, vad, pre_roll_s: float = 0.25):
        self.vad = vad
        self.buf = np.zeros(0, dtype=np.float32)     # unprocessed samples (< WIN)
        self.hist = np.zeros(0, dtype=np.float32)    # recent audio for pre-roll
        self.cur: list[np.ndarray] | None = None
        self.cur_len = 0
        self.pre = int(pre_roll_s * SR)

    def feed(self, pcm16: bytes) -> list[np.ndarray]:
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, x])
        out = []
        while len(self.buf) >= WIN:
            w, self.buf = self.buf[:WIN], self.buf[WIN:]
            ev = self.vad(w)
            if self.cur is None:
                self.hist = np.concatenate([self.hist, w])[-self.pre:]
                if ev and "start" in ev:
                    self.cur, self.cur_len = [self.hist.copy()], len(self.hist)
                continue
            self.cur.append(w)
            self.cur_len += len(w)
            ended = bool(ev and "end" in ev)
            if ended or self.cur_len >= MAX_UTTER_S * SR:
                u = np.concatenate(self.cur)
                # at the length cap speech is still going: start the next utterance right away
                self.cur, self.cur_len = (None, 0) if ended else ([], 0)
                self.hist = np.zeros(0, dtype=np.float32)
                if len(u) >= MIN_UTTER_S * SR:
                    out.append(u)
        return out

    @property
    def speaking(self) -> bool:
        return self.cur is not None


# ---------------------------------------------------------------------------- members / rooms
@dataclass
class Member:
    id: str
    name: str
    ws: object
    language: str | None = None
    mode: str = "rewrite"
    voice_ref: tuple[bytes, str] | None = None        # enrolled (wav bytes, exact text)
    auto_ref: list[tuple[np.ndarray, str]] = field(default_factory=list)  # clean utterances (16 kHz)
    muted: bool = False
    seg: Segmenter | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seq: int = 0
    detected_lang: str | None = None

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "muted": self.muted,
                "voice": "enrolled" if self.voice_ref else "learning"}

    def ref(self) -> tuple[bytes, str] | None:
        if self.voice_ref:
            return self.voice_ref
        if not self.auto_ref:
            return None
        clip = np.concatenate([np.concatenate([a, np.zeros(int(0.15 * SR), np.float32)]) for a, _ in self.auto_ref])
        if len(clip) < 3 * SR:
            return None
        buf = io.BytesIO()
        sf.write(buf, audio_io.resample(clip, SR, 24000), 24000, format="WAV", subtype="PCM_16")
        return buf.getvalue(), " ".join(t for _, t in self.auto_ref)

    def learn(self, y: np.ndarray, text: str):
        """Keep up to ~12 s of this speaker's clean utterances as the cloning reference."""
        if self.voice_ref or len(y) < 1.0 * SR or len(y) > REF_MAX_S * SR:
            return
        self.auto_ref.append((y, text))
        while sum(len(a) for a, _ in self.auto_ref) > REF_MAX_S * SR:
            self.auto_ref.pop(0)


@dataclass
class Room:
    id: str
    members: dict[str, Member] = field(default_factory=dict)
    context: list[str] = field(default_factory=list)  # recent cleaned captions for LLM context

    def roster(self) -> list[dict]:
        return [m.public() for m in self.members.values()]


def pack(header: dict, pcm16: bytes) -> bytes:
    h = json.dumps(header, ensure_ascii=False).encode()
    return struct.pack(">I", len(h)) + h + pcm16


def to_pcm16(y: np.ndarray) -> bytes:
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


# ---------------------------------------------------------------------------- hub
class RoomHub:
    def __init__(self, service, vad_factory: Callable[[], object] | None = None):
        self.svc = service
        self.rooms: dict[str, Room] = {}
        self.vad_factory = vad_factory or SileroStream
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="civitas-live")

    # ---- websocket session -------------------------------------------------
    async def serve(self, ws, room_id: str):
        from starlette.websockets import WebSocketDisconnect

        await ws.accept()
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", room_id):
            await ws.send_json({"type": "error", "message": "invalid room code"})
            await ws.close()
            return
        room = self.rooms.setdefault(room_id, Room(room_id))
        if len(room.members) >= MAX_MEMBERS:
            await ws.send_json({"type": "error", "message": f"room is full ({MAX_MEMBERS} people max)"})
            await ws.close()
            return
        member = None
        try:
            hello = await asyncio.wait_for(ws.receive_json(), timeout=30)
            if hello.get("type") != "join":
                raise ValueError("first message must be join")
            member = Member(uuid.uuid4().hex[:8], str(hello.get("name") or "Guest")[:40], ws,
                            language=None if hello.get("language") in (None, "", "auto") else
                            ("ur" if hello["language"] == "ur_roman" else hello["language"]),
                            mode=hello.get("mode") if hello.get("mode") in ("rewrite", "counter") else "rewrite")
            v = self.svc.voice(hello.get("voice_id") or "")
            if v:
                member.voice_ref = ((self.svc.voices_dir / f"{v['id']}.wav").read_bytes(), v["text"])
            member.seg = Segmenter(await asyncio.get_running_loop().run_in_executor(self.pool, self.vad_factory))
            room.members[member.id] = member
            await ws.send_json({"type": "joined", "member_id": member.id, "room": room_id,
                                "members": room.roster()})
            await self._broadcast(room, {"type": "presence", "members": room.roster()})
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if not member.muted:
                        await self._on_audio(room, member, msg["bytes"])
                elif msg.get("text"):
                    ctl = json.loads(msg["text"])
                    if ctl.get("type") == "leave":
                        break
                    if ctl.get("type") == "mute":
                        member.muted = bool(ctl.get("muted"))
                        await self._broadcast(room, {"type": "presence", "members": room.roster()})
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        except Exception as e:  # pragma: no cover
            log.exception("room session error")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if member and member.id in room.members:
                del room.members[member.id]
                await self._broadcast(room, {"type": "presence", "members": room.roster()})
            if not room.members:
                self.rooms.pop(room_id, None)
            try:
                await ws.close()
            except Exception:
                pass

    async def _on_audio(self, room: Room, m: Member, data: bytes):
        was = m.seg.speaking
        utts = m.seg.feed(data)
        if m.seg.speaking and not was:
            await self._broadcast(room, {"type": "state", "member_id": m.id, "state": "speaking"})
        for u in utts:
            m.seq += 1
            asyncio.create_task(self._handle(room, m, u, m.seq))

    async def _handle(self, room: Room, m: Member, y: np.ndarray, seq: int):
        async with m.lock:  # keep each speaker's utterances in order
            await self._broadcast(room, {"type": "state", "member_id": m.id, "state": "processing"})
            t0 = time.perf_counter()
            try:
                r = await asyncio.get_running_loop().run_in_executor(self.pool, self.process, room, m, y)
            except Exception as e:
                log.exception("utterance failed")
                r = {"skip": False, "text": "", "edited": False, "audio": y, "error": str(e)}
            await self._broadcast(room, {"type": "state", "member_id": m.id, "state": "idle"})
            if r.get("skip"):
                return
            lat = round((time.perf_counter() - t0) * 1000)
            head = {"type": "utterance", "from": m.id, "name": m.name, "seq": seq, "text": r["text"],
                    "edited": r["edited"], "method": r.get("method"), "severity": r.get("severity"),
                    "language": r.get("language"), "latency_ms": lat, "duration_s": round(len(r["audio"]) / SR, 2)}
            caption = {**head, "type": "caption"}
            payload = pack(head, to_pcm16(r["audio"]))
            for other in list(room.members.values()):
                try:
                    if other.id == m.id:
                        await other.ws.send_json({**head, "type": "self_result",
                                                  "original_text": r.get("original_text", "")})
                    else:
                        await other.ws.send_bytes(payload)
                        await other.ws.send_json(caption)
                except Exception:
                    pass

    # ---- the per-utterance pipeline (runs in a worker thread) ---------------
    def process(self, room: Room, m: Member, y: np.ndarray) -> dict:
        from .asr import transcribe

        svc, s = self.svc, self.svc.s
        lang, segs, _ = transcribe(svc.reg, y, m.language or m.detected_lang)
        text = " ".join(x.text for x in segs).strip()
        if not text:
            return {"skip": True}
        if not m.language:
            m.detected_lang = lang
        det = svc.reg.detector()
        p = det.score([text])[0]
        context = " ".join(room.context[-6:] + [text])
        v, _ = svc._verdict(context, text, lang, m.mode, "calm", p)
        flagged = (v is not None and v.severity != "none") or (v is None and p >= 0.8)
        if not flagged:
            m.learn(y, text)
            room.context.append(text)
            return {"text": text, "edited": False, "audio": y, "language": lang}
        repl, audio, method = "", None, "muted"
        if v and v.tts_text and det.score([v.tts_text])[0] < s.safety_threshold:
            repl = v.tts_text
            ref = m.ref()
            if ref:
                try:
                    w, wsr, _ = svc.reg.tts().synth(repl, lang, ref[0], ref[1])
                    w = trim_silence(audio_io.resample(w, wsr, SR), SR)
                    if len(w):
                        audio, method = match_loud(w, y, SR), "tts"
                except Exception as e:
                    log.warning("live TTS failed: %s", e)
        if audio is None:
            audio = soft_beep(min(len(y) / SR, 2.0), SR)
        shown = (v.rewrite if m.mode == "rewrite" else v.counter) if (v and method == "tts") else "[removed]"
        room.context.append(shown)
        return {"text": shown or repl, "original_text": text, "edited": True, "audio": audio, "method": method,
                "severity": v.severity if v else "hate", "language": lang}

    async def _broadcast(self, room: Room, msg: dict):
        for mem in list(room.members.values()):
            try:
                await mem.ws.send_json(msg)
            except Exception:
                pass
