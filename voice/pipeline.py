"""
End-to-end processing of one recording:

  decode -> ASR (+language ID) -> classifiers -> LLM judge/rewrite (gated)
  -> safety gate -> voice reference -> voice-cloned TTS -> splice -> save

`emit(stage, progress, message)` reports progress (used for the live progress bar).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from . import audio_io
from .asr import transcribe
from .registry import Registry
from .splice import soft_beep, splice, trim_silence
from .types import Edit, Seg, Verdict
from .voice_ref import any_speech, pick_ref

log = logging.getLogger("civitas.pipeline")

Emit = Callable[[str, float, str], None]
REF_SR = 24000


@dataclass
class JobOptions:
    mode: str = "rewrite"                  # rewrite | counter
    language: Optional[str] = None         # None = auto-detect
    replace_offensive: bool = True         # also replace insults/profanity (not only hate)
    voice_ref_path: Optional[str] = None   # enrolled voice wav (exact text in voice_ref_text)
    voice_ref_text: Optional[str] = None


@dataclass
class Timer:
    t: dict = field(default_factory=dict)

    def __call__(self, k: str, t0: float):
        self.t[k] = round((time.perf_counter() - t0) * 1000, 1)


def script_of(text: str) -> str:
    return "native" if any(ord(c) > 0x024F for c in text if c.isalpha()) else "latin"


def _judge_all(reg: Registry, segs: list[Seg], p: list[float], lang: str, opts: JobOptions,
               emit: Emit) -> dict[int, Verdict]:
    s = reg.s
    context = " ".join(x.text for x in segs)
    cand = [i for i, x in enumerate(segs)
            if s.always_llm or p[i] >= s.llm_gate or lang not in s.clf_langs]
    out: dict[int, Verdict] = {}
    llm = None
    try:
        llm = reg.llm()
    except Exception as e:
        log.warning("LLM unavailable: %s", e)
    for n, i in enumerate(cand):
        emit("judge", 0.35 + 0.2 * n / max(1, len(cand)), f"Checking sentence {n + 1}/{len(cand)}")
        v = llm.judge(context, segs[i].text, lang, opts.mode) if llm else None
        if v is None and p[i] >= 0.8:  # LLM down: trust a confident classifier, mute the sentence
            v = Verdict(True, "hate", "other", [], "", "", "", "classifier only (LLM unavailable)",
                        source="classifier")
        if v is None or v.severity == "none":
            continue
        if v.severity == "offensive" and not opts.replace_offensive:
            continue
        out[i] = v
    return out


def _safe_text(reg: Registry, v: Verdict, seg: Seg, context: str, lang: str, opts: JobOptions
               ) -> tuple[str, float | None]:
    """Replacement text that passes the classifier gate, or '' (mute)."""
    if not v.tts_text:
        return "", None
    det, thr = reg.detector(), reg.s.safety_threshold
    p = det.score([v.tts_text])[0]
    if p < thr:
        return v.tts_text, p
    try:
        v2 = reg.llm().judge(context, seg.text, lang, opts.mode,
                             feedback=f"Your previous replacement was still flagged as toxic: "
                                      f"'{v.tts_text}'. Write a clearly non-toxic one.")
    except Exception:
        v2 = None
    if v2 and v2.tts_text:
        p2 = det.score([v2.tts_text])[0]
        if p2 < thr:
            v.rewrite, v.counter, v.tts_text = v2.rewrite, v2.counter, v2.tts_text
            return v2.tts_text, p2
    return "", p


def _ref_audio(y: np.ndarray, sr: int, segs: list[Seg], flagged: set[int], opts: JobOptions, s
               ) -> tuple[bytes, str, str] | None:
    """(wav bytes @24k, exact transcript, source) for voice cloning."""
    if opts.voice_ref_path and Path(opts.voice_ref_path).exists():
        return Path(opts.voice_ref_path).read_bytes(), opts.voice_ref_text or "", "enrolled"
    r = pick_ref(segs, flagged, s.ref_min_s, s.ref_max_s)
    src = "auto"
    if r is None:
        r, src = any_speech(segs, 3.0, s.ref_max_s), "auto_fallback"
    if r is None:
        return None
    spans, text = r
    gap = np.zeros(int(0.15 * REF_SR), dtype=np.float32)
    pieces = []
    for a, b in spans:
        pieces += [audio_io.resample(audio_io.slice_s(y, sr, a, b), sr, REF_SR), gap]
    clip = np.concatenate(pieces[:-1])
    buf = io.BytesIO()
    sf.write(buf, clip, REF_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue(), text, src


def process(reg: Registry, src_path: str | Path, out_dir: str | Path, opts: JobOptions,
            emit: Emit = lambda *a: None) -> dict:
    s, tm, out_dir = reg.s, Timer(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    # 1. decode ---------------------------------------------------------------
    emit("decode", 0.02, "Reading audio")
    t0 = time.perf_counter()
    y, sr = audio_io.decode(src_path)
    dur = len(y) / sr
    if dur > s.max_audio_s:
        raise ValueError(f"audio is {dur:.0f}s; the limit is {s.max_audio_s:.0f}s")
    if dur < 0.5:
        raise ValueError("audio is too short")
    y16 = audio_io.resample(y, sr, 16000)
    audio_io.save_mp3(out_dir / "original.mp3", y, sr)
    tm("decode", t0)

    # 2. ASR ------------------------------------------------------------------
    emit("asr", 0.08, "Transcribing speech")
    t0 = time.perf_counter()
    lang, segs, asr_name = transcribe(reg, y16, opts.language)
    tm("asr", t0)

    # 3. classifiers ----------------------------------------------------------
    emit("detect", 0.30, "Scanning for hate speech")
    t0 = time.perf_counter()
    p = reg.detector().score([x.text for x in segs]) if segs else []
    tm("classifiers", t0)

    # 4. LLM judge + rewrite ----------------------------------------------------
    t0 = time.perf_counter()
    verdicts = _judge_all(reg, segs, p, lang, opts, emit) if segs else {}
    tm("llm", t0)

    # 5. safety gate + voice ref + TTS -----------------------------------------
    context = " ".join(x.text for x in segs)
    edits: list[Edit] = []
    wavs: list[np.ndarray] = []
    ref = None
    voice_source = None
    t_tts = time.perf_counter()
    if verdicts:
        emit("voice", 0.58, "Learning the speaker's voice")
        ref = _ref_audio(y, sr, segs, set(verdicts), opts, s)
        voice_source = ref[2] if ref else None
        tts = None
        up: set[str] = set()
        try:
            tts = reg.tts()
            up = {k for k, ok in tts.health().items() if ok}
        except Exception as e:
            log.warning("TTS unavailable: %s", e)
        for n, (i, v) in enumerate(sorted(verdicts.items())):
            emit("tts", 0.62 + 0.28 * n / len(verdicts), f"Re-voicing sentence {n + 1}/{len(verdicts)}")
            seg = segs[i]
            text, sp = _safe_text(reg, v, seg, context, lang, opts)
            w, eng = None, None
            if text and ref and tts and up:
                try:
                    w, wsr, eng = tts.synth(text, lang, ref[0], ref[1], up)
                    w = trim_silence(audio_io.resample(w, wsr, sr), sr)
                except Exception as e:
                    log.warning("TTS failed for sentence %d: %s", i, e)
                    w = None
            method = "tts" if w is not None and len(w) else "muted"
            if method == "muted":
                w = soft_beep(seg.end - seg.start, sr)
            edits.append(Edit(i, seg.start, seg.end, seg.text, text if method == "tts" else "",
                              v.severity, v.target, v.spans, float(p[i]), method, v.reason,
                              tts_engine=eng, safety_p=sp))
            wavs.append(w)
    tm("tts", t_tts)

    # 6. splice + save --------------------------------------------------------
    emit("splice", 0.92, "Stitching the cleaned audio")
    t0 = time.perf_counter()
    raw = [{**e.to_dict(), "wav": w} for e, w in zip(edits, wavs)]
    for r, e in zip(raw, edits):  # unrounded times for cutting
        r["start"], r["end"] = e.start, e.end
    clean, log_ = splice(y, sr, raw) if raw else (y, [])
    audio_io.save_mp3(out_dir / "cleaned.mp3", clean, sr)
    audio_io.save_wav(out_dir / "cleaned.wav", clean, sr)
    tm("splice", t0)
    tm("total", t_all)

    flagged = {e["seg_index"] for e in log_}
    edit_dicts = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in log_]
    full_text = " ".join(x.text for x in segs)
    emit("done", 1.0, f"{len(edit_dicts)} sentence(s) replaced")
    return {
        "input": {"mode": "audio", "text": full_text, "language": lang, "script": script_of(full_text),
                  "asr_confidence": round(float(np.mean([x.conf for x in segs])), 3) if segs else None},
        "detection": {
            "is_hate": any(e["severity"] == "hate" for e in edit_dicts),
            "hate_prob": round(max(p), 4) if p else 0.0,
            "flagged_sentences": len(edit_dicts),
            "decision_path": "classifier+llm",
        },
        "audio_edit": {
            "original_url": "original.mp3",
            "cleaned_url": "cleaned.mp3",
            "duration_s": round(dur, 3),
            "cleaned_duration_s": round(len(clean) / sr, 3),
            "sample_rate": sr,
            "language": lang,
            "mode": opts.mode,
            "voice_source": voice_source,
            "asr_model": asr_name,
            "edits": edit_dicts,
            "transcript": [{**x.to_dict(), "index": i, "p_toxic": round(float(p[i]), 4),
                            "flagged": i in flagged,
                            "verdict": verdicts[i].to_dict() if i in verdicts else None}
                           for i, x in enumerate(segs)],
        },
        "timing_ms": tm.t,
    }
