"""Glue between the API and the pipeline: jobs, text analysis, voice enrollment."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from . import audio_io
from .jobs import JobManager, JobStore
from .pipeline import JobOptions, process
from .registry import Registry
from .settings import Settings

SCRIPT_LANG = [  # (unicode range, language) for quick text language guess
    ((0x0B80, 0x0BFF), "ta"), ((0x0C00, 0x0C7F), "te"), ((0x0D00, 0x0D7F), "ml"),
    ((0x0C80, 0x0CFF), "kn"), ((0x0980, 0x09FF), "bn"), ((0x0900, 0x097F), "hi"),
    ((0x0600, 0x06FF), "ur"),
]

ENROLL_PROMPTS = {
    "en": "Every morning I walk to the market, greet my neighbours, and buy fresh vegetables for the day.",
    "hi": "हर सुबह मैं बाज़ार जाता हूँ, अपने पड़ोसियों से मिलता हूँ और दिन भर के लिए ताज़ी सब्ज़ियाँ खरीदता हूँ।",
    "ta": "ஒவ்வொரு காலையும் நான் சந்தைக்குச் சென்று, அண்டை வீட்டாரை வாழ்த்தி, அன்றைக்கான புதிய காய்கறிகளை வாங்குகிறேன்.",
    "te": "ప్రతి ఉదయం నేను మార్కెట్‌కు వెళ్లి, పొరుగువారిని పలకరించి, ఆ రోజుకు తాజా కూరగాయలు కొంటాను.",
    "ml": "എല്ലാ ദിവസവും രാവിലെ ഞാൻ ചന്തയിൽ പോയി, അയൽക്കാരെ അഭിവാദ്യം ചെയ്ത്, അന്നത്തേക്കുള്ള പുതിയ പച്ചക്കറികൾ വാങ്ങുന്നു.",
    "ur": "हर सुबह मैं बाज़ार जाता हूँ, अपने पड़ोसियों से मिलता हूँ और दिन भर के लिए ताज़ी सब्ज़ियाँ खरीदता हूँ।",
}


def guess_lang(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        for (a, b), lang in SCRIPT_LANG:
            if a <= o <= b:
                counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=counts.get) if counts else "en"


class VoiceService:
    def __init__(self, s: Settings, reg: Registry | None = None):
        self.s = s
        self.reg = reg or Registry(s)
        root = Path(s.data_dir)
        self.jobs_dir, self.voices_dir = root / "jobs", root / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(root / "jobs.db")
        self.jobs = JobManager(self.store, self.jobs_dir, self._run)

    def _run(self, jid, src, out_dir, opts, emit):
        return process(self.reg, src, out_dir, opts, emit)

    # ---- voices -----------------------------------------------------------
    def enroll(self, data: bytes, filename: str, language: str, text: str | None) -> dict:
        vid = uuid.uuid4().hex[:12]
        tmp = self.voices_dir / f"{vid}{Path(filename).suffix or '.bin'}"
        tmp.write_bytes(data)
        try:
            y, sr = audio_io.decode(tmp, 24000)
        finally:
            tmp.unlink(missing_ok=True)
        from .splice import trim_silence

        y = trim_silence(y, sr)[: int(self.s.ref_max_s * sr)]
        if len(y) < 3 * sr:
            raise ValueError("recording too short: read the whole sentence (at least 3 seconds)")
        audio_io.save_wav(self.voices_dir / f"{vid}.wav", y, sr)
        meta = {"id": vid, "language": language, "text": text or ENROLL_PROMPTS.get(language, ""),
                "duration_s": round(len(y) / sr, 2), "created_at": time.time()}
        (self.voices_dir / f"{vid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return meta

    def voice(self, vid: str) -> dict | None:
        if not re.fullmatch(r"[0-9a-f]{12}", vid or ""):
            return None
        p = self.voices_dir / f"{vid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def delete_voice(self, vid: str) -> bool:
        if not self.voice(vid):
            return False
        for ext in (".wav", ".json"):
            (self.voices_dir / f"{vid}{ext}").unlink(missing_ok=True)
        return True

    def job_options(self, mode: str, language: str | None, voice_id: str | None, replace_offensive: bool
                    ) -> JobOptions:
        o = JobOptions(mode=mode if mode in ("rewrite", "counter") else "rewrite",
                       language=None if language in (None, "", "auto") else language,
                       replace_offensive=replace_offensive)
        v = self.voice(voice_id) if voice_id else None
        if v:
            o.voice_ref_path = str(self.voices_dir / f"{v['id']}.wav")
            o.voice_ref_text = v["text"]
        return o

    # ---- text -------------------------------------------------------------
    def _verdict(self, context: str, text: str, lang: str, mode: str, tone: str, p: float):
        """LLM verdict when the classifier gate (or language coverage) asks for it."""
        if not (self.s.always_llm or p >= self.s.llm_gate or lang not in self.s.clf_langs):
            return None, False
        try:
            return self.reg.llm().judge(context, text, lang, mode, tone=tone), True
        except Exception:
            return None, True

    def _canonical(self, text: str, lang: str, p: float, v, asked_llm: bool, det, t_clf: float, t_llm: float,
                   mode_in: str = "text", asr_conf=None, t_asr=None, tone: str = "calm") -> dict:
        llm_failed = asked_llm and v is None
        sev = v.severity if v else ("hate" if p >= 0.8 else "offensive" if p >= 0.5 else "none")
        sev_label = {"none": "normal", "offensive": "offensive_profanity", "hate": "hate"}[sev]
        sugg = {"rewrite": [], "respond": [], "fallback": False, "safety_status": "verified_safe"}
        if v and sev != "none":
            for key, txt in (("rewrite", v.rewrite), ("respond", v.counter)):
                if txt:
                    sp = round(det.score([txt])[0], 4)
                    if sp < self.s.safety_threshold:
                        sugg[key].append({"text": txt, "language": lang, "tone": tone,
                                          "safety_check_hate_prob": sp})
            if not (sugg["rewrite"] or sugg["respond"]):
                sugg.update(fallback=True, safety_status="review_required")
        elif llm_failed and sev != "none":
            sugg.update(fallback=True, safety_status="llm_unavailable")
        spans = []
        for sp in (v.spans if v else []):
            i = text.find(sp)
            if i >= 0:
                spans.append({"start_char": i, "end_char": i + len(sp), "text": sp, "score": 1.0})
        warning = None
        if llm_failed:
            warning = (f"The LLM ({self.s.llm_model}) did not respond; verdict is from the classifiers only. "
                       "Start Ollama to get rewrites and explanations.")
        return {
            "input": {"mode": mode_in, "text": text, "language": lang,
                      "script": "latin" if guess_lang(text) == "en" else "native", "asr_confidence": asr_conf},
            "detection": {
                "hate_prob": round(p, 4), "is_hate": sev == "hate",
                "severity": {"label": sev_label, "probs": {}},
                "target": {"label": v.target if v else "none", "probs": {}},
                "rationale": spans,
                "second_stage": {"nli_ran": False, "nli_score": None, "dehumanization_hits": []},
                "decision_path": "classifier+llm" if v else "classifier",
                "reason": v.reason if v else "",
                "is_demo_mode": False,
                "warning": warning,
            },
            "suggestions": sugg,
            "timing_ms": {"asr": t_asr, "model_a": round(t_clf, 1), "nli": None,
                          "model_b": round(t_llm, 1) if asked_llm else None},
        }

    def analyze_text(self, text: str, language: str | None, mode: str = "rewrite", tone: str = "calm") -> dict:
        t0 = time.perf_counter()
        lang = language if language and language not in ("auto", "") else guess_lang(text)
        lang = "ur" if lang == "ur_roman" else lang
        det = self.reg.detector()
        p = det.score([text])[0]
        t_clf = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        v, asked = self._verdict(text, text, lang, mode, tone, p)
        return self._canonical(text, lang, p, v, asked, det, t_clf, (time.perf_counter() - t1) * 1000, tone=tone)

    # ---- audio (analysis only, no re-voicing) ------------------------------
    def analyze_audio(self, data: bytes, filename: str, language: str | None, tone: str = "calm") -> dict:
        """Transcribe, then analyse each sentence; report the worst one with audio-timed rationale."""
        from .asr import transcribe

        tmp = self.voices_dir.parent / "tmp" / f"{uuid.uuid4().hex}{Path(filename).suffix or '.bin'}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        try:
            y16, _ = audio_io.decode(tmp, 16000)
        finally:
            tmp.unlink(missing_ok=True)
        if len(y16) / 16000 > self.s.max_audio_s:
            raise ValueError(f"audio longer than {self.s.max_audio_s:.0f}s")
        t0 = time.perf_counter()
        lang_hint = None if language in (None, "", "auto") else ("ur" if language == "ur_roman" else language)
        lang, segs, _ = transcribe(self.reg, y16, lang_hint)
        t_asr = (time.perf_counter() - t0) * 1000
        full = " ".join(x.text for x in segs).strip()
        if not full:
            raise ValueError("no speech found in the recording")
        det = self.reg.detector()
        t0 = time.perf_counter()
        ps = det.score([x.text for x in segs])
        t_clf = (time.perf_counter() - t0) * 1000
        worst = max(range(len(segs)), key=lambda i: ps[i])
        t1 = time.perf_counter()
        v, asked = self._verdict(full, segs[worst].text, lang, "rewrite", tone, ps[worst])
        res = self._canonical(full, lang, ps[worst], v, asked, det, t_clf, (time.perf_counter() - t1) * 1000,
                              "audio", round(sum(x.conf for x in segs) / len(segs), 3), round(t_asr, 1), tone)
        # rationale: flag the worst sentence's spans with audio times
        off = full.find(segs[worst].text)
        rat = []
        for sp in (v.spans if v else []):
            j = segs[worst].text.find(sp)
            if j >= 0 and off >= 0:
                rat.append({"start_char": off + j, "end_char": off + j + len(sp), "text": sp, "score": 1.0,
                            "audio_start": round(segs[worst].start, 2), "audio_end": round(segs[worst].end, 2)})
        if res["detection"]["severity"]["label"] != "normal" and not rat and off >= 0:
            rat.append({"start_char": off, "end_char": off + len(segs[worst].text), "text": segs[worst].text,
                        "score": round(ps[worst], 3), "audio_start": round(segs[worst].start, 2),
                        "audio_end": round(segs[worst].end, 2)})
        res["detection"]["rationale"] = rat
        res["sentences"] = [{**x.to_dict(), "p_toxic": round(ps[i], 4)} for i, x in enumerate(segs)]
        return res

    # ---- rewrite studio -----------------------------------------------------
    def rewrite(self, text: str, tone: str = "calm", language: str | None = None, n: int = 2) -> dict:
        lang = language if language and language not in ("auto", "") else guess_lang(text)
        lang = "ur" if lang == "ur_roman" else lang
        det = self.reg.detector()
        out = None
        try:
            out = self.reg.llm().rewrite(text, lang, tone, n)
        except Exception:
            out = None
        rewrites, responses, dropped = [], [], 0
        for key, dst in (("rewrites", rewrites), ("responses", responses)):
            for t in (out or {}).get(key, []):
                sp = det.score([t])[0]
                if sp < self.s.safety_threshold:
                    dst.append({"text": t, "language": lang, "tone": tone, "safety_check_hate_prob": round(sp, 4)})
                else:
                    dropped += 1
        status = "verified_safe" if (rewrites or responses) else "review_required"
        warning = None
        if out is None:
            warning = f"The LLM ({self.s.llm_model}) did not respond. Start Ollama and try again."
        elif dropped:
            warning = f"{dropped} candidate(s) were removed because they still scored as toxic."
        return {"original_text": text, "requested_tone": tone, "language": lang, "rewrites": rewrites,
                "responses": responses, "safety_status": status, "model_identifier": self.s.llm_model,
                "is_demo_mode": False, "warning": warning}

    # ---- status ---------------------------------------------------------------
    def model_status(self) -> dict:
        st = self.reg.status()
        det_ok = "detector" not in st["errors"]
        dev = "cpu"
        try:
            from devices import resolve

            dev = str(resolve(self.s.detector_device))
        except Exception:
            pass
        return {
            "model_a": {"name": "Hate speech detector (pretrained)", "model_type": "hate_detector",
                        "is_installed": det_ok, "weights_path": " + ".join(self.s.clf_models), "device": dev,
                        "is_demo_mode": False,
                        "status_message": ("Loaded" if st["loaded"].get("detector") else "Loads on first use")
                        if det_ok else f"Error: {st['errors']['detector']}"},
            "model_b": {"name": "Rewrite / counter-narrative LLM", "model_type": "alternative_generator",
                        "is_installed": True, "weights_path": f"{self.s.llm_model} @ {self.s.llm_base_url}",
                        "device": "ollama", "is_demo_mode": False,
                        "status_message": "Served by the local LLM server"},
        }
