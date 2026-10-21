"""
Lazy, thread-safe holder for every model the voice pipeline uses.
Models load on first use (one instance per process) so the API starts fast.
Tests replace the loaders with fakes via Registry(overrides=...).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from .settings import Settings

log = logging.getLogger("civitas.registry")


class Registry:
    def __init__(self, s: Settings, overrides: dict[str, Any] | None = None):
        self.s = s
        self._lock = threading.RLock()
        self._objs: dict[str, Any] = dict(overrides or {})
        self._errors: dict[str, str] = {}

    def _get(self, key: str, load: Callable[[], Any], optional: bool = False):
        with self._lock:
            if key in self._objs:
                return self._objs[key]
            if key in self._errors and optional:
                return None
            try:
                log.info("loading %s ...", key)
                obj = load()
                self._objs[key] = obj
                self._errors.pop(key, None)
                log.info("loaded %s", key)
                return obj
            except Exception as e:
                self._errors[key] = f"{type(e).__name__}: {e}"
                log.warning("failed to load %s: %s", key, e)
                if optional:
                    self._objs[key] = None
                    return None
                raise

    # ---- models -----------------------------------------------------------
    def whisper(self):
        from .asr import WhisperAsr

        s = self.s
        return self._get("whisper", lambda: WhisperAsr(s.whisper_model, s.whisper_device, s.whisper_compute,
                                                        s.whisper_threads, s.whisper_beam))

    def indic_asr(self):
        if not self.s.enable_indic_asr:
            return None
        from .asr import IndicAsr

        s = self.s
        return self._get("indic_asr", lambda: IndicAsr(s.indic_asr_model, s.indic_decoding, s.hf_token),
                         optional=True)

    def detector(self):
        from devices import resolve

        from .detect import Detector

        s = self.s
        return self._get("detector", lambda: Detector(s.clf_models, resolve(s.detector_device), s.hf_token))

    def llm(self):
        from .llm import Llm

        s = self.s
        return self._get("llm", lambda: Llm(s.llm_base_url, s.llm_model, s.llm_api_key, s.llm_timeout_s,
                                            s.llm_temperature, reasoning_effort=s.llm_reasoning_effort))

    def tts(self):
        from .tts_client import TtsClient

        s = self.s
        return self._get("tts", lambda: TtsClient(s.tts_route_map, s.tts_timeout_s))

    # ---- status -----------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            loaded = {k: v is not None for k, v in self._objs.items()}
            return {"loaded": loaded, "errors": dict(self._errors)}

    def warmup(self, parts=("whisper", "detector", "indic_asr")):
        for p in parts:
            try:
                getattr(self, p)()
            except Exception:
                pass
