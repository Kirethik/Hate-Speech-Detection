"""HTTP client for the TTS services (each engine runs in its own venv/process)."""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf

log = logging.getLogger("civitas.tts")

INDICF5_LANGS = {"ta", "te", "ml", "kn", "bn", "mr", "gu", "pa", "or", "as"}
CHATTERBOX_LANGS = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
                    "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}


def route(lang: str, available: set[str]) -> tuple[str, str] | None:
    """(engine, language code to send) for a language, honouring what's running."""
    prefs: list[tuple[str, str]] = []
    if lang in INDICF5_LANGS:
        prefs.append(("indicf5", lang))
    if lang == "hi":
        prefs += [("chatterbox", "hi"), ("indicf5", "hi")]
    elif lang == "ur":
        prefs.append(("chatterbox", "hi"))  # Urdu speech via Hindi voice + Devanagari text
    elif lang in CHATTERBOX_LANGS:
        prefs.append(("chatterbox", lang))
    prefs += [("omnivoice", lang), ("dummy", lang)]
    for eng, code in prefs:
        if eng in available:
            return eng, code
    return None


class TtsClient:
    def __init__(self, routes: dict[str, str], timeout: float = 300.0):
        import httpx

        self.routes = routes
        # TTS services are internal: never send them through a system HTTP proxy
        self.http = httpx.Client(timeout=timeout, trust_env=False)

    def health(self) -> dict[str, bool]:
        out = {}
        for eng, url in self.routes.items():
            try:
                out[eng] = self.http.get(f"{url}/health", timeout=3).status_code == 200
            except Exception:
                out[eng] = False
        return out

    def synth(self, text: str, lang: str, ref_wav: bytes, ref_text: str, up: set[str] | None = None
              ) -> tuple[np.ndarray, int, str]:
        up = up if up is not None else {k for k, v in self.health().items() if v}
        r = route(lang, up)
        if r is None:
            raise RuntimeError(f"no TTS engine running for language '{lang}' (running: {sorted(up)})")
        eng, code = r
        resp = self.http.post(f"{self.routes[eng]}/synthesize",
                              data={"text": text, "lang": code, "ref_text": ref_text},
                              files={"ref_audio": ("ref.wav", ref_wav, "audio/wav")})
        resp.raise_for_status()
        y, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr, eng
