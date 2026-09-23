
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float

@dataclass
class ASRResult:
    text: str
    language: str
    words: list[Word]
    avg_confidence: float

class ASRBackend(Protocol):
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> ASRResult: ...
    def is_available(self) -> bool: ...

class FasterWhisperBackend:
    def __init__(self, model_size=None, device=None):
        pass
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> ASRResult:
        return ASRResult("test", "en", [], 0.9)
    def is_available(self) -> bool:
        return True

class IndicASRBackend:
    def is_available(self) -> bool:
        try:
            import nemo.collections.asr
            return True
        except ImportError:
            return False
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> ASRResult:
        return ASRResult("test", "ta", [], 0.9)

def get_asr_backend(lang: str | None = None) -> ASRBackend:
    indic_langs = {"ta", "te", "ml", "hi"}
    if lang in indic_langs:
        backend = IndicASRBackend()
        if backend.is_available():
            return backend
    return FasterWhisperBackend()
