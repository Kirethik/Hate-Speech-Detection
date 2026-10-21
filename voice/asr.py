"""
Speech-to-text.

- WhisperAsr: faster-whisper (CTranslate2). On Intel machines it runs on the CPU
  with int8 weights; gives language ID and word timestamps.
- IndicAsr: AI4Bharat IndicConformer 600M (ONNX, runs on CPU). Better for
  ta/te/ml/kn/bn/... Gives text only, so we transcribe VAD chunks one by one and
  use the chunk boundaries as segment times (we replace whole sentences anyway).
"""

from __future__ import annotations

import logging

import numpy as np

from .sentences import split_sentences
from .types import Seg, Word
from .vad import merge_regions, speech_regions

log = logging.getLogger("civitas.asr")

INDIC_ASR_LANGS = {"ta", "te", "ml", "kn", "bn", "mr", "gu", "pa", "or", "as"}


class WhisperAsr:
    def __init__(self, model: str, device: str = "cpu", compute: str = "int8", threads: int = 0, beam: int = 5):
        from faster_whisper import WhisperModel

        self.m = WhisperModel(model, device=device, compute_type=compute, cpu_threads=threads)
        self.beam = beam
        self.name = f"faster-whisper/{model}"

    def detect_language(self, y16: np.ndarray) -> tuple[str, float]:
        clip = y16[: 30 * 16000]
        if hasattr(self.m, "detect_language"):
            lang, prob, _ = self.m.detect_language(clip)
            return lang, float(prob)
        _, info = self.m.transcribe(clip, beam_size=1, without_timestamps=True)
        return info.language, float(info.language_probability)

    def transcribe(self, y16: np.ndarray, lang: str | None = None) -> tuple[str, list[Seg]]:
        segs, info = self.m.transcribe(y16, language=lang, task="transcribe", beam_size=self.beam,
                                       word_timestamps=True, vad_filter=True,
                                       condition_on_previous_text=False)
        out = []
        for g in segs:
            words = [Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                     for w in (g.words or []) if w.word.strip()]
            conf = sum(w.prob for w in words) / len(words) if words else float(np.exp(g.avg_logprob))
            if g.no_speech_prob > 0.8 and conf < 0.4:  # hallucination guard
                continue
            out.append(Seg(float(g.start), float(g.end), g.text.strip(), words, conf))
        return info.language, out


class IndicAsr:
    def __init__(self, model_id: str, decoding: str = "ctc", token: str | None = None):
        from transformers import AutoModel

        self.m = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=token or None)
        self.decoding = decoding
        self.name = model_id.split("/")[-1]

    def transcribe(self, y16: np.ndarray, lang: str) -> list[Seg]:
        import torch

        regions = merge_regions(speech_regions(y16), max_len=12.0, max_gap=0.35)
        out = []
        for a, b in regions:
            chunk = y16[int(a * 16000): int(b * 16000)]
            if len(chunk) < 0.3 * 16000:
                continue
            text = self.m(torch.from_numpy(chunk).unsqueeze(0), lang, self.decoding)
            text = (text[0] if isinstance(text, (list, tuple)) else str(text)).strip()
            if text:
                out.append(Seg(a, b, text, [], 0.9))
        return out


def transcribe(reg, y16: np.ndarray, lang_hint: str | None = None) -> tuple[str, list[Seg], str]:
    """Route to the best ASR for the language. Returns (lang, sentence segs, asr name)."""
    whisper = reg.whisper()
    lang = lang_hint
    if not lang:
        lang, p = whisper.detect_language(y16)
        log.info("detected language %s (p=%.2f)", lang, p)
    if lang in INDIC_ASR_LANGS:
        indic = reg.indic_asr()
        if indic is not None:
            try:
                return lang, indic.transcribe(y16, lang), indic.name
            except Exception as e:  # pragma: no cover - depends on gated model
                log.warning("IndicConformer failed (%s); falling back to Whisper", e)
    lang2, segs = whisper.transcribe(y16, lang)
    return lang or lang2, split_sentences(segs), whisper.name
