"""
Voice-cloning TTS engines. Each runs in its own Python environment because their
pinned dependencies conflict (see scripts/setup_windows.ps1):

  indicf5     ai4bharat/IndicF5        11 Indian languages, needs ref audio + exact ref text   (MIT, gated)
  chatterbox  Chatterbox Multilingual   23 languages incl. en/hi, needs ref audio               (MIT)
  omnivoice   k2-fsa/OmniVoice          600+ languages, ref audio + ref text                    (CC-BY-NC weights)
  dummy       tone generator            for wiring tests without any model

Device: CIVITAS_TTS_DEVICE=auto|xpu|cuda|cpu. "auto" tries Intel XPU, then CUDA,
then CPU. If a model fails on the GPU it is reloaded on CPU automatically.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("civitas.tts")


def pick_device() -> str:
    want = os.getenv("CIVITAS_TTS_DEVICE", "auto").lower()
    if want != "auto":
        return want
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class Engine:
    name = "base"
    sr = 24000
    needs_ref_text = False

    def synth(self, text: str, lang: str, ref_path: str, ref_text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class DummyEngine(Engine):
    name = "dummy"

    def synth(self, text, lang, ref_path, ref_text):
        dur = max(0.6, 0.065 * len(text))
        t = np.arange(int(dur * self.sr)) / self.sr
        env = np.minimum(1, np.minimum(t, dur - t) / 0.05)
        y = 0.2 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * env
        return y.astype(np.float32), self.sr


def _patch_torchaudio_load():
    """torchaudio >= 2.9 decodes only through the separate torchcodec package. f5_tts calls
    torchaudio.load() on the reference wav, so without torchcodec read it with soundfile."""
    import importlib.util

    import torchaudio

    if importlib.util.find_spec("torchcodec") is not None or getattr(torchaudio.load, "_civitas", False):
        return
    import soundfile as sf
    import torch

    def load(path, *args, **kwargs):
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
        return torch.from_numpy(y.T.copy()), sr

    load._civitas = True
    torchaudio.load = load


class IndicF5Engine(Engine):
    name = "indicf5"
    needs_ref_text = True

    def __init__(self, device: str):
        # IndicF5's remote code needs transformers 4.x (5.x builds models on the meta device,
        # which breaks its vocoder loading) - see tts_service/requirements-indicf5.txt
        from transformers import AutoModel

        _patch_torchaudio_load()
        tok = os.getenv("HF_TOKEN") or None
        self.m = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, token=tok)
        self.device = "cpu"
        if device != "cpu":
            try:
                self.m = self.m.to(device)
                self.device = device
            except Exception as e:
                log.warning("IndicF5 on %s failed (%s); using CPU", device, e)

    def synth(self, text, lang, ref_path, ref_text):
        a = np.asarray(self.m(text, ref_audio_path=ref_path, ref_text=ref_text))
        a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
        return a.reshape(-1), 24000


class ChatterboxEngine(Engine):
    name = "chatterbox"

    def __init__(self, device: str):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        try:
            self.m = ChatterboxMultilingualTTS.from_pretrained(device=device)
            self.device = device
        except Exception as e:
            if device == "cpu":
                raise
            log.warning("Chatterbox on %s failed (%s); using CPU", device, e)
            self.m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            self.device = "cpu"
        self.sr = self.m.sr

    def synth(self, text, lang, ref_path, ref_text):
        kw = {}
        if os.getenv("CIVITAS_CB_CFG"):
            kw["cfg_weight"] = float(os.environ["CIVITAS_CB_CFG"])
        w = self.m.generate(text, language_id=lang, audio_prompt_path=ref_path, **kw)
        return w.squeeze(0).detach().float().cpu().numpy(), self.sr


class OmniVoiceEngine(Engine):
    name = "omnivoice"
    needs_ref_text = True

    def __init__(self, device: str):
        import torch
        from omnivoice import OmniVoice

        dm = {"xpu": "xpu:0", "cuda": "cuda:0"}.get(device, "cpu")
        dt = torch.float32 if dm == "cpu" else torch.float16
        self.m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dm, dtype=dt)
        self.device = dm

    def synth(self, text, lang, ref_path, ref_text):
        audio = self.m.generate(text=text, ref_audio=ref_path, ref_text=ref_text)
        return np.asarray(audio[0], dtype=np.float32).reshape(-1), 24000


ENGINES = {"dummy": DummyEngine, "indicf5": IndicF5Engine, "chatterbox": ChatterboxEngine,
           "omnivoice": OmniVoiceEngine}


def load(name: str) -> Engine:
    if name not in ENGINES:
        raise ValueError(f"unknown engine '{name}' (choose from {sorted(ENGINES)})")
    if name == "dummy":
        return DummyEngine()
    dev = pick_device()
    log.info("loading %s on %s", name, dev)
    return ENGINES[name](dev)
