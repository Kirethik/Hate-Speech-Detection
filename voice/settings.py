"""
Runtime settings for the Civitas Voice pipeline (pretrained models, no training).

Every field can be overridden with an environment variable named CIVITAS_<FIELD>
(upper case), or from a `.env` file in the repo root. Defaults target an
Intel laptop (Core Ultra CPU + Arc iGPU): Whisper on CPU int8, classifiers on
XPU if PyTorch-XPU is installed (else CPU), LLM through Ollama (Vulkan on Arc).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _split(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Settings:
    # storage / limits
    data_dir: str = str(REPO_ROOT / "data" / "voice")
    max_audio_s: float = 300.0
    max_upload_mb: float = 50.0

    # compute: "auto" -> cuda > xpu > cpu (see devices.py)
    device: str = "auto"
    detector_device: str = "auto"

    # ASR (faster-whisper runs on CPU on Intel machines; int8 is fast on Core Ultra)
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    whisper_threads: int = 0  # 0 = let CTranslate2 decide
    whisper_beam: int = 5
    enable_indic_asr: bool = True
    indic_asr_model: str = "ai4bharat/indic-conformer-600m-multilingual"
    indic_decoding: str = "ctc"  # ctc | rnnt

    # detection
    clf_models: list[str] = field(default_factory=lambda: [
        "textdetox/xlmr-large-toxicity-classifier-v2",
        "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL",
    ])
    # languages the classifiers cover reasonably; others always go to the LLM
    clf_langs: list[str] = field(default_factory=lambda: ["en", "hi", "ur", "ta", "ml", "kn", "bn", "mr"])
    llm_gate: float = 0.30       # classifier prob above which the LLM judges the sentence
    always_llm: bool = False     # send every sentence to the LLM (slower, more recall)
    safety_threshold: float = 0.5  # replacement text must score below this

    # LLM (any OpenAI-compatible server: Ollama, llama.cpp server, LM Studio, vLLM, hosted)
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gemma4:e4b"
    llm_api_key: str = ""
    llm_timeout_s: float = 180.0
    llm_temperature: float = 0.2
    # "none" turns off hidden thinking in reasoning models (gemma4 thinks by default: ~550 extra
    # tokens, 40-50 s per sentence on CPU instead of ~10 s). Empty = don't send the field.
    llm_reasoning_effort: str = "none"

    # TTS services: engine=url pairs. Each engine runs in its own venv (tts_service/).
    tts_routes: str = "indicf5=http://127.0.0.1:9001,chatterbox=http://127.0.0.1:9002"
    tts_timeout_s: float = 300.0

    # voice reference
    ref_min_s: float = 4.0
    ref_max_s: float = 12.0

    # misc
    default_mode: str = "rewrite"  # rewrite | counter
    hf_token: str = ""

    @property
    def tts_route_map(self) -> dict[str, str]:
        out = {}
        for pair in _split(self.tts_routes):
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k.strip()] = v.strip().rstrip("/")
        return out

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        for f in fields(cls):
            raw = os.getenv(f"CIVITAS_{f.name.upper()}")
            if raw is None:
                continue
            cur = getattr(s, f.name)
            if isinstance(cur, bool):
                val = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            elif isinstance(cur, list):
                val = _split(raw)
            else:
                val = raw
            setattr(s, f.name, val)
        if not s.hf_token:
            s.hf_token = os.getenv("HF_TOKEN", "")
        return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
