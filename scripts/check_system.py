"""Print what Civitas Voice can see: device, ffmpeg, LLM server, TTS services."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.settings import get_settings  # noqa: E402

s = get_settings()
ok = lambda b: "OK  " if b else "MISSING"  # noqa: E731

try:
    import torch

    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"[{ok(True)}] torch {torch.__version__}  | Intel XPU: {xpu}"
          + (f" ({torch.xpu.get_device_name(0)})" if xpu else ""))
except Exception as e:
    print(f"[MISSING] torch: {e}")

ff = shutil.which("ffmpeg")
if not ff:
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = None
print(f"[{ok(ff)}] ffmpeg: {ff}")

try:
    from voice.llm import Llm

    up = Llm(s.llm_base_url, s.llm_model).ping()
    print(f"[{ok(up)}] LLM server {s.llm_base_url} (model {s.llm_model})"
          + ("" if up else "  -> start Ollama and run: ollama pull " + s.llm_model))
except Exception as e:
    print(f"[MISSING] LLM client: {e}")

from voice.tts_client import TtsClient  # noqa: E402

for eng, alive in TtsClient(s.tts_route_map).health().items():
    print(f"[{ok(alive)}] TTS {eng} at {s.tts_route_map[eng]}")
print(f"[{ok(bool(s.hf_token))}] HF_TOKEN set (needed for IndicF5 / IndicConformer)")
