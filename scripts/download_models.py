"""
Pre-download every model so the first upload isn't slow, and check gated access.

    python scripts/download_models.py               # ASR + classifiers + IndicConformer (main .venv)
    python scripts/download_models.py --tts indicf5     (run with .venv-tts-indicf5)
    python scripts/download_models.py --tts chatterbox  (run with .venv-tts-chatterbox)
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TOKEN = os.getenv("HF_TOKEN") or None


def snap(repo, gated=False):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        p = snapshot_download(repo, token=TOKEN)
        print(f"  ok   {repo}")
        return p
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print(f"  FAIL {repo}: gated model. Open https://huggingface.co/{repo}, click 'Agree',"
              f" and put a read token in .env as HF_TOKEN. ({type(e).__name__})")
    except Exception as e:
        print(f"  FAIL {repo}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", choices=["indicf5", "chatterbox", "omnivoice"])
    a = ap.parse_args()
    if a.tts == "indicf5":
        snap("ai4bharat/IndicF5", gated=True)
        return
    if a.tts == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        print("  ok   Chatterbox Multilingual")
        return
    if a.tts == "omnivoice":
        snap("k2-fsa/OmniVoice")
        return

    from voice.settings import get_settings

    s = get_settings()
    print("Whisper:", s.whisper_model)
    from faster_whisper import WhisperModel

    WhisperModel(s.whisper_model, device="cpu", compute_type="int8")
    print("  ok   faster-whisper", s.whisper_model)
    print("Classifiers:")
    for m in s.clf_models:
        snap(m)
    print("Silero VAD:")
    from silero_vad import load_silero_vad

    load_silero_vad()
    print("  ok   silero-vad")
    if s.enable_indic_asr:
        print("IndicConformer (gated):")
        snap(s.indic_asr_model, gated=True)


if __name__ == "__main__":
    main()
