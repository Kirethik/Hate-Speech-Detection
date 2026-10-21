"""Download every model the Space uses into the HF cache at image build time."""

import os
import sys

from huggingface_hub import snapshot_download

TOKEN = os.getenv("HF_TOKEN") or None
WEIGHTS = ["*.json", "*.txt", "*.model", "model.safetensors"]  # skip optimizer/trainer state

JOBS = [
    ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", None),  # faster-whisper "large-v3-turbo"
    ("textdetox/xlmr-large-toxicity-classifier-v2", WEIGHTS),
    ("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL", WEIGHTS),
    ("ResembleAI/chatterbox", ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                               "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]),
    ("charactr/vocos-mel-24khz", None),
    ("ai4bharat/IndicF5", None),  # gated
]

failed = []
for repo, patterns in JOBS:
    try:
        snapshot_download(repo, allow_patterns=patterns, token=TOKEN)
        print(f"ok    {repo}", flush=True)
    except Exception as e:  # don't fail the build: the app downloads lazily at runtime too
        failed.append(repo)
        print(f"FAIL  {repo}: {type(e).__name__}: {e}", flush=True)

if failed:
    print("Not pre-downloaded (will retry at runtime):", ", ".join(failed), file=sys.stderr)
    if "ai4bharat/IndicF5" in failed and not TOKEN:
        print("IndicF5 is gated: add HF_TOKEN as a Space secret and rebuild.", file=sys.stderr)
