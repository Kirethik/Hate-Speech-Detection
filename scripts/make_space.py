"""
Assemble (and optionally upload) the Hugging Face Docker Space for Civitas AI.

    python scripts/make_space.py                         # build build/hf_space/ only
    python scripts/make_space.py --upload USER/SPACE     # build + upload (needs `hf auth login`, write token)
    python scripts/make_space.py --upload USER/SPACE --set-hf-secret   # also store HF_TOKEN from .env

The folder holds only what the running app needs: the API/voice code, the TTS service,
the built web UI (frontend/dist -> web/), deploy/hf_space/* and constraints-*.txt frozen
from the local virtualenvs that are known to work (.venv, .venv-tts-chatterbox, .venv-tts-indicf5).
Run `npm run build` in frontend/ first.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "hf_space"

CODE_DIRS = ["server", "voice", "pipeline", "src", "tts_service"]
CODE_FILES = ["config.py", "label_maps.py", "devices.py", "dehumanization.py", "requirements-voice.txt"]
VENVS = {"constraints-api.txt": ".venv", "constraints-cb.txt": ".venv-tts-chatterbox",
         "constraints-f5.txt": ".venv-tts-indicf5"}
# torch comes from the CPU wheel index in the Dockerfile; the rest are Windows-only
SKIP = {"torch", "torchaudio", "torchvision", "pywin32", "pywin32-ctypes", "pywinpty", "wmi"}


def freeze(venv: str) -> str:
    py = ROOT / venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    out = subprocess.run([str(py), "-m", "pip", "freeze", "--exclude-editable"], check=True,
                         capture_output=True, text=True).stdout
    keep = []
    for line in out.splitlines():
        name = line.split("==")[0].strip().lower()
        if "==" not in line or "@" in line or name in SKIP:
            continue
        keep.append(line.split("+")[0])  # drop local version tags like +cpu
    return "\n".join(keep) + "\n"


def build() -> Path:
    dist = ROOT / "frontend" / "dist" / "index.html"
    if not dist.exists():
        sys.exit("frontend/dist is missing: run `npm run build` in frontend/ first")
    if OUT.exists():
        shutil.rmtree(OUT)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log")
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=ignore)
    for f in CODE_FILES:
        shutil.copy2(ROOT / f, OUT / f)
    shutil.copytree(ROOT / "frontend" / "dist", OUT / "web")
    for f in (ROOT / "deploy" / "hf_space").iterdir():
        shutil.copy2(f, OUT / f.name)
    for name, venv in VENVS.items():
        (OUT / name).write_text(freeze(venv), encoding="utf-8")
    # Spaces build on Linux: make sure the entrypoint has LF line endings
    sh = OUT / "start.sh"
    sh.write_bytes(sh.read_bytes().replace(b"\r\n", b"\n"))
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"built {OUT.relative_to(ROOT)} ({n} files)")
    return OUT


def upload(folder: Path, space: str, set_secret: bool, private: bool):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    if set_secret:
        try:
            from dotenv import dotenv_values

            tok = dotenv_values(ROOT / ".env").get("HF_TOKEN") or os.getenv("HF_TOKEN")
        except ImportError:
            tok = os.getenv("HF_TOKEN")
        if not tok:
            sys.exit("--set-hf-secret: no HF_TOKEN in .env or the environment")
        api.add_space_secret(space, "HF_TOKEN", tok)
        print("stored HF_TOKEN as a Space secret")
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(folder),
                      commit_message="Deploy Civitas AI", delete_patterns=["*"])
    print(f"uploaded -> https://huggingface.co/spaces/{space}")
    print(f"app      -> https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="USER/SPACE")
    ap.add_argument("--set-hf-secret", action="store_true", help="copy HF_TOKEN from .env into the Space secrets")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    folder = build()
    if a.upload:
        upload(folder, a.upload, a.set_hf_secret, a.private)


if __name__ == "__main__":
    main()
