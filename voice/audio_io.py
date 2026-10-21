"""Decode any upload (webm/opus, m4a, mp3, wav...) with ffmpeg; save wav/mp3."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # pip install imageio-ffmpeg bundles a static binary (works on Windows)
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from e


def decode(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return mono float32 audio. sr=None keeps the source rate (capped at 48 kHz)."""
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    else:
        sr = probe_rate(path)
        cmd += ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy(), sr


def probe_rate(path: str | Path) -> int:
    try:
        r = sf.info(str(path)).samplerate
    except Exception:
        r = 48000  # compressed formats: decode at 48 kHz
    return int(min(max(r, 16000), 48000))


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32)
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=sr_from, target_sr=sr_to)


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1, 1), sr, subtype="PCM_16")
    return path


def save_mp3(path: str | Path, y: np.ndarray, sr: int, bitrate: str = "160k") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    save_wav(tmp, y, sr)
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(tmp),
                    "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)], check=True)
    tmp.unlink(missing_ok=True)
    return path


def slice_s(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return y[max(0, int(start * sr)): max(0, int(end * sr))]
