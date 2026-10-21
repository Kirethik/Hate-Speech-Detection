"""Cut replacements into the original audio without clicks or loudness jumps."""

from __future__ import annotations

import numpy as np


def snap(y: np.ndarray, sr: int, t: float, win: float = 0.08, hop_ms: float = 5) -> float:
    """Move a cut point to the quietest 5 ms frame within +/-80 ms."""
    a, b = max(0, int((t - win) * sr)), min(len(y), int((t + win) * sr))
    h = max(1, int(sr * hop_ms / 1000))
    seg = y[a:b]
    if len(seg) <= h:
        return t
    e = [float(np.sum(seg[i:i + h] ** 2)) for i in range(0, len(seg) - h, h)]
    return (a + int(np.argmin(e)) * h + h // 2) / sr


def fit(w: np.ndarray, slot: int, lo: float = 0.8, hi: float = 1.25) -> np.ndarray:
    """Time-stretch w to exactly `slot` samples if the ratio is mild; else leave it."""
    if slot <= 0 or len(w) == 0:
        return w
    r = len(w) / slot
    if lo <= r <= hi and abs(r - 1) > 0.02:
        import librosa

        w = librosa.effects.time_stretch(w.astype(np.float32), rate=r)
        w = w[:slot] if len(w) >= slot else np.pad(w, (0, slot - len(w)))
    return w.astype(np.float32)


def match_loud(w: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
    """Match integrated loudness (LUFS) of w to ref; fall back to RMS matching."""
    if len(w) == 0 or len(ref) == 0:
        return w
    try:
        import pyloudnorm as pyln

        if len(ref) >= 0.5 * sr and len(w) >= 0.5 * sr:
            m = pyln.Meter(sr)
            lw, lr = m.integrated_loudness(w), m.integrated_loudness(ref)
            if np.isfinite(lw) and np.isfinite(lr):
                return np.clip(pyln.normalize.loudness(w, lw, lr), -1, 1).astype(np.float32)
    except ImportError:  # pragma: no cover
        pass
    rw, rr = np.sqrt(np.mean(w ** 2)) + 1e-9, np.sqrt(np.mean(ref ** 2)) + 1e-9
    return np.clip(w * (rr / rw), -1, 1).astype(np.float32)


def trim_silence(w: np.ndarray, sr: int, top_db: float = 35, pad_ms: float = 40) -> np.ndarray:
    if len(w) == 0:
        return w
    import librosa

    _, (a, b) = librosa.effects.trim(w, top_db=top_db)
    p = int(sr * pad_ms / 1000)
    return w[max(0, a - p): min(len(w), b + p)]


def xjoin(parts: list[np.ndarray], sr: int, ms: float = 25) -> np.ndarray:
    """Concatenate with equal-power crossfades."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    n, out = int(sr * ms / 1000), parts[0].astype(np.float32)
    for p in parts[1:]:
        k = min(n, len(out), len(p))
        if k == 0:
            out = np.concatenate([out, p])
            continue
        t = np.linspace(0, np.pi / 2, k, dtype=np.float32)
        out = np.concatenate([out[:-k], out[-k:] * np.cos(t) + p[:k] * np.sin(t), p[k:]])
    return out.astype(np.float32)


def splice(y: np.ndarray, sr: int, edits: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """edits: dicts with start, end (seconds) and wav (np array at sr), sorted and
    non-overlapping. Returns (new audio, edits with new_start/new_end, wav removed)."""
    parts, cur, shift, log = [], 0, 0.0, []
    for e in sorted(edits, key=lambda d: d["start"]):
        s, t = snap(y, sr, e["start"]), snap(y, sr, e["end"])
        s = max(s, cur / sr)
        if t <= s:
            t = min(len(y) / sr, s + 0.05)
        a, b = int(s * sr), int(t * sr)
        w = match_loud(fit(e["wav"], b - a), y[a:b], sr)
        parts += [y[cur:a], w]
        meta = {k: v for k, v in e.items() if k != "wav"}
        meta.update(start=s, end=t, new_start=s + shift, new_end=s + shift + len(w) / sr)
        log.append(meta)
        shift += len(w) / sr - (t - s)
        cur = b
    parts.append(y[cur:])
    return np.clip(xjoin(parts, sr), -1, 1), log


def soft_beep(dur: float, sr: int, freq: float = 660.0, level: float = 0.05) -> np.ndarray:
    """Quiet tone used when a sentence is muted instead of replaced."""
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    env = np.minimum(1, np.minimum(t, dur - t) / 0.05)
    return (level * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
