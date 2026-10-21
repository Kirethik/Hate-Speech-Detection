"""Silero VAD: speech regions for chunking (Indic ASR route) and live mode."""

from __future__ import annotations

import numpy as np

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def speech_regions(y16: np.ndarray, min_silence_ms: int = 300, min_speech_ms: int = 250,
                   threshold: float = 0.5) -> list[tuple[float, float]]:
    """(start_s, end_s) of speech in a 16 kHz mono signal."""
    import torch
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(torch.from_numpy(y16.astype(np.float32)), _model(), sampling_rate=16000,
                               threshold=threshold, min_silence_duration_ms=min_silence_ms,
                               min_speech_duration_ms=min_speech_ms, speech_pad_ms=80, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def merge_regions(regions: list[tuple[float, float]], max_len: float = 12.0, max_gap: float = 0.35
                  ) -> list[tuple[float, float]]:
    """Join regions separated by short gaps, never exceeding max_len seconds."""
    out: list[list[float]] = []
    for a, b in regions:
        if out and a - out[-1][1] <= max_gap and b - out[-1][0] <= max_len:
            out[-1][1] = b
        else:
            out.append([a, b])
    # hard-split anything still too long
    res = []
    for a, b in out:
        while b - a > max_len:
            res.append((a, a + max_len))
            a += max_len
        res.append((a, b))
    return res
