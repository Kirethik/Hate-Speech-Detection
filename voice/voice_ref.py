"""Choose the reference audio used to clone the speaker's voice.

Returns (spans, exact_text): one or more (start, end) pieces of the recording
that are joined into a single reference clip, plus their transcript (IndicF5 and
OmniVoice need the exact words spoken in the reference).
"""

from __future__ import annotations

from .types import Seg

Ref = tuple[list[tuple[float, float]], str]


def pick_ref(segs: list[Seg], flagged: set[int], lo: float = 4.0, hi: float = 12.0, gap: float = 0.8
             ) -> Ref | None:
    """1) best contiguous run of clean segments lasting lo..hi s (score = dur x conf);
    2) else join clean segments (most confident first, kept in time order) until >= lo;
    3) else whatever clean speech exists if >= 2 s. Never uses flagged segments."""
    best, best_score = None, -1.0
    n = len(segs)
    for i in range(n):
        if i in flagged:
            continue
        j = i
        while (j + 1 < n and j + 1 not in flagged and segs[j + 1].end - segs[i].start <= hi
               and segs[j + 1].start - segs[j].end < gap):
            j += 1
        dur = segs[j].end - segs[i].start
        if dur < lo or dur > hi:
            continue
        conf = sum(s.conf for s in segs[i:j + 1]) / (j - i + 1)
        if conf * dur > best_score:
            best_score = conf * dur
            best = ([(segs[i].start, segs[j].end)], " ".join(s.text for s in segs[i:j + 1]))
    if best:
        return best
    clean = [i for i in range(n) if i not in flagged and segs[i].end - segs[i].start <= hi]
    return _join(segs, sorted(clean, key=lambda i: -segs[i].conf), lo, hi, min_total=2.0)


def any_speech(segs: list[Seg], lo: float = 3.0, hi: float = 12.0) -> Ref | None:
    """Last resort: any speech (even flagged); only the timbre is copied, not the words."""
    order = sorted(range(len(segs)), key=lambda i: -(segs[i].end - segs[i].start))
    return _join(segs, [i for i in order if segs[i].end - segs[i].start <= hi], lo, hi, min_total=1.0)


def _join(segs, order, lo, hi, min_total) -> Ref | None:
    picked, total = [], 0.0
    for i in order:
        d = segs[i].end - segs[i].start
        if total + d > hi:
            continue
        picked.append(i)
        total += d
        if total >= lo:
            break
    if total < min_total:
        return None
    picked.sort()
    return [(segs[i].start, segs[i].end) for i in picked], " ".join(segs[i].text for i in picked)
