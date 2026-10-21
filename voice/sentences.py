"""Turn ASR output into replaceable sentence-sized units (about 1-10 s each)."""

from __future__ import annotations

from .types import Seg, Word

END_PUNCT = (".", "?", "!", "।", "॥", "؟", "۔", "…")  # . ? ! । ॥ ؟ ۔ …


def _flush(words: list[Word]) -> Seg:
    text = " ".join(w.text for w in words).strip()
    conf = sum(w.prob for w in words) / len(words)
    return Seg(words[0].start, words[-1].end, text, list(words), conf)


def split_sentences(segs: list[Seg], pause: float = 0.6, max_len: float = 10.0, min_words: int = 2
                    ) -> list[Seg]:
    """Split on sentence punctuation or long pauses; cap length. Segments without
    word timings are kept as-is (they are already VAD-sized chunks)."""
    out: list[Seg] = []
    for s in segs:
        if not s.words:
            if s.text.strip():
                out.append(s)
            continue
        cur: list[Word] = []
        for i, w in enumerate(s.words):
            if cur and (w.start - cur[-1].end >= pause or w.end - cur[0].start > max_len):
                out.append(_flush(cur))
                cur = []
            cur.append(w)
            nxt_gap = s.words[i + 1].start - w.end if i + 1 < len(s.words) else 0
            if w.text.rstrip().endswith(END_PUNCT) and (len(cur) >= min_words or nxt_gap > 0.25):
                out.append(_flush(cur))
                cur = []
        if cur:
            out.append(_flush(cur))
    return [s for s in out if s.text]
