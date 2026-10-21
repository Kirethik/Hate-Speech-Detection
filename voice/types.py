"""Plain data types shared by every stage of the voice pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Seg:
    """One replaceable unit of speech (roughly a sentence)."""
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    conf: float = 1.0

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text,
                "conf": round(self.conf, 3)}


@dataclass
class Verdict:
    is_hate: bool
    severity: str           # none | offensive | hate
    target: str             # label_maps.TARGET_CLASSES value
    spans: list[str]
    rewrite: str
    counter: str
    tts_text: str
    reason: str = ""
    source: str = "llm"     # llm | classifier

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edit:
    seg_index: int
    start: float
    end: float
    original_text: str
    replacement_text: str
    severity: str
    target: str
    spans: list[str]
    p_classifier: float
    method: str              # tts | muted
    reason: str = ""
    new_start: Optional[float] = None
    new_end: Optional[float] = None
    tts_engine: Optional[str] = None
    safety_p: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "new_start", "new_end", "p_classifier", "safety_p"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d
