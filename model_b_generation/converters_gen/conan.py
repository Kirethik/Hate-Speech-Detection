"""
CONAN (Chung et al. 2019): multilingual Islamophobia hate speech /
counter-narrative pairs. English rows only. Task = respond, target = religion.

EXPECTED (verify against your download; see docs/SOURCES.md):
    data/raw/conan/CONAN.csv  (or CONAN.json as a list of records)
    a hate-speech column   : one of hateSpeech | HATE_SPEECH | hate_speech
    a counter column       : one of counterSpeech | COUNTER_NARRATIVE | counter_narrative
    optional language column: cn_id prefix "EN"/"FR"/"IT" or a LANGUAGE column
"""

import logging
from pathlib import Path

import pandas as pd

from ._common import empty_df, finalize, make_row

logger = logging.getLogger(__name__)

_HATE_COLS = ("hatespeech", "hate_speech")
_CN_COLS = ("counterspeech", "counter_narrative", "counternarrative")


def _pick(df, names):
    low = {c.lower(): c for c in df.columns}
    return next((low[n] for n in names if n in low), None)


def _load(raw_dir: str) -> pd.DataFrame | None:
    root = Path(raw_dir) / "conan"
    if (root / "CONAN.csv").exists():
        return pd.read_csv(root / "CONAN.csv")
    if (root / "CONAN.json").exists():
        import json
        data = json.loads((root / "CONAN.json").read_text(encoding="utf-8"))
        return pd.DataFrame(data.get("conan", data) if isinstance(data, dict) else data)
    logger.warning("CONAN not found under %s", root)
    return None


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    df = _load(raw_dir)
    if df is None or df.empty:
        return empty_df()
    hs, cn = _pick(df, _HATE_COLS), _pick(df, _CN_COLS)
    if not hs or not cn:
        logger.warning("CONAN: expected hate/counter columns, got %s", list(df.columns))
        return empty_df()
    lang_col = _pick(df, ("language", "lang"))
    id_col = _pick(df, ("cn_id",))
    if lang_col:
        df = df[df[lang_col].astype(str).str.lower().str.startswith("en")]
    elif id_col:
        df = df[df[id_col].astype(str).str.upper().str.startswith("EN")]
    return finalize(make_row("respond", h, c, "en", "conan", "religion")
                    for h, c in zip(df[hs], df[cn]))
