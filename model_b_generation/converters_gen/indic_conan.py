"""
IndicCONAN: Hindi (and English) hate speech / counter-narrative pairs.
Task = respond.

EXPECTED (verify against your download; see docs/SOURCES.md):
    data/raw/indic_conan/*.csv
    a column whose name contains "hate"     -> source_text
    a column whose name contains "counter"  -> target_text
    optional language column ("hi"/"en"/"Hindi"/...); default hi
    optional target column mapped through CONAN_TARGET_MAP (else "unknown")
"""

import logging
from pathlib import Path

import pandas as pd

from ._common import CONAN_TARGET_MAP, empty_df, finalize, make_row

logger = logging.getLogger(__name__)


def _lang(v) -> str:
    s = str(v).strip().lower()
    return "en" if s.startswith("en") else "hi"


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    files = sorted((Path(raw_dir) / "indic_conan").glob("*.csv"))
    if not files:
        logger.warning("IndicCONAN not found under %s/indic_conan", raw_dir)
        return empty_df()
    rows = []
    for f in files:
        df = pd.read_csv(f)
        low = {c.lower(): c for c in df.columns}
        hs = next((low[c] for c in low if "hate" in c), None)
        cn = next((low[c] for c in low if "counter" in c), None)
        if not hs or not cn:
            logger.warning("IndicCONAN %s: no hate/counter columns in %s", f.name, list(df.columns))
            continue
        lang_col = next((low[c] for c in low if c in ("language", "lang")), None)
        tgt_col = next((low[c] for c in low if c == "target"), None)
        for i in range(len(df)):
            lang = _lang(df[lang_col].iloc[i]) if lang_col else "hi"
            tgt = (CONAN_TARGET_MAP.get(str(df[tgt_col].iloc[i]).strip().upper(), "unknown")
                   if tgt_col else "unknown")
            rows.append(make_row("respond", df[hs].iloc[i], df[cn].iloc[i], lang,
                                 "indic_conan", tgt))
    return finalize(rows)
