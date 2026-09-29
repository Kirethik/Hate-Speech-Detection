"""
Parallel detoxification data for the Rewrite task (toxic -> neutral, same meaning).

  ParaDetox        https://huggingface.co/datasets/s-nlp/paradetox           (en)
  TextDetox 2024   https://huggingface.co/datasets/textdetox/multilingual_paradetox (en, hi)

EXPECTED (verify; see docs/SOURCES.md): toxic/neutral column pairs named one of
  en_toxic_comment / en_neutral_comment   (ParaDetox)
  toxic_sentence   / neutral_sentence     (TextDetox; one split per language code)
Local fallback: data/raw/paradetox/<anything>.csv with those columns and an
optional `language` column (default en).
"""

import logging
from pathlib import Path

import pandas as pd

from ._common import empty_df, finalize, make_row

logger = logging.getLogger(__name__)

_TOXIC = ("en_toxic_comment", "toxic_sentence", "toxic")
_NEUTRAL = ("en_neutral_comment", "neutral_sentence", "neutral")
_TEXTDETOX_LANGS = {"en": "en", "hi": "hi"}


def _pairs(df: pd.DataFrame, language: str, source: str) -> list:
    low = {c.lower(): c for c in df.columns}
    tox = next((low[c] for c in _TOXIC if c in low), None)
    neu = next((low[c] for c in _NEUTRAL if c in low), None)
    if not tox or not neu:
        logger.warning("%s: no toxic/neutral columns in %s", source, list(df.columns))
        return []
    lang_col = low.get("language")
    return [make_row("rewrite", t, n, str(df[lang_col].iloc[i]) if lang_col else language, source)
            for i, (t, n) in enumerate(zip(df[tox], df[neu]))]


def _hf(name: str, **kw):
    from datasets import load_dataset
    return load_dataset(name, **kw)


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    rows = []
    local = sorted((Path(raw_dir) / "paradetox").glob("*.csv"))
    for f in local:
        rows += _pairs(pd.read_csv(f), "en", "paradetox_local")
    if not local:
        try:
            rows += _pairs(_hf("s-nlp/paradetox", split="train").to_pandas(), "en", "paradetox")
        except Exception as e:
            logger.warning("s-nlp/paradetox unavailable: %s", e)
        try:
            ds = _hf("textdetox/multilingual_paradetox")
            for split, lang in _TEXTDETOX_LANGS.items():
                if split in ds:
                    rows += _pairs(ds[split].to_pandas(), lang, "textdetox2024")
        except Exception as e:
            logger.warning("textdetox/multilingual_paradetox unavailable: %s", e)
    return finalize(rows) if rows else empty_df()
