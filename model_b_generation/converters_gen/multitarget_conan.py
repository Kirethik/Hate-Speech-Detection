"""
Multitarget-CONAN (Fanton et al. 2021): English hate speech / counter-narrative
pairs over 8 targets. Task = respond.

Loaded from the HuggingFace mirror `Rhma/Multitarget-CONAN`, whose columns
were checked: INDEX, HATE_SPEECH, COUNTER_NARRATIVE, TARGET, VERSION.
Fallback: a local CSV with the same columns at
data/raw/multitarget_conan/Multitarget-CONAN.csv
"""

import logging
from pathlib import Path

import pandas as pd

from ._common import CONAN_TARGET_MAP, empty_df, finalize, make_row

logger = logging.getLogger(__name__)


def _load(raw_dir: str) -> pd.DataFrame | None:
    local = Path(raw_dir) / "multitarget_conan" / "Multitarget-CONAN.csv"
    if local.exists():
        return pd.read_csv(local)
    try:
        from datasets import load_dataset
        return load_dataset("Rhma/Multitarget-CONAN", split="train").to_pandas()
    except Exception as e:
        logger.warning("Multitarget-CONAN unavailable (%s). Put the CSV at %s", e, local)
        return None


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    df = _load(raw_dir)
    if df is None or df.empty:
        return empty_df()
    df = df.rename(columns={c: c.upper() for c in df.columns})
    if not {"HATE_SPEECH", "COUNTER_NARRATIVE"} <= set(df.columns):
        logger.warning("Multitarget-CONAN: unexpected columns %s", list(df.columns))
        return empty_df()
    targets = df["TARGET"] if "TARGET" in df.columns else pd.Series(["other"] * len(df))
    return finalize(
        make_row("respond", h, c, "en", "multitarget_conan",
                 CONAN_TARGET_MAP.get(str(t).strip().upper(), "other"))
        for h, c, t in zip(df["HATE_SPEECH"], df["COUNTER_NARRATIVE"], targets)
    )
