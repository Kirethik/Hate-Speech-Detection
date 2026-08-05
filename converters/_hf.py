"""
Shared helpers for the implicit-hate converters.

Unlike the five original sources, these four ship from the HuggingFace Hub
rather than `raw_data/`, so they download-and-cache instead of reading a local
path. Everything lands in the normal HF cache (~/.cache/huggingface), so a
second run is offline.

All four are English-only. That is a real limitation, not an oversight:
implicit-hate annotation essentially does not exist for Tamil/Kannada/Telugu/
Malayalam, so these sources widen the model's English capability without
touching the Indic side. Say so in the write-up rather than implying the
improvement is multilingual.
"""

import hashlib

import pandas as pd

# Deterministic 90/10 train/val split for sources that ship no official split.
# Hash-based rather than random so re-running the converter is reproducible
# without threading a seed through every call site. prepare_splits.py re-pools
# and re-splits everything downstream anyway; this only decides which rows are
# *eligible* to become val/test.
VAL_FRACTION = 0.10


def hash_split(text: str, val_fraction: float = VAL_FRACTION) -> str:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return "val" if (int(digest[:8], 16) % 1000) < val_fraction * 1000 else "train"


def cap(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    """
    Subsample a source to at most `max_rows`, stratified by hate_label so the
    class balance survives the cut.

    This matters: DynaHate alone is 40k English rows against 16k English in the
    existing corpus, so importing everything uncapped would flip a multilingual
    model into a mostly-English one and cost Indic performance to buy English
    gains. Capping keeps the additions targeted.
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    # Iterate the groups explicitly rather than using groupby().apply(): as of
    # pandas 3 the grouping column is excluded from the frame passed to apply,
    # so the reassembled result silently comes back with hate_label all-NaN.
    parts = [
        g.sample(n=max(1, round(len(g) * frac)), random_state=seed)
        for _, g in df.groupby("hate_label", sort=False)
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def strip_byte_prefix(text: str) -> str:
    """ToxiGen's CSV stores Python byte-string reprs: b'asians are ...'."""
    s = str(text).strip()
    if len(s) >= 3 and s[:2] in ("b'", 'b"') and s[-1] in ("'", '"'):
        s = s[2:-1]
    return s.replace("\\'", "'").replace('\\"', '"').replace("\\n", " ").strip()
