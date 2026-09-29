"""
Shared schema + helpers for the Model B (alternate speech) converters.

Every converter returns a DataFrame with PAIR_COLUMNS:
    task         "rewrite" | "respond"
    source_text  the hateful / toxic input
    target_text  the non-hateful rewrite, or the counter-narrative reply
    language     output language code (see prompts.LANGUAGE_NAMES)
    target       Model A target class ("unknown" when the source has none)
    source       dataset name
    split        "train" | "val" (hash of group_id, stable across rebuilds)
    group_id     hash of the original English source_text. Silver translations
                 inherit it, so a translation can never land in a different
                 split from the sentence it was translated from.
"""

import hashlib

import pandas as pd

PAIR_COLUMNS = ["task", "source_text", "target_text", "language", "target",
                "source", "split", "group_id"]

VAL_FRAC = 0.10

# CONAN-family TARGET values -> label_maps.TARGET_CLASSES
# (checked against Rhma/Multitarget-CONAN: MUSLIMS, MIGRANTS, WOMEN, LGBT+,
#  JEWS, POC, other, DISABLED). LGBT+ -> gender follows HATEXPLAIN_TARGET_MAP.
CONAN_TARGET_MAP = {
    "MUSLIMS": "religion", "JEWS": "religion",
    "MIGRANTS": "nationality_migrant",
    "WOMEN": "gender", "LGBT+": "gender",
    "POC": "caste_ethnicity",
    "DISABLED": "disability",
    "OTHER": "other",
}


def group_id(text: str) -> str:
    return hashlib.md5(str(text).strip().encode("utf-8")).hexdigest()[:16]


def hash_split(gid: str, val_frac: float = VAL_FRAC) -> str:
    return "val" if (int(gid[:8], 16) % 1000) < val_frac * 1000 else "train"


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIR_COLUMNS)


def make_row(task: str, source_text, target_text, language: str, source: str,
             target: str = "unknown", gid: str | None = None) -> dict | None:
    src = "" if source_text is None else str(source_text).strip()
    tgt = "" if target_text is None else str(target_text).strip()
    if not src or not tgt or src.lower() in ("nan", "none") or tgt.lower() in ("nan", "none"):
        return None
    gid = gid or group_id(src)
    return {"task": task, "source_text": src, "target_text": tgt, "language": language,
            "target": target or "unknown", "source": source, "split": hash_split(gid),
            "group_id": gid}


def finalize(rows) -> pd.DataFrame:
    rows = [r for r in rows if r]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS) if rows else empty_df()
