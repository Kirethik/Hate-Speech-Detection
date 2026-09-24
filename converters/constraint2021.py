"""
CONSTRAINT 2021 Hindi Hostility Detection converter.

Expected directory layout (VERIFY against your download):
    data/raw/constraint2021/
        train.csv     text column: "Post" or "text"
        val.csv       label column: "Labels Set" or "label"
        (test file, if present, is ignored: shared-task test labels were withheld)

Each post carries a comma-separated label SET drawn from:
    non-hostile | fake | hate | offensive | defamation
e.g. "hate,offensive" or "fake,defamation". Matching is case-insensitive.

Resolution (see label_maps.CONSTRAINT_SEVERITY_MAP), most severe wins:
    contains hate                   -> hate_label=1, severity=hate
    contains offensive/defamation   -> hate_label=1, severity=offensive_profanity
    exactly non-hostile             -> hate_label=0, severity=normal
    only fake                       -> DROPPED (misinformation, not abuse)
    any unrecognised label          -> DROPPED and counted (never guessed)

Text is native Devanagari Hindi, which matches ASR output and fills the
native-script Hindi gap in the training data.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from label_maps import CONSTRAINT_SEVERITY_MAP, SEVERITY_CLASSES  # noqa: E402

_TEXT_COLS = ("post", "text")
_LABEL_COLS = ("labels set", "label", "labels")
_COLUMNS = ["text", "language", "hate_label", "target_label",
            "severity_label", "rationale_spans", "source", "split"]


def _find_col(df: pd.DataFrame, candidates) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    return next((lower[c] for c in candidates if c in lower), None)


def resolve_labels(raw: str) -> tuple[int, str] | None:
    """
    Map one post's label set to (hate_label, severity) or None to drop it.
    Unknown labels drop the row rather than defaulting to hateful.
    """
    labels = {t.strip().casefold() for t in str(raw).split(",") if t.strip()}
    if not labels or labels - set(CONSTRAINT_SEVERITY_MAP):
        return None
    if "hate" in labels:
        return 1, "hate"
    if labels & {"offensive", "defamation"}:
        return 1, "offensive_profanity"
    if labels == {"non-hostile"}:
        return 0, "normal"
    return None  # fake-only, or non-hostile mixed with fake (contradictory)


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    root = Path(raw_dir) / "constraint2021"
    rows = []
    dropped = Counter()

    for filename, our_split in [("train.csv", "train"), ("val.csv", "val")]:
        path = root / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, on_bad_lines="skip")
        text_col, label_col = _find_col(df, _TEXT_COLS), _find_col(df, _LABEL_COLS)
        if text_col is None or label_col is None:
            print(f"  [constraint2021] WARNING: {filename} columns {list(df.columns)} — "
                  f"expected one of {_TEXT_COLS} and one of {_LABEL_COLS}; skipping")
            continue

        for _, r in df.dropna(subset=[text_col, label_col]).iterrows():
            text = str(r[text_col]).strip()
            if not text:
                continue
            resolved = resolve_labels(r[label_col])
            if resolved is None:
                dropped[str(r[label_col]).strip().casefold()] += 1
                continue
            hate_label, severity = resolved
            rows.append({
                "text": text,
                "language": "hi",
                "hate_label": hate_label,
                "target_label": -1,
                "severity_label": SEVERITY_CLASSES.index(severity),
                "rationale_spans": "[]",
                "source": "constraint2021",
                "split": our_split,
            })

    if dropped:
        print(f"  [constraint2021] dropped {sum(dropped.values())} rows "
              f"(fake-only / unknown labels): {dict(dropped.most_common(8))}")
    return pd.DataFrame(rows, columns=_COLUMNS)


if __name__ == "__main__":
    df = convert()
    print(f"CONSTRAINT 2021: {df.shape}")
    if not df.empty:
        print(df["hate_label"].value_counts())
        print(df["severity_label"].value_counts())
