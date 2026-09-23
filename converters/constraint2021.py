"""
CONSTRAINT 2021 Hindi Hostility Detection converter.

Expected directory layout:
    data/raw/constraint2021/
        train.csv    (columns: post_id, text, label)
        val.csv
        test.csv     (may not have labels)

Labels (6 classes from the shared task):
    Defamation       → hate_label=1, severity=hate
    Fake News        → hate_label=1, severity=offensive_profanity
    Hate Speech      → hate_label=1, severity=hate
    Offensive        → hate_label=1, severity=offensive_profanity
    Non-hostile      → hate_label=0, severity=normal
    Hostile          → hate_label=1, severity=offensive_profanity (fallback, ambiguous)

Text is native Devanagari Hindi — critically, this matches ASR output and
fills the native-script Hindi gap in the training data.
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from label_maps import SEVERITY_CLASSES  # noqa: E402
from dataset import detect_script  # noqa: E402


_LABEL_MAP = {
    "Defamation":   ("hate",                 1),
    "Fake News":    ("offensive_profanity",   1),
    "Hate Speech":  ("hate",                 1),
    "Offensive":    ("offensive_profanity",   1),
    "Hostile":      ("offensive_profanity",   1),   # fallback — user confirmed
    "Non-hostile":  ("normal",               0),
}


def convert(raw_dir: str = "raw_data") -> pd.DataFrame:
    root = Path(raw_dir) / "constraint2021"
    rows = []

    for filename, our_split in [("train.csv", "train"), ("val.csv", "val")]:
        path = root / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["text", "label"])

        for _, r in df.iterrows():
            text = str(r["text"]).strip()
            label = str(r["label"]).strip()
            if not text:
                continue
            severity_str, hate_label = _LABEL_MAP.get(label, ("offensive_profanity", 1))
            rows.append({
                "text": text,
                "language": "hi",
                "script": detect_script(text),
                "hate_label": hate_label,
                "target_label": -1,
                "severity_label": SEVERITY_CLASSES.index(severity_str),
                "rationale_spans": "[]",
                "source": "constraint2021",
                "split": our_split,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "text", "language", "script", "hate_label", "target_label",
            "severity_label", "rationale_spans", "source", "split",
        ])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = convert()
    print(f"CONSTRAINT 2021: {df.shape}")
    if not df.empty:
        print(df["hate_label"].value_counts())
        print(df["severity_label"].value_counts())
        print(df["script"].value_counts())
