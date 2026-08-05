"""
Converts ShareChatAI/MACD (raw_data/MACD/dataset_80_10_10/<lang>_{train,val,test}.csv)
to the unified schema.

Per MACD's own README: label 0 = abusive, label 1 = non-abusive — the
INVERSE of this project's hate_label convention (1 = hate) — so we flip it.
Binary hate signal only; no target/severity annotations in this source.
"""

from pathlib import Path

import pandas as pd

LANGS = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malyalam": "ml",  # MACD's own filename spelling
}


def convert(raw_dir="raw_data"):
    root = Path(raw_dir) / "MACD" / "dataset_80_10_10"
    rows = []
    for lang_name, code in LANGS.items():
        for filesplit, our_split in [("train", "train"), ("val", "val")]:
            path = root / f"{lang_name}_{filesplit}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).dropna(subset=["text", "label"])
            for _, r in df.iterrows():
                rows.append(
                    {
                        "text": str(r["text"]).strip(),
                        "language": code,
                        "hate_label": 1 - int(r["label"]),
                        "target_label": -1,
                        "severity_label": -1,
                        "rationale_spans": "[]",
                        "source": "macd",
                        "split": our_split,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = convert()
    print(df.shape, df["split"].value_counts().to_dict())
    print(df["language"].value_counts())
    print(df.head(3))
