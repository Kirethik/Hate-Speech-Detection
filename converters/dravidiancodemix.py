"""
Converts bharathichezhiyan/DravidianCodeMix-Dataset (Tamil/Kannada/Malayalam
offensive-language files, extracted from DravidianCodeMix-2020.zip) to the
unified schema.

Only hate_label is populated. DravidianCodeMix's fine-grained labels
(Offensive_Targeted_Insult_Individual/Group/Other) describe WHO the target
IS STRUCTURALLY (a person vs. a group vs. something else), not WHICH
IDENTITY CATEGORY was attacked — that's a different axis than this project's
TARGET_CLASSES (religion/gender/caste_ethnicity/...), so per the README's own
warning, target_label is left as -1 (unknown/masked) rather than guessing a
mapping. Same for severity_label — this source has no severity annotation.

Rows labeled "not-<language>" mean the annotator judged the text isn't
actually in that code-mixed language (e.g. pure English/off-topic); those
aren't a hate/not-hate judgment, so they're dropped.
"""

from pathlib import Path

import pandas as pd

LANGS = {
    "tamil": ("ta", "tamil_offensive_full"),
    "kannada": ("kn", "kannada_offensive"),
    "malayalam": ("ml", "mal_full_offensive"),  # Note: The raw data might lack Malayalam files; dravidianlt.py (Phase 2) will cover it if so.
}

OFFENSIVE_LABELS = {
    "Offensive_Untargetede",
    "Offensive_Targeted_Insult_Individual",
    "Offensive_Targeted_Insult_Group",
    "Offensive_Targeted_Insult_Other",
}


def _load_file(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", header=None, quoting=3, on_bad_lines="skip", engine="python")
    df = df.rename(columns={0: "text", 1: "label"})
    return df[["text", "label"]]


def convert(raw_dir="data/raw"):
    root = Path(raw_dir) / "DravidianCodeMix-Dataset" / "DravidianCodeMix"
    rows = []
    for lang_name, (code, prefix) in LANGS.items():
        for filesplit, our_split in [("train", "train"), ("dev", "val")]:
            df = _load_file(root / f"{prefix}_{filesplit}.csv")
            if df is None:
                continue
            df = df.dropna(subset=["text", "label"])
            df = df[~df["label"].str.startswith("not-")]
            for _, r in df.iterrows():
                rows.append(
                    {
                        "text": str(r["text"]).strip(),
                        "language": code,
                        "hate_label": 1 if r["label"] in OFFENSIVE_LABELS else 0,
                        "target_label": -1,
                        "severity_label": -1,
                        "rationale_spans": "[]",
                        "source": "dravidiancodemix",
                        "split": our_split,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = convert()
    print(df.shape, df["split"].value_counts().to_dict())
    print(df["language"].value_counts())
    print(df.head(3))
