"""
Converts community-datasets/roman_urdu_hate_speech (Fine_Grained config,
data/raw/RUHSOLD/train.csv) to the unified schema.

Only the "train" split has labels — the HF dataset's "test" split is
unlabeled (withheld, shared-task style) and its "validation" split is a
duplicate of "train" (verified: identical content, likely a bug in that
dataset's loading script) — so this converter makes its own stratified
90/10 train/val split instead of trusting the upstream split names.

Label mapping (5 fine-grained classes -> this project's 3 SEVERITY_CLASSES):
  Normal              -> normal
  Profane/Untargeted  -> offensive_profanity
  Abusive/Offensive   -> offensive_profanity
  Religious Hate      -> hate
  Sexism              -> hate
hate_label is 0 only for Normal. target_label is left -1 (unknown): the
README's source table lists RUHSOLD as contributing hate_label + severity_label
only, not target.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from label_maps import SEVERITY_CLASSES, RUHSOLD_SEVERITY_MAP  # noqa: E402


def convert(raw_dir="data/raw"):
    path = Path(raw_dir) / "RUHSOLD" / "train.csv"
    df = pd.read_csv(path).dropna(subset=["tweet", "label"])

    label_names = ["Abusive/Offensive", "Normal", "Religious Hate", "Sexism", "Profane/Untargeted"]
    df["label_name"] = df["label"].apply(
        lambda i: label_names[int(i)] if isinstance(i, (int, float)) else i
    )

    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label_name"]
    )

    rows = []
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for _, r in split_df.iterrows():
            severity = RUHSOLD_SEVERITY_MAP[r["label_name"]]
            rows.append(
                {
                    "text": str(r["tweet"]).strip(),
                    "language": "ur_roman",
                    "hate_label": 0 if r["label_name"] == "Normal" else 1,
                    "target_label": -1,
                    "severity_label": SEVERITY_CLASSES.index(severity),
                    "rationale_spans": "[]",
                    "source": "ruhsold",
                    "split": split_name,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = convert()
    print(df.shape, df["split"].value_counts().to_dict())
    print(df.head(3))
