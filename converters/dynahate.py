"""
Converts the Dynamically Generated Hate Speech Dataset (DynaHate; Vidgen et al.,
ACL 2021) to the unified schema.

DynaHate was built adversarially over four rounds: human annotators were shown a
live model and asked to write examples that FOOL it. Rounds 3-4 additionally
asked for perturbations of existing examples. The result is a corpus
concentrated in exactly the region where a profanity-keyed detector fails —
which is where this project's model currently sits.

The `type` column is the useful part beyond the binary label. It includes an
explicit `dehumanization` class (840 rows), which is the mechanism behind the
motivating failure case ("...are a plague on every town"). Those rows are also
what the dehumanization lexicon in `lexicons/dehumanization.py` is validated
against, so this converter keeps `type` reachable via the source name.

Loaded from the `sophieb/...` mirror's raw CSV: both `aps/dynahate` and
`tasksource/dynahate` fail on datasets>=3.0 (loading script removed / schema
cast error), but the mirror ships the original CSV intact.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

from dataset import SEVERITY_CLASSES

from ._hf import cap

REPO = "sophieb/dynamically_generated_hate_speech_dataset"
FILENAME = "2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv"

# DynaHate ships its own train/dev/test. prepare_splits.py re-pools and re-splits
# everything anyway, so dev and test both land in the held-out pool.
SPLIT_MAP = {"train": "train", "dev": "val", "test": "val"}


def convert(raw_dir="raw_data", max_rows: int | None = None):
    path = hf_hub_download(REPO, FILENAME, repo_type="dataset")
    df = pd.read_csv(path).dropna(subset=["text", "label"])

    rows = []
    for _, r in df.iterrows():
        text = str(r["text"]).strip()
        if not text:
            continue
        is_hate = str(r["label"]).strip().casefold() == "hate"
        rows.append(
            {
                "text": text,
                "language": "en",
                "hate_label": int(is_hate),
                "target_label": -1,  # target file is separate; entries carry none
                "severity_label": SEVERITY_CLASSES.index("hate" if is_hate else "normal"),
                "rationale_spans": "[]",
                "source": "dynahate",
                "split": SPLIT_MAP.get(str(r.get("split", "train")).strip().casefold(), "train"),
            }
        )
    return cap(pd.DataFrame(rows), max_rows)


if __name__ == "__main__":
    d = convert()
    print(d.shape, d["split"].value_counts().to_dict())
    print("hate:", d["hate_label"].value_counts().to_dict())
    print(d.head(3).to_string())
