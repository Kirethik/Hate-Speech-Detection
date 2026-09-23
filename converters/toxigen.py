"""
Converts ToxiGen (Hartvigsen et al., ACL 2022) to the unified schema.

ToxiGen was machine-generated adversarially so that toxic and benign statements
about the same 13 target groups are lexically similar — a model cannot separate
them by spotting slurs, because there are none. That is precisely the shortcut
this project's detector was caught using.

Two properties make it worth more than its row count suggests:

  1. It is BALANCED (4,462 hate / 4,498 neutral here), and the neutral half is
     *benign statements that mention the same identity groups*. That is direct
     supervision against the identity-mention false positive the probe already
     tests for ("i am a muslim woman and i am proud of my heritage").
  2. It carries target_group, which is one of only two sources in the whole
     corpus that can supervise the target head at all (HateXplain was the other,
     at 8.5% coverage).

The CSV stores text as Python byte-string reprs (b'asians are ...'), which must
be stripped or every example trains on a spurious leading "b'".
"""

import pandas as pd
from huggingface_hub import hf_hub_download

from label_maps import TARGET_CLASSES, SEVERITY_CLASSES, TOXIGEN_TOXIGEN_TARGET_MAP, TARGET_CLASSES

from ._hf import cap, hash_split, strip_byte_prefix

REPO = "toxigen/toxigen-data"

# Only annotated_train.csv is used. annotated_test.csv (940 rows) ships WITHOUT
# the `label` column — only the 1-5 `toxicity_human` scale. Calibrating a
# threshold on annotated_train, where both columns exist, the best achievable
# agreement with the gold label is 80.3% (at >2.5). Importing the test split
# through that proxy would buy 940 rows at the cost of ~190 mislabelled ones,
# in the one source specifically chosen for its label quality. Not worth it.
FILENAME = "annotated_train.csv"

# ToxiGen's 13 demographic groups -> this project's 8-class target scheme.


def _target_index(raw) -> int:
    key = str(raw).strip().casefold().replace(" ", "_").replace("-", "_")
    if key in TOXIGEN_TARGET_MAP:
        return TARGET_CLASSES.index(TOXIGEN_TARGET_MAP[key])
    for needle, mapped in TOXIGEN_TARGET_MAP.items():  # e.g. "black folks" -> black
        if needle in key:
            return TARGET_CLASSES.index(mapped)
    return -1  # unknown group: mask rather than guess


def convert(raw_dir="raw_data", max_rows: int | None = None):
    path = hf_hub_download(REPO, FILENAME, repo_type="dataset")
    df = pd.read_csv(path).dropna(subset=["text", "label"])

    rows = []
    for _, r in df.iterrows():
        text = strip_byte_prefix(r["text"])
        if not text:
            continue
        is_hate = str(r["label"]).strip().casefold() == "hate"
        rows.append(
            {
                "text": text,
                "language": "en",
                "hate_label": int(is_hate),
                # Only the hateful half has a meaningful target; the neutral
                # half mentions the group without targeting it.
                "target_label": _target_index(r.get("target_group")) if is_hate else TARGET_CLASSES.index("none"),
                "severity_label": SEVERITY_CLASSES.index("hate" if is_hate else "normal"),
                "rationale_spans": "[]",
                "source": "toxigen",
                "split": hash_split(text),
            }
        )
    return cap(pd.DataFrame(rows), max_rows)


if __name__ == "__main__":
    d = convert()
    print(d.shape, d["split"].value_counts().to_dict())
    print(d["hate_label"].value_counts().to_dict())
    print(d.head(3).to_string())
