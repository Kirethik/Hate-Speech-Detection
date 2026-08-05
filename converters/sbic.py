"""
Converts the Social Bias Inference Corpus (SBIC; Sap et al., ACL 2020) to the
unified schema.

SBIC's value here is `whoTarget`: it distinguishes abuse aimed at an INDIVIDUAL
from abuse aimed at a GROUP. That is exactly the offensive-vs-hate axis the
severity head is supposed to model, and it is the axis the original corpus was
starved of. So:

    offensive + group-targeted    -> severity = hate
    offensive + individual        -> severity = offensive_profanity
    not offensive                 -> severity = normal

SBIC is distributed at ANNOTATOR level — the same post appears once per
annotator, with that annotator's judgement. Reading it row-wise would (a)
duplicate popular posts many times over, silently over-weighting them, and
(b) treat one annotator's minority opinion as ground truth. So aggregate to one
row per post by mean-pooling the numeric judgements first.

The mirror `Ayush-Singh/social-bias-frames-splits` is used because
`allenai/social_bias_frames` ships only a loading script, which datasets>=3.0
no longer executes. Its "splits" are target categories, not train/val, so they
are concatenated and re-split by hash.
"""

import pandas as pd
from datasets import load_dataset

from dataset import SEVERITY_CLASSES, TARGET_CLASSES

from ._hf import cap, hash_split

REPO = "Ayush-Singh/social-bias-frames-splits"

TARGET_MAP = {
    "race": "caste_ethnicity",
    "gender": "gender",
    "culture": "religion",       # SBIC's culture bucket is religion/culture
    "disabled": "disability",
    "social": "political",
    "body": "other",
    "victim": "other",
}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", None), errors="coerce")


def convert(raw_dir="raw_data", max_rows: int | None = None):
    ds = load_dataset(REPO)
    df = pd.concat([ds[k].to_pandas() for k in ds.keys()], ignore_index=True)
    df = df.dropna(subset=["post"])
    df["post"] = df["post"].astype(str).str.strip()
    df = df[df["post"].str.len() > 0]

    df["_off"] = _num(df["offensiveYN"])
    df["_grp"] = _num(df["whoTarget"])
    df = df.dropna(subset=["_off"])

    agg = df.groupby("post").agg(
        off=("_off", "mean"),
        grp=("_grp", "mean"),
        category=("targetCategory", lambda s: next((v for v in s if str(v).strip()), "")),
    ).reset_index()

    # 0.5 means the annotators genuinely disagreed about whether it is offensive.
    # Forcing those to a side would inject label noise into precisely the
    # ambiguous region the decision threshold has to get right, so drop them.
    agg = agg[(agg["off"] < 0.5) | (agg["off"] > 0.5)]

    rows = []
    for _, r in agg.iterrows():
        offensive = bool(r["off"] > 0.5)
        group_targeted = bool(pd.notna(r["grp"]) and r["grp"] >= 0.5)
        if not offensive:
            severity = "normal"
        elif group_targeted:
            severity = "hate"
        else:
            severity = "offensive_profanity"

        cat = str(r["category"]).strip().casefold()
        if offensive and group_targeted and cat in TARGET_MAP:
            target = TARGET_CLASSES.index(TARGET_MAP[cat])
        elif not offensive:
            target = TARGET_CLASSES.index("none")
        else:
            target = -1

        rows.append(
            {
                "text": r["post"],
                "language": "en",
                "hate_label": int(offensive),
                "target_label": target,
                "severity_label": SEVERITY_CLASSES.index(severity),
                "rationale_spans": "[]",
                "source": "sbic",
                "split": hash_split(r["post"]),
            }
        )
    return cap(pd.DataFrame(rows), max_rows)


if __name__ == "__main__":
    d = convert()
    print(d.shape, d["split"].value_counts().to_dict())
    print("hate:", d["hate_label"].value_counts().to_dict())
    print("severity:", d["severity_label"].value_counts().to_dict())
    print(d.head(3).to_string())
