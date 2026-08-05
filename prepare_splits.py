"""
Turns the raw concatenated build_dataset.py output into trustworthy splits.

build_dataset.py just concatenates each converter's own train/val split. That is
fine for getting data on disk, but it produces an evaluation set you cannot
believe, for three separate reasons:

  1. LEAKAGE. The same text appears in both train and val (981 rows, 3.71% of
     val — mostly DravidianCodeMix, which ships overlapping train/dev files).
     A model memorising those rows scores higher without generalising better.
  2. NO TEST SET. HateXplain's official val AND test both got folded into "val",
     and train.py picks its best checkpoint by val macro-F1 — so the number that
     gets reported is the number that was *optimised against*. That is
     optimistically biased by construction, however honest the training loop is.
  3. CONTRADICTIONS + DUPLICATES. 4,576 duplicate rows inside train and 33 texts
     carrying both hate_label 0 and 1 across sources.

This script fixes all three and then *proves* it did, by re-checking leakage on
its own output before writing anything.

Dedup key is deliberately loose (casefold, strip every non-word character) so
that "you idiot!!!" and "You idiot" collapse together — social-media corpora are
full of punctuation/emoji variants of the same string, and exact-match dedup
would leave most of that leakage in place.

Usage:
    python prepare_splits.py            # rewrites data/{train,val,test}.csv
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SCHEMA_COLUMNS = [
    "text", "language", "hate_label", "target_label",
    "severity_label", "rationale_spans", "source",
]

# Rows carrying target/severity/rationale annotation are worth more than rows
# with only a hate label, so when two rows collide on the dedup key we keep the
# richer one rather than whichever happened to be read first.
def _richness(df: pd.DataFrame) -> pd.Series:
    has_target = (df["target_label"].fillna(-1).astype(int) >= 0).astype(int)
    has_severity = (df["severity_label"].fillna(-1).astype(int) >= 0).astype(int)
    has_rationale = (df["rationale_spans"].astype(str).str.len() > 2).astype(int)
    return has_target + has_severity + has_rationale


_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def dedup_key(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.casefold()
    return s.str.replace(_NON_WORD, "", regex=True)


def dedup(df: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(df)
    out = (
        df.assign(_r=_richness(df))
          .sort_values("_r", ascending=False, kind="stable")
          .drop_duplicates("_key", keep="first")
          .drop(columns="_r")
          .sort_index()
    )
    print(f"  dedup {label}: {before} -> {len(out)} (-{before - len(out)})")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="fraction of the held-out pool reserved for test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path, val_path = data_dir / "train.csv", data_dir / "val.csv"

    # Back up the originals once, so this script is safe to re-run.
    backup = data_dir / "_original"
    if not backup.exists():
        backup.mkdir(parents=True)
        for p in (train_path, val_path):
            if p.exists():
                shutil.copy2(p, backup / p.name)
        print(f"backed up original splits -> {backup}")
    src_dir = backup if (backup / "train.csv").exists() else data_dir

    train = pd.read_csv(src_dir / "train.csv")
    pool = pd.read_csv(src_dir / "val.csv")  # becomes val + test
    print(f"loaded: train={len(train)}  held-out pool={len(pool)}")

    for df in (train, pool):
        df["text"] = df["text"].astype(str)
        df["_key"] = dedup_key(df["text"])

    # --- 1. drop empty / degenerate text -------------------------------------
    for name, df in (("train", train), ("pool", pool)):
        bad = df["_key"].str.len() == 0
        if bad.any():
            print(f"  dropping {bad.sum()} empty-after-normalisation rows from {name}")
    train = train[train["_key"].str.len() > 0].copy()
    pool = pool[pool["_key"].str.len() > 0].copy()

    # --- 2. drop texts with contradictory hate labels ------------------------
    both = pd.concat([train[["_key", "hate_label"]], pool[["_key", "hate_label"]]])
    conflicting = both.groupby("_key")["hate_label"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    if conflicting:
        print(f"  dropping {len(conflicting)} texts labelled both 0 and 1 "
              f"({train['_key'].isin(conflicting).sum()} train + "
              f"{pool['_key'].isin(conflicting).sum()} eval rows)")
        train = train[~train["_key"].isin(conflicting)].copy()
        pool = pool[~pool["_key"].isin(conflicting)].copy()

    # --- 3. dedup within each side -------------------------------------------
    train = dedup(train, "train")
    pool = dedup(pool, "held-out pool")

    # --- 4. remove train->eval leakage ---------------------------------------
    train_keys = set(train["_key"])
    leaked = pool["_key"].isin(train_keys)
    print(f"  removing {leaked.sum()} leaked eval rows "
          f"({leaked.mean():.2%} of pool) also present in train")
    if leaked.any():
        print("    by source:", pool[leaked]["source"].value_counts().to_dict())
    pool = pool[~leaked].copy()

    # --- 5. split the clean pool into val (model selection) and test (report) -
    # Stratify on source x language x hate_label so both halves stay comparable
    # per-slice; fall back to coarser strata if any cell is too small to split.
    strata = (pool["source"] + "|" + pool["language"] + "|" + pool["hate_label"].astype(str))
    if strata.value_counts().min() < 2:
        print("  note: some source|language|label cells are singletons, "
              "stratifying on source|label instead")
        strata = pool["source"] + "|" + pool["hate_label"].astype(str)
    val, test = train_test_split(
        pool, test_size=args.test_size, random_state=args.seed, stratify=strata
    )

    # --- 6. verify, then write ------------------------------------------------
    val_keys, test_keys = set(val["_key"]), set(test["_key"])
    assert not (train_keys & val_keys), "leakage train<->val survived"
    assert not (train_keys & test_keys), "leakage train<->test survived"
    assert not (val_keys & test_keys), "leakage val<->test survived"
    print("  verified: zero key overlap between train / val / test")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        out = df[SCHEMA_COLUMNS]
        out.to_csv(data_dir / f"{name}.csv", index=False)
        print(f"\nwrote {len(out):>6} rows -> {data_dir / (name + '.csv')}")
        print("  hate_label:", out["hate_label"].value_counts().to_dict())
        print("  language:  ", out["language"].value_counts().to_dict())
        print("  source:    ", out["source"].value_counts().to_dict())
        print("  labelled target/severity/rationale:",
              int((out['target_label'] >= 0).sum()),
              int((out['severity_label'] >= 0).sum()),
              int((out['rationale_spans'].astype(str).str.len() > 2).sum()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
