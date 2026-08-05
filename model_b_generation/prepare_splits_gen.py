"""
Turns the raw build_dataset_gen.py output into trustworthy splits.

Mirrors Model A's prepare_splits.py pattern:
  1. Dedup by normalized text key (casefold + strip non-word chars).
  2. Remove train/val overlap.
  3. Strip PII patterns (emails, phone numbers, URLs).
  4. Check for overlap with Model A's own training data.
  5. Split the held-out pool into val (model selection) and test (report).
  6. Verify zero leakage between all splits.

Usage:
    python -m model_b_generation.prepare_splits_gen [--data_dir data]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SCHEMA_COLUMNS = [
    "hate_text", "language", "style", "response_text", "source",
]

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)

# PII patterns
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
_URL = re.compile(r"https?://\S+|www\.\S+")


def dedup_key(series: pd.Series) -> pd.Series:
    """Loose dedup key: casefold + strip non-word characters."""
    s = series.astype(str).str.strip().str.casefold()
    return s.str.replace(_NON_WORD, "", regex=True)


def strip_pii(text: str) -> str:
    """Replace email, phone, URL patterns with [REDACTED]."""
    text = _EMAIL.sub("[REDACTED]", text)
    text = _PHONE.sub("[REDACTED]", text)
    text = _URL.sub("[REDACTED]", text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clean train/val/test splits for Model B"
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="Fraction of held-out pool reserved for test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train_gen.csv"
    val_path = data_dir / "val_gen.csv"

    if not train_path.exists() or not val_path.exists():
        print("Input files not found. Run build_dataset_gen.py first.")
        sys.exit(1)

    train = pd.read_csv(train_path)
    pool = pd.read_csv(val_path)
    print(f"loaded: train={len(train)}  held-out pool={len(pool)}")

    # ── 1. Strip PII ────────────────────────────────────────────────────
    for df in (train, pool):
        df["hate_text"] = df["hate_text"].astype(str).apply(strip_pii)
        df["response_text"] = df["response_text"].astype(str).apply(strip_pii)
    print("  stripped PII patterns (email/phone/URL)")

    # ── 2. Compute dedup keys ───────────────────────────────────────────
    for df in (train, pool):
        df["_key"] = dedup_key(df["hate_text"])

    # ── 3. Drop empty text ──────────────────────────────────────────────
    for name, df in [("train", train), ("pool", pool)]:
        bad = df["_key"].str.len() == 0
        if bad.any():
            print(f"  dropping {bad.sum()} empty rows from {name}")
    train = train[train["_key"].str.len() > 0].copy()
    pool = pool[pool["_key"].str.len() > 0].copy()

    # ── 4. Dedup within each split ──────────────────────────────────────
    before = len(train)
    train = train.drop_duplicates("_key", keep="first")
    print(f"  dedup train: {before} → {len(train)} (-{before - len(train)})")

    before = len(pool)
    pool = pool.drop_duplicates("_key", keep="first")
    print(f"  dedup pool:  {before} → {len(pool)} (-{before - len(pool)})")

    # ── 5. Remove train→val leakage ─────────────────────────────────────
    train_keys = set(train["_key"])
    leaked = pool["_key"].isin(train_keys)
    if leaked.any():
        print(f"  removing {leaked.sum()} leaked eval rows "
              f"({leaked.mean():.2%} of pool)")
        pool = pool[~leaked].copy()

    # ── 6. Check for overlap with Model A's training data ───────────────
    model_a_train = data_dir / "train.csv"
    if model_a_train.exists():
        ma_df = pd.read_csv(model_a_train)
        ma_keys = set(dedup_key(ma_df["text"]))
        overlap_train = train["_key"].isin(ma_keys).sum()
        overlap_pool = pool["_key"].isin(ma_keys).sum()
        print(f"  overlap with Model A train: {overlap_train} train, "
              f"{overlap_pool} pool (informational — not removed, different task)")
    else:
        print("  Model A's data/train.csv not found — skipping overlap check")

    # ── 7. Split pool into val + test ───────────────────────────────────
    if len(pool) < 4:
        print("  WARNING: pool too small for a meaningful val/test split")
        val, test = pool, pd.DataFrame(columns=pool.columns)
    else:
        strata = pool["source"] + "|" + pool["language"]
        if strata.value_counts().min() < 2:
            strata = pool["language"]
        val, test = train_test_split(
            pool, test_size=args.test_size,
            random_state=args.seed, stratify=strata,
        )

    # ── 8. Verify zero leakage ──────────────────────────────────────────
    val_keys, test_keys = set(val["_key"]), set(test["_key"])
    assert not (train_keys & val_keys), "leakage train↔val survived"
    assert not (train_keys & test_keys), "leakage train↔test survived"
    assert not (val_keys & test_keys), "leakage val↔test survived"
    print("  verified: zero key overlap between train / val / test")

    # ── 9. Write ────────────────────────────────────────────────────────
    for name, df in [("train_gen", train), ("val_gen", val), ("test_gen", test)]:
        out = df[SCHEMA_COLUMNS]
        out_path = data_dir / f"{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"\nwrote {len(out):>6} rows → {out_path}")
        if len(out) > 0:
            print(f"  language:  {out['language'].value_counts().to_dict()}")
            print(f"  source:    {out['source'].value_counts().to_dict()}")
            print(f"  style:     {out['style'].value_counts().to_dict()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
