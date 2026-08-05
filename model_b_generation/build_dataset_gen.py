"""
Runs every converters_gen/<source>.py and concatenates their output into
data/train_gen.csv / data/val_gen.csv (the files train_gen.py expects).

Mirrors the structure of Model A's build_dataset.py.

Each source module carries its own hash-based train/val split, so no
global stratified split is needed here — concatenating per-source splits
preserves language representation in both train and val.

Usage:
    python -m model_b_generation.build_dataset_gen [--output_dir data]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python build_dataset_gen.py` from model_b_generation/
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from converters_gen import conan, multitarget_conan, indic_conan, lt_edi, ter_mini

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = [
    "hate_text", "language", "style", "response_text", "source",
]

# Ordered: primary sources first, then fallback, then multilingual
SOURCES = [
    ("indic_conan",       indic_conan,       "PRIMARY — IndicCONAN (Hindi/English)"),
    ("lt_edi",            lt_edi,             "PRIMARY — LT-EDI 2026 (English/Tamil)"),
    ("conan",             conan,              "FALLBACK — CONAN ACL 2019 (English)"),
    ("multitarget_conan", multitarget_conan,  "FALLBACK — Multitarget-CONAN (English)"),
    ("ter_mini",          ter_mini,           "MULTILINGUAL — TER Mini-Corpus (en/hi/ta)"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Build the counter-narrative generation dataset"
    )
    parser.add_argument("--output_dir", default="data",
                        help="Directory to write train_gen.csv / val_gen.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    used_sources = []
    skipped_sources = []

    print("=" * 72)
    print("  Model B Dataset Builder — converters_gen")
    print("=" * 72)

    for name, module, label in SOURCES:
        print(f"\n{'─' * 60}")
        print(f"  [{label}]  {name}")
        try:
            df = module.convert()
        except Exception as e:
            logger.error("  converter %s raised: %s", name, e)
            df = pd.DataFrame()

        if df is None or df.empty:
            skipped_sources.append(name)
            print(f"  ⚠  {name}: 0 rows (skipped or unavailable)")
        else:
            # Validate schema
            missing = set(SCHEMA_COLUMNS) - set(df.columns)
            if missing:
                logger.error(
                    "  %s is missing columns %s — skipping", name, missing
                )
                skipped_sources.append(name)
                continue

            # Validate no nulls in required fields
            null_counts = df[["hate_text", "response_text"]].isnull().sum()
            if null_counts.any():
                before = len(df)
                df = df.dropna(subset=["hate_text", "response_text"])
                logger.warning(
                    "  %s: dropped %d rows with null hate_text/response_text",
                    name, before - len(df),
                )

            split_counts = (
                df["split"].value_counts().to_dict()
                if "split" in df.columns
                else {"unsplit": len(df)}
            )
            lang_counts = df["language"].value_counts().to_dict()

            print(f"  ✓  {name}: {len(df)} rows")
            print(f"       splits:    {split_counts}")
            print(f"       languages: {lang_counts}")

            used_sources.append(name)
            frames.append(df)

    if not frames:
        logger.error("No data collected from any source. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["hate_text"].astype(str).str.strip().str.len() > 0]

    # ── Write train / val splits ──
    for split_name in ["train", "val"]:
        if "split" in combined.columns:
            split_df = combined[combined["split"] == split_name][SCHEMA_COLUMNS]
        else:
            # If no split column, default all to train
            split_df = combined[SCHEMA_COLUMNS] if split_name == "train" else pd.DataFrame(columns=SCHEMA_COLUMNS)

        out_path = out_dir / f"{split_name}_gen.csv"
        split_df.to_csv(out_path, index=False)
        print(f"\nwrote {len(split_df):>6} rows → {out_path}")
        if len(split_df) > 0:
            print(f"  per-language: {split_df['language'].value_counts().to_dict()}")
            print(f"  per-source:   {split_df['source'].value_counts().to_dict()}")
            print(f"  per-style:    {split_df['style'].value_counts().to_dict()}")

    # ── Audit log ──
    print(f"\n{'=' * 72}")
    print("  AUDIT SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Sources USED:    {used_sources}")
    print(f"  Sources SKIPPED: {skipped_sources}")
    print(f"  Total rows:      {len(combined)}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
