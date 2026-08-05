"""
Runs every converters/<source>.py and concatenates their output into
data/train.csv / data/val.csv (the files train.py expects).

Each source module already carries its own train/val split (see each
converter's docstring for how that split was derived), so no global
stratified split is needed here — concatenating per-source splits keeps
every language represented in both train and val, per the README's
instruction.

Usage:
    python build_dataset.py [--raw_dir raw_data] [--output_dir data]
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from converters import (
    dravidiancodemix, dynahate, hatexplain, implicit_hate, macd, ruhsold, sbic,
    toxigen,
)

# Per-source row caps for the English implicit-hate additions.
#
# These four sources are English-only, and uncapped they total ~87k rows against
# 16k English in the original corpus — importing all of them would turn a
# multilingual detector into a mostly-English one, buying English gains with
# Indic losses. ImplicitHate and ToxiGen are kept whole because they are the
# most on-target (pure implicit hate; balanced with identity-mention negatives);
# the two larger, noisier sources are capped.
DEFAULT_CAPS = {
    "implicithate": None,    # 6,346 — all of it, most on-target source
    "toxigen": None,         # 8,960 — balanced, supervises the target head
    "sbic": 15_000,          # of 31,472 — carries the offensive-vs-hate axis
    "dynahate": 15_000,      # of 40,623 — adversarial, largest source
}

SCHEMA_COLUMNS = [
    "text",
    "language",
    "hate_label",
    "target_label",
    "severity_label",
    "rationale_spans",
    "source",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="raw_data")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--no_implicit", action="store_true",
                        help="skip the four English implicit-hate sources")
    parser.add_argument("--implicit_cap", type=int, default=None,
                        help="override the per-source cap for sbic/dynahate")
    args = parser.parse_args()

    local_sources = [
        ("hatexplain", hatexplain),
        ("dravidiancodemix", dravidiancodemix),
        ("macd", macd),
        ("ruhsold", ruhsold),
    ]
    implicit_sources = [
        ("implicithate", implicit_hate),
        ("toxigen", toxigen),
        ("sbic", sbic),
        ("dynahate", dynahate),
    ]

    frames = []
    for name, module in local_sources:
        df = module.convert(args.raw_dir)
        print(f"{name}: {len(df)} rows, {df['split'].value_counts().to_dict()}")
        frames.append(df)

    if not args.no_implicit:
        for name, module in implicit_sources:
            cap = DEFAULT_CAPS[name] if args.implicit_cap is None else args.implicit_cap
            df = module.convert(args.raw_dir, max_rows=cap)
            note = f" (capped from source)" if cap and len(df) >= cap else ""
            print(f"{name}: {len(df)} rows, {df['split'].value_counts().to_dict()}{note}")
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["text"].str.strip().str.len() > 0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val"]:
        split_df = combined[combined["split"] == split_name][SCHEMA_COLUMNS]
        out_path = out_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"wrote {len(split_df)} rows -> {out_path}")
        print("  per-language:", split_df["language"].value_counts().to_dict())
        print("  per-source:", split_df["source"].value_counts().to_dict())
        print("  hate_label balance:", split_df["hate_label"].value_counts().to_dict())

    # prepare_splits.py backs up its input to data/_original/ on first run and
    # thereafter reads from that backup, so that it stays idempotent. That means
    # a stale backup silently shadows any rebuild done here — the new sources
    # would never reach training. Refresh it so the backup matches what we just
    # wrote, which is exactly what "the unsplit build output" is supposed to be.
    backup = out_dir / "_original"
    backup.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val"]:
        shutil.copy2(out_dir / f"{split_name}.csv", backup / f"{split_name}.csv")
    print(f"\nrefreshed {backup} so prepare_splits.py sees this build, not the previous one")


if __name__ == "__main__":
    main()
