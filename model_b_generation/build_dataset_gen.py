"""
Step 1 of the Model B data build: run every converters_gen source and write
one table of (task, source_text, target_text, ...) pairs.

    python -m model_b_generation.build_dataset_gen [--raw_dir data/raw] [--out data/gen/pairs.parquet]

Next: python -m model_b_generation.prepare_splits_gen
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from model_b_generation.converters_gen import PAIR_COLUMNS, SOURCES
from model_b_generation.prompts import LANGUAGE_NAMES, TASKS

logger = logging.getLogger(__name__)


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(PAIR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    bad_task = ~df["task"].isin(TASKS)
    bad_lang = ~df["language"].isin(LANGUAGE_NAMES)
    if bad_task.any() or bad_lang.any():
        logger.warning("%s: dropping %d rows with unknown task/language", name,
                       int((bad_task | bad_lang).sum()))
        df = df[~(bad_task | bad_lang)]
    return df[PAIR_COLUMNS]


def build(raw_dir: str) -> pd.DataFrame:
    frames = []
    for name, module in SOURCES.items():
        try:
            df = module.convert(raw_dir)
        except Exception as e:  # one broken source must not stop the build
            logger.error("%s failed: %s", name, e)
            continue
        if df.empty:
            print(f"  {name:<18} 0 rows (not downloaded?)")
            continue
        df = validate(df, name)
        print(f"  {name:<18} {len(df):>7} rows  {df.groupby(['task', 'language']).size().to_dict()}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out", default="data/gen/pairs.parquet")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df = build(a.raw_dir)
    if df.empty:
        print("No Model B data found. See docs/SOURCES.md for where each dataset goes.")
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {len(df)} pairs -> {out}")
    print(df.groupby(["task", "language", "split"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
