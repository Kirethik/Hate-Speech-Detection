"""
Runs every converters/<source>.py and concatenates their output into
data/train.csv / data/val.csv (the files train.py expects).

Configuration is loaded from data_config.yaml (per-source caps and the
test-only source list). Per-language weights in the same file are NOT applied
here: prepare_splits.py dedups train, which would delete repeated rows again.
train.py applies them as sampling weights instead (--lang_weights).

Usage:
    python build_dataset.py [--raw_dir data/raw] [--output_dir data] [--config data_config.yaml]
    python build_dataset.py --no_implicit   # skip English implicit-hate sources
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from text_norm import detect_script

from converters import (
    constraint2021,
    dravidiancodemix,
    dravidianlt,
    dynahate,
    hasoc,
    hatexplain,
    ieee_razi,
    implicit_hate,
    macd,
    ruhsold,
    sbic,
    toxigen,
)

SCHEMA_COLUMNS = [
    "text",
    "language",
    "hate_label",
    "target_label",
    "severity_label",
    "rationale_spans",
    "source",
    "script",
]

# Sources that must NEVER enter the training pool — hard-coded as a safety net
# in addition to what data_config.yaml says.
_ALWAYS_TEST_ONLY = frozenset({"hatecheck", "multilingual_hatecheck"})


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"WARNING: {config_path} not found — using empty config (no caps, no weights).")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _assert_no_test_only_in_pool(df: pd.DataFrame, test_only: set) -> None:
    """Hard assertion: functional test sources must never enter the training pool."""
    contaminated = set(df["source"].unique()) & test_only
    if contaminated:
        print(f"FATAL: test-only sources found in training pool: {contaminated}", file=sys.stderr)
        print("Remove them from the source list in build_dataset.py and re-run.", file=sys.stderr)
        sys.exit(1)


def _converter_failed(name: str, err: Exception, skip: bool) -> None:
    """A converter exception is a bug (missing data returns 0 rows), so stop by default."""
    msg = f"{name} converter raised {type(err).__name__}: {err}"
    if not skip:
        raise SystemExit(f"ERROR: {msg}\n(re-run with --skip_failed to build without it)")
    print(f"WARNING: {msg} -- skipping (--skip_failed)")


def main():
    # Windows consoles default to cp1252; never crash on an Indic/Urdu label in a summary
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--config", default="data_config.yaml")
    parser.add_argument("--no_implicit", action="store_true",
                        help="skip the four English implicit-hate sources")
    parser.add_argument("--skip_failed", action="store_true",
                        help="continue when a converter raises (default: stop). Missing "
                             "data is not a failure; converters return 0 rows for that.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    caps: dict = cfg.get("per_source_caps", {})
    test_only_cfg: list = cfg.get("test_only_sources", [])
    test_only: set = _ALWAYS_TEST_ONLY | set(test_only_cfg)
    min_rows: int = cfg.get("min_train_rows_per_language", 3000)

    # ── Local sources (always included) ───────────────────────────────────
    local_sources = [
        ("hatexplain",       hatexplain,       lambda m, d: m.convert(d)),
        ("dravidiancodemix", dravidiancodemix, lambda m, d: m.convert(d)),
        ("macd",             macd,             lambda m, d: m.convert(d)),
        ("ruhsold",          ruhsold,          lambda m, d: m.convert(d)),
        ("hasoc",            hasoc,            lambda m, d: m.convert(d)),
        ("dravidianlt",      dravidianlt,      lambda m, d: m.convert(d)),
        ("constraint2021",   constraint2021,   lambda m, d: m.convert(d)),
        ("ieee_razi",        ieee_razi,        lambda m, d: m.convert(d)),
    ]

    # ── English implicit-hate sources (cappable) ───────────────────────
    implicit_sources = [
        ("implicithate", implicit_hate, lambda m, d, cap: m.convert(d, max_rows=cap)),
        ("toxigen",      toxigen,       lambda m, d, cap: m.convert(d, max_rows=cap)),
        ("sbic",         sbic,          lambda m, d, cap: m.convert(d, max_rows=cap)),
        ("dynahate",     dynahate,      lambda m, d, cap: m.convert(d, max_rows=cap)),
    ]

    frames = []

    for name, module, loader in local_sources:
        try:
            df = loader(module, args.raw_dir)
            if df.empty:
                print(f"{name}: 0 rows (data not found at {args.raw_dir}/{name} — skipping)")
                continue
            print(f"{name}: {len(df)} rows  splits={df['split'].value_counts().to_dict()}"
                  f"  langs={df['language'].value_counts().to_dict()}")
            frames.append(df)
        except Exception as e:
            _converter_failed(name, e, args.skip_failed)

    if not args.no_implicit:
        for name, module, loader in implicit_sources:
            cap = caps.get(name)
            try:
                df = loader(module, args.raw_dir, cap)
                if df.empty:
                    print(f"{name}: 0 rows — skipping")
                    continue
                note = f" (capped at {cap})" if cap and len(df) >= cap else ""
                print(f"{name}: {len(df)} rows{note}")
                frames.append(df)
            except Exception as e:
                _converter_failed(name, e, args.skip_failed)

    if not frames:
        print("ERROR: No data was loaded. Check that data/raw/ is populated "
              "(expected layout: docs/SOURCES.md).", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["text"].astype(str).str.strip().str.len() > 0]

    # Safety assertion — must run before anything is written
    _assert_no_test_only_in_pool(combined, test_only)

    # Script is derived from the text, so every source gets it the same way
    combined["script"] = combined["text"].astype(str).map(detect_script)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val"]:
        split_df = combined[combined["split"] == split_name][SCHEMA_COLUMNS].copy()
        out_path = out_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"\nwrote {len(split_df):>7} rows -> {out_path}")
        print("  per-language:", split_df["language"].value_counts().to_dict())
        print("  per-source:  ", split_df["source"].value_counts().to_dict())
        print("  hate_label:  ", split_df["hate_label"].value_counts().to_dict())
        print("  script:      ", split_df["script"].value_counts().to_dict())

        if split_name == "train":
            for lang, count in split_df["language"].value_counts().items():
                if count < min_rows:
                    print(f"  WARNING: {lang} has only {count} train rows "
                          f"(min_train_rows_per_language={min_rows})")

    # Refresh backup so prepare_splits.py sees the latest build
    backup = out_dir / "_original"
    backup.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val"]:
        shutil.copy2(out_dir / f"{split_name}.csv", backup / f"{split_name}.csv")
    print(f"\nrefreshed {backup}: prepare_splits.py will see this build")


if __name__ == "__main__":
    main()
