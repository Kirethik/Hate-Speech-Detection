"""
HateCheck (English) and Multilingual HateCheck converter.

IMPORTANT: These are FUNCTIONAL TEST sets only.
They must NEVER be added to the training pool.
build_dataset.py asserts this at startup.

Output goes to data/processed/functional_tests/ — NOT to data/train.csv.

HateCheck: https://github.com/paul-rottger/hatecheck-data
  File: hatecheck-data/test_suite_cases.csv
  Columns: functionality, test_case, label_gold (hateful/non-hateful)

Multilingual HateCheck: https://github.com/paul-rottger/multilingual-hatecheck
  File: multilingual_hatecheck/test_suite_cases.csv
  Same schema; language column added.

Usage:
    from converters.hatecheck import convert_hatecheck, convert_multilingual
    en_df = convert_hatecheck()
    multi_df = convert_multilingual()
"""

from pathlib import Path

import pandas as pd


def convert_hatecheck(raw_dir: str = "raw_data") -> pd.DataFrame:
    """English HateCheck functional test set."""
    path = Path(raw_dir) / "hatecheck" / "test_suite_cases.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "text", "language", "functionality", "hate_label", "source",
        ])
    df = pd.read_csv(path).dropna(subset=["test_case", "label_gold"])
    return pd.DataFrame({
        "text": df["test_case"].str.strip(),
        "language": "en",
        "functionality": df["functionality"],
        "hate_label": (df["label_gold"].str.lower() == "hateful").astype(int),
        "source": "hatecheck",
    })


def convert_multilingual(raw_dir: str = "raw_data") -> pd.DataFrame:
    """Multilingual HateCheck functional test set (en + hi included)."""
    path = Path(raw_dir) / "multilingual_hatecheck" / "test_suite_cases.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "text", "language", "functionality", "hate_label", "source",
        ])
    df = pd.read_csv(path).dropna(subset=["test_case", "label_gold"])
    lang_col = "language" if "language" in df.columns else "lang"
    return pd.DataFrame({
        "text": df["test_case"].str.strip(),
        "language": df[lang_col] if lang_col in df.columns else "en",
        "functionality": df.get("functionality", "unknown"),
        "hate_label": (df["label_gold"].str.lower() == "hateful").astype(int),
        "source": "multilingual_hatecheck",
    })


def save_functional_tests(raw_dir: str = "raw_data",
                           out_dir: str = "data/processed/functional_tests") -> None:
    """Save HateCheck sets to the functional-tests directory (never to training)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    en_df = convert_hatecheck(raw_dir)
    if not en_df.empty:
        en_df.to_csv(out / "hatecheck_en.csv", index=False)
        print(f"HateCheck EN: {len(en_df)} rows → {out / 'hatecheck_en.csv'}")

    multi_df = convert_multilingual(raw_dir)
    if not multi_df.empty:
        multi_df.to_csv(out / "multilingual_hatecheck.csv", index=False)
        print(f"Multilingual HateCheck: {len(multi_df)} rows → {out / 'multilingual_hatecheck.csv'}")


if __name__ == "__main__":
    save_functional_tests()
