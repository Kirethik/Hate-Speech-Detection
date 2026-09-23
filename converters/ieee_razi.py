"""
IEEE DataPort — Razi & Ejaz Roman-Urdu Hate Speech Dataset converter.

Expected directory layout:
    data/raw/ieee_razi/
        (CSV or Excel file downloaded from IEEE DataPort)

This dataset requires an IEEE account to download. The exact filename and
column schema are not documented in the paper. The converter inspects the
actual files at runtime and adapts to whatever columns are present.

Documented schema from the paper (Razi & Ejaz, 2022):
    - Column containing tweet text (name varies: "tweet", "text", "post")
    - Binary label column (name varies: "label", "class", "hate")
      Values observed: 0/1, hate/not-hate, yes/no

YOU: Download the dataset from https://ieee-dataport.org and place the
file(s) in data/raw/ieee_razi/. Then run this converter to see what
columns it found and whether the mapping looks correct.
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from converters._hf import hash_split  # noqa: E402
from dataset import detect_script  # noqa: E402


# Candidate column names for text and label — checked in priority order
_TEXT_CANDIDATES = ["tweet", "text", "post", "sentence", "content"]
_LABEL_CANDIDATES = ["label", "class", "hate", "hate_label", "category"]

# Values that map to hate_label=1
_HATE_VALUES = frozenset({"1", "hate", "yes", "hateful", "abusive", "offensive", "1.0"})


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _load_file(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    try:
        if suffix in (".csv",):
            return pd.read_csv(path, on_bad_lines="skip", engine="python")
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif suffix in (".tsv",):
            return pd.read_csv(path, sep="\t", on_bad_lines="skip", engine="python")
    except Exception:
        pass
    return None


def convert(raw_dir: str = "raw_data") -> pd.DataFrame:
    root = Path(raw_dir) / "ieee_razi"
    rows = []

    if not root.exists():
        return pd.DataFrame(columns=[
            "text", "language", "script", "hate_label", "target_label",
            "severity_label", "rationale_spans", "source", "split",
        ])

    data_files = sorted(
        list(root.glob("*.csv")) +
        list(root.glob("*.xlsx")) +
        list(root.glob("*.xls")) +
        list(root.glob("*.tsv"))
    )

    for path in data_files:
        df = _load_file(path)
        if df is None or df.empty:
            continue

        text_col = _find_col(df, _TEXT_CANDIDATES)
        label_col = _find_col(df, _LABEL_CANDIDATES)

        if text_col is None:
            print(f"  [ieee_razi] WARNING: no text column found in {path.name}. "
                  f"Columns: {list(df.columns)}")
            continue
        if label_col is None:
            print(f"  [ieee_razi] WARNING: no label column found in {path.name}. "
                  f"Columns: {list(df.columns)}")
            continue

        print(f"  [ieee_razi] {path.name}: text='{text_col}', label='{label_col}', "
              f"rows={len(df)}, label_values={df[label_col].value_counts().to_dict()}")

        for _, r in df.dropna(subset=[text_col, label_col]).iterrows():
            text = str(r[text_col]).strip()
            raw_label = str(r[label_col]).strip().lower()
            if not text:
                continue
            rows.append({
                "text": text,
                "language": "ur_roman",
                "script": detect_script(text),
                "hate_label": 1 if raw_label in _HATE_VALUES else 0,
                "target_label": -1,
                "severity_label": -1,
                "rationale_spans": "[]",
                "source": "ieee_razi",
                "split": hash_split(text),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "text", "language", "script", "hate_label", "target_label",
            "severity_label", "rationale_spans", "source", "split",
        ])
    return pd.DataFrame(rows).drop_duplicates(subset=["text"])


if __name__ == "__main__":
    df = convert()
    print(f"IEEE Razi: {df.shape}")
    if not df.empty:
        print(df["hate_label"].value_counts())
