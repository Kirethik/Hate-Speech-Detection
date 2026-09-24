"""
Qian et al. Gab & Reddit counter-speech dataset converter.

Paper: "Benchmark Dataset for Automatic Detection of Online Misogyny" → actually
this is the dataset from:
  Qian et al. (2019) "Benchmark Dataset for Automatic Counter-Narrative Generation"
  GitHub: https://github.com/ziqizhang/iac_counter_speech

Expected layout:
    data/raw/qian_counter/
        gab_dataset_sample_67K.csv    (columns: hate_speech, counter_speech, ...)
        reddit_dataset_sample_52K.csv

Maps to the alternate-speech unified schema:
    {task, source_text, target_text, language, source, split}
task = "respond"  (these are counter-narratives, not rewrites)
"""

import hashlib
from pathlib import Path

import pandas as pd


def _hash_split(text: str, val_frac: float = 0.10) -> str:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return "val" if (int(digest[:8], 16) % 1000) < val_frac * 1000 else "train"


def _load_qian_file(path: Path, source_name: str) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    try:
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except Exception:
        return rows

    # Find hate and counter columns
    col = {c.lower().strip(): c for c in df.columns}
    hate_col = next((col[k] for k in ("hate_speech", "hatespeech", "hate", "post") if k in col), None)
    counter_col = next((col[k] for k in ("counter_speech", "counter", "response", "reply") if k in col), None)

    if hate_col is None or counter_col is None:
        print(f"  [qian] WARNING: could not identify columns in {path.name}. "
              f"Columns: {list(df.columns)}")
        return rows

    for _, r in df.dropna(subset=[hate_col, counter_col]).iterrows():
        src = str(r[hate_col]).strip()
        tgt = str(r[counter_col]).strip()
        if not src or not tgt:
            continue
        rows.append({
            "task": "respond",
            "source_text": src,
            "target_text": tgt,
            "language": "en",
            "source": source_name,
            "split": _hash_split(src),
        })
    return rows


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    root = Path(raw_dir) / "qian_counter"
    all_rows = []

    for filename, src_name in [
        ("gab_dataset_sample_67K.csv", "qian_gab"),
        ("reddit_dataset_sample_52K.csv", "qian_reddit"),
    ]:
        all_rows.extend(_load_qian_file(root / filename, src_name))

    if not all_rows:
        return pd.DataFrame(columns=["task", "source_text", "target_text", "language", "source", "split"])
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = convert()
    print(f"Qian counter-speech: {df.shape}")
    if not df.empty:
        print(df["source"].value_counts())
