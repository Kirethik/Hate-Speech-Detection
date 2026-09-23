"""
ParaDetox + TextDetox 2024 multilingual converter for Model B.

ParaDetox: https://huggingface.co/datasets/s-nlp/paradetox
  English toxic → non-toxic parallel rewrites. Core of "Rewrite" task.

TextDetox 2024: https://huggingface.co/datasets/textdetox/multilingual_paradetox
  Multilingual (includes Hindi). Same format.
"""

import hashlib
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def _hash_split(text: str, val_frac: float = 0.10) -> str:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return "val" if (int(digest[:8], 16) % 1000) < val_frac * 1000 else "train"


_LANG_MAP = {
    "en": "en", "english": "en",
    "hi": "hi", "hindi": "hi",
    "ru": None,  # Russian — not in our target language set, skip
    "de": None,
    "zh": None,
}


def _rows_from_hf(ds, task: str, language: str, source: str) -> list[dict]:
    rows = []
    df = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(ds)

    # Detect column names
    col = {c.lower().strip(): c for c in df.columns}
    toxic_col = next((col[k] for k in ("toxic_sentence", "toxic", "input", "source") if k in col), None)
    neutral_col = next((col[k] for k in ("neutral_sentence", "neutral", "target", "reference") if k in col), None)

    if toxic_col is None or neutral_col is None:
        print(f"  [paradetox] WARNING: columns not found. Available: {list(df.columns)}")
        return rows

    for _, r in df.dropna(subset=[toxic_col, neutral_col]).iterrows():
        src = str(r[toxic_col]).strip()
        tgt = str(r[neutral_col]).strip()
        if not src or not tgt:
            continue
        rows.append({
            "task": task,
            "source_text": src,
            "target_text": tgt,
            "language": language,
            "source": source,
            "split": _hash_split(src),
        })
    return rows


def convert(raw_dir: str = "raw_data") -> pd.DataFrame:
    all_rows: list[dict] = []

    if not _HF_AVAILABLE:
        print("  [paradetox] HuggingFace datasets not installed — skipping.")
        return pd.DataFrame(columns=["task", "source_text", "target_text", "language", "source", "split"])

    # --- ParaDetox (English) ---
    try:
        ds = load_dataset("s-nlp/paradetox", split="train")
        all_rows.extend(_rows_from_hf(ds, task="rewrite", language="en", source="paradetox"))
        print(f"  [paradetox] paradetox EN: {len(all_rows)} rows loaded")
    except Exception as e:
        print(f"  [paradetox] WARNING: could not load s-nlp/paradetox: {e}")

    # --- TextDetox 2024 (multilingual) ---
    try:
        multi_ds = load_dataset("textdetox/multilingual_paradetox")
        for lang_key, lang_code in _LANG_MAP.items():
            if lang_code is None:
                continue
            split_name = "train" if "train" in multi_ds else list(multi_ds.keys())[0]
            split_ds = multi_ds[split_name]
            # Filter by language if there's a language column
            df = split_ds.to_pandas()
            lang_col = next((c for c in df.columns if c.lower() in ("lang", "language")), None)
            if lang_col:
                df = df[df[lang_col].str.lower().str.startswith(lang_key)]
            if df.empty:
                continue
            before = len(all_rows)
            # Re-run as rows
            col = {c.lower().strip(): c for c in df.columns}
            toxic_col = next((col[k] for k in ("toxic_sentence", "toxic", "input") if k in col), None)
            neutral_col = next((col[k] for k in ("neutral_sentence", "neutral", "target") if k in col), None)
            if toxic_col and neutral_col:
                for _, r in df.dropna(subset=[toxic_col, neutral_col]).iterrows():
                    src, tgt = str(r[toxic_col]).strip(), str(r[neutral_col]).strip()
                    if src and tgt:
                        all_rows.append({
                            "task": "rewrite",
                            "source_text": src,
                            "target_text": tgt,
                            "language": lang_code,
                            "source": "textdetox2024",
                            "split": _hash_split(src),
                        })
            print(f"  [textdetox] {lang_key}: {len(all_rows) - before} rows")
    except Exception as e:
        print(f"  [textdetox] WARNING: could not load textdetox/multilingual_paradetox: {e}")

    if not all_rows:
        return pd.DataFrame(columns=["task", "source_text", "target_text", "language", "source", "split"])
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = convert()
    print(f"ParaDetox/TextDetox: {df.shape}")
    if not df.empty:
        print(df["source"].value_counts())
        print(df["language"].value_counts())
