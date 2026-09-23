"""
HASOC 2019 / 2020 / 2021 converter.

Expected directory layout:
    data/raw/hasoc/
        2019/
            english_dataset/
                hasoc2019_en_train-2919.tsv   (tab-sep; cols: text_id, text, task_1, task_2)
                hasoc2019_en_test-post.tsv
            hindi_dataset/
                hindi_hatespeech_train.tsv
                hindi_hatespeech_test.tsv
        2020/
            hasoc2020_en_train.tsv
            hasoc2020_en_test.tsv
            hasoc2020_hi_train.tsv
            hasoc2020_hi_test.tsv
        2021/
            hasoc2021_en_train.tsv   (has rationale_spans column for EN)
            hasoc2021_en_test.tsv
            hasoc2021_hi_train.tsv

task_1 labels: HOF (hate-or-offensive), NOT
task_2 labels: HATE, OFFN, PRFN, NONE  (only present when task_1==HOF)

Label decisions (confirmed by user):
  - HOF with no task_2 → offensive_profanity (safe fallback)
  - HATE → hate
  - OFFN → offensive_profanity
  - PRFN → offensive_profanity
  - NOT / NONE → normal, hate_label=0
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from label_maps import (  # noqa: E402
    SEVERITY_CLASSES,
    TARGET_CLASSES,
    HASOC_SEVERITY_MAP,
)
from dataset import detect_script  # noqa: E402


_TASK1_HOF = "HOF"
_TASK1_NOT = "NOT"


def _load_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip", engine="python")
        return df
    except Exception:
        return None


def _hate_from_task1(t1: str) -> int:
    return 0 if str(t1).strip().upper() == _TASK1_NOT else 1


def _severity_from_tasks(t1: str, t2) -> int:
    t1 = str(t1).strip().upper()
    t2 = str(t2).strip().upper() if pd.notna(t2) else ""
    raw = HASOC_SEVERITY_MAP.get(t2) or HASOC_SEVERITY_MAP.get(t1)
    if raw is None:
        raw = "normal" if t1 == _TASK1_NOT else "offensive_profanity"
    return SEVERITY_CLASSES.index(raw)


def _parse_rationale(raw) -> str:
    """HASOC 2021 provides rationale as token indices or span strings."""
    if pd.isna(raw) or str(raw).strip() in ("", "[]", "nan"):
        return "[]"
    # If it's already a JSON list, validate and pass through
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, list):
            return json.dumps(parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    return "[]"


def _rows_from_df(df: pd.DataFrame, language: str, split: str, year: int) -> list[dict]:
    rows = []
    # Normalise column names (different years have slightly different headers)
    col = {c.lower().strip(): c for c in df.columns}

    text_col = next((col[k] for k in ("text", "post_text", "tweet_text") if k in col), None)
    t1_col = next((col[k] for k in ("task_1", "task1", "label_1") if k in col), None)
    t2_col = next((col[k] for k in ("task_2", "task2", "label_2") if k in col), None)
    rat_col = next((col[k] for k in ("rationale", "rationale_spans") if k in col), None)

    if text_col is None or t1_col is None:
        return rows

    for _, r in df.dropna(subset=[text_col, t1_col]).iterrows():
        text = str(r[text_col]).strip()
        if not text:
            continue
        t1 = r[t1_col]
        t2 = r[t2_col] if t2_col else None
        rationale = _parse_rationale(r[rat_col]) if rat_col else "[]"

        rows.append({
            "text": text,
            "language": language,
            "script": detect_script(text),
            "hate_label": _hate_from_task1(t1),
            "target_label": -1,
            "severity_label": _severity_from_tasks(t1, t2),
            "rationale_spans": rationale,
            "source": f"hasoc{year}",
            "split": split,
        })
    return rows


def _load_year(root: Path, year: int) -> list[dict]:
    rows = []
    year_dir = root / str(year)
    if not year_dir.exists():
        return rows

    # Try to discover files dynamically
    for lang_code, lang_key in [("en", "english"), ("hi", "hindi")]:
        for filesplit, our_split in [("train", "train"), ("test", "val")]:
            # Try multiple naming patterns seen across HASOC years
            candidates = list(year_dir.rglob(f"*{lang_key}*{filesplit}*.tsv")) + \
                         list(year_dir.rglob(f"*{filesplit}*{lang_key}*.tsv")) + \
                         list(year_dir.rglob(f"*{lang_code}*{filesplit}*.tsv"))
            df = None
            for cand in candidates:
                df = _load_tsv(cand)
                if df is not None and not df.empty:
                    break
            if df is None:
                continue
            rows.extend(_rows_from_df(df, lang_code, our_split, year))
    return rows


def convert(raw_dir="raw_data") -> pd.DataFrame:
    root = Path(raw_dir) / "hasoc"
    all_rows: list[dict] = []

    for year in (2019, 2020, 2021):
        year_rows = _load_year(root, year)
        all_rows.extend(year_rows)

    if not all_rows:
        return pd.DataFrame(columns=[
            "text", "language", "script", "hate_label", "target_label",
            "severity_label", "rationale_spans", "source", "split",
        ])

    df = pd.DataFrame(all_rows)
    # Deduplicate across years (same tweet may appear in 2019 and 2020)
    df = df.drop_duplicates(subset=["text", "language"])
    return df


if __name__ == "__main__":
    df = convert()
    print(f"HASOC: {df.shape}")
    if not df.empty:
        print(df["source"].value_counts())
        print(df["language"].value_counts())
        print(df["severity_label"].value_counts())
