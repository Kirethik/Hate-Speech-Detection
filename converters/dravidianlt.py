"""
DravidianLangTech shared task converter.
Covers the hate/offensive tracks for Tamil, Malayalam, and Telugu.

Expected directory layout:
    data/raw/dravidianlt/
        tamil/
            tamil_offensive_train.tsv    (text \\t label)
            tamil_offensive_dev.tsv
        malayalam/
            malayalam_offensive_train.tsv
            malayalam_offensive_dev.tsv
        telugu/
            telugu_hate_train.tsv
            telugu_hate_dev.tsv

Column format: tab-separated, no header row — col0=text, col1=label.
(Some files have a header; the converter detects this automatically.)

Label sets vary slightly by language/year; all non-"Not_offensive" and
non-"none" labels map to hate_label=1. "not-<language>" rows (annotator judged
the text is not in that language) are not a hate judgement and are DROPPED,
matching converters/dravidiancodemix.py. target_label and severity_label
are left -1 (DravidianLangTech does not provide these).
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Labels that map to hate_label=0 (not offensive)
_NOT_OFFENSIVE = frozenset({"not_offensive", "none", "not", "0", "non-hate", "non_hate"})


_LANG_DIRS = {
    "ta": "tamil",
    "ml": "malayalam",
    "te": "telugu",
}

_FILE_PATTERNS = [
    ("train", "train"),
    ("dev", "val"),
    ("test", "val"),
]


def _load_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path, sep="\t", header=None, quoting=3,
                          on_bad_lines="skip", engine="python",
                          names=["text", "label"])
        # Detect and drop header rows
        if raw.iloc[0]["label"] in ("label", "Label", "LABEL"):
            raw = raw.iloc[1:]
        return raw.dropna(subset=["text", "label"])
    except Exception:
        return None


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    root = Path(raw_dir) / "dravidianlt"
    rows = []

    for lang_code, lang_dir_name in _LANG_DIRS.items():
        lang_dir = root / lang_dir_name
        if not lang_dir.exists():
            continue

        for split_keyword, our_split in _FILE_PATTERNS:
            # Try to find any matching file
            candidates = list(lang_dir.glob(f"*{split_keyword}*.tsv")) + \
                         list(lang_dir.glob(f"*{split_keyword}*.csv"))
            for cand in candidates:
                df = _load_file(cand)
                if df is None:
                    continue
                for _, r in df.iterrows():
                    text = str(r["text"]).strip()
                    label = str(r["label"]).strip().casefold()
                    if not text or label.startswith("not-"):
                        continue
                    hate = 0 if label in _NOT_OFFENSIVE else 1
                    rows.append({
                        "text": text,
                        "language": lang_code,
                        "hate_label": hate,
                        "target_label": -1,
                        "severity_label": -1,
                        "rationale_spans": "[]",
                        "source": "dravidianlt",
                        "split": our_split,
                    })
                break  # one file per split per language

    if not rows:
        return pd.DataFrame(columns=[
            "text", "language", "hate_label", "target_label",
            "severity_label", "rationale_spans", "source", "split",
        ])
    return pd.DataFrame(rows).drop_duplicates(subset=["text", "language"])


if __name__ == "__main__":
    df = convert()
    print(f"DravidianLangTech: {df.shape}")
    if not df.empty:
        print(df["language"].value_counts())
        print(df["hate_label"].value_counts())
