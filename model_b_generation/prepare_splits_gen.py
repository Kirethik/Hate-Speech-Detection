"""
Step 2 of the Model B data build: pairs (+ optional reviewed silver
translations) -> leak-free train / val / test.

    python -m model_b_generation.prepare_splits_gen \
        [--pairs data/gen/pairs.parquet] \
        [--silver data/gen/silver_pairs.parquet --review data/gen/silver_review.jsonl] \
        [--out_dir data/gen]

Rules
  * Splits are decided per group_id (hash of the original English sentence),
    so a sentence, its other responses and all its translations share a split.
  * The converters put ~10% of groups in "val"; half of those groups (by a
    second, salted hash) become "test". Test is for the final report only.
  * Eval rows whose normalised source text also occurs in train are dropped.
  * Silver rows marked "drop" in the review file are removed, "fix" rows take
    the reviewer's text.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from model_b_generation.converters_gen import PAIR_COLUMNS

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
_HANDLE = re.compile(r"(?<!\w)@\w{2,}")


def strip_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", str(text))
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub("[PHONE]", text)
    return _HANDLE.sub("@user", text)


def norm_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.casefold().str.replace(_NON_WORD, "", regex=True)


def is_test_group(gid: str, test_frac: float = 0.5) -> bool:
    h = hashlib.md5(f"test-salt:{gid}".encode()).hexdigest()
    return (int(h[:8], 16) % 1000) < test_frac * 1000


def load_review(path) -> dict:
    """silver_id -> decision record (last decision wins)."""
    out = {}
    p = Path(path) if path else None
    if p and p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["silver_id"]] = rec
    return out


def apply_review(silver: pd.DataFrame, review: dict) -> pd.DataFrame:
    if not review or "silver_id" not in silver.columns:
        return silver
    silver = silver.copy()
    keep = []
    for i, sid in enumerate(silver["silver_id"]):
        rec = review.get(sid)
        if rec and rec.get("decision") == "drop":
            keep.append(False)
            continue
        if rec and rec.get("decision") == "fix":
            for col in ("source_text", "target_text"):
                if rec.get(col):
                    silver.iloc[i, silver.columns.get_loc(col)] = rec[col]
        keep.append(True)
    return silver[keep]


def prepare(pairs: pd.DataFrame, silver: pd.DataFrame | None = None,
            review: dict | None = None, test_frac: float = 0.5) -> dict[str, pd.DataFrame]:
    frames = [pairs[PAIR_COLUMNS]]
    if silver is not None and not silver.empty:
        frames.append(apply_review(silver, review or {})[PAIR_COLUMNS])
    df = pd.concat(frames, ignore_index=True)

    for col in ("source_text", "target_text"):
        df[col] = df[col].astype(str).map(strip_pii).str.strip()
    df = df[(df["source_text"].str.len() > 0) & (df["target_text"].str.len() > 0)]

    df["_src"] = norm_key(df["source_text"])
    df["_tgt"] = norm_key(df["target_text"])
    df = df[(df["_src"].str.len() > 0) & (df["_tgt"].str.len() > 0)]
    df = df.drop_duplicates(["task", "language", "_src", "_tgt"])

    held_out = df["split"] != "train"
    df.loc[held_out, "split"] = [
        "test" if is_test_group(g, test_frac) else "val" for g in df.loc[held_out, "group_id"]
    ]

    train = df[df["split"] == "train"]
    train_src = set(train["_src"])
    splits = {"train": train}
    for name in ("val", "test"):
        part = df[df["split"] == name]
        splits[name] = part[~part["_src"].isin(train_src)]

    g = {k: set(v["group_id"]) for k, v in splits.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"]), \
        "group leakage between splits"
    return {k: v[PAIR_COLUMNS].reset_index(drop=True) for k, v in splits.items()}


def summary(splits: dict[str, pd.DataFrame]) -> dict:
    return {name: {
        "rows": int(len(df)),
        "by_task_language": {f"{t}|{l}": int(n) for (t, l), n in
                             df.groupby(["task", "language"]).size().items()},
        "by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
    } for name, df in splits.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Leak-free Model B splits")
    p.add_argument("--pairs", default="data/gen/pairs.parquet")
    p.add_argument("--silver", default=None, help="silver_pairs.parquet from notebook 03a")
    p.add_argument("--review", default="data/gen/silver_review.jsonl")
    p.add_argument("--out_dir", default="data/gen")
    p.add_argument("--test_frac", type=float, default=0.5, help="share of held-out groups that become test")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pairs = pd.read_parquet(a.pairs)
    silver = pd.read_parquet(a.silver) if a.silver else None
    splits = prepare(pairs, silver, load_review(a.review) if a.silver else {}, a.test_frac)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"wrote {len(df):>7} rows -> {out / (name + '.parquet')}")
    rep = summary(splits)
    (out / "SPLITS_REPORT.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
