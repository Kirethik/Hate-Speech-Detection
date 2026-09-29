"""
Review machine-translated (silver) Model B pairs in the terminal.

    python -m model_b_generation.review_silver --silver data/gen/silver_pairs.parquet \
        --language ta [--n 100] [--review data/gen/silver_review.jsonl]

For each pair you see the English original and the translation, then press:
    k  keep      d  drop      f  fix (type corrected source/target)
    s  skip      q  quit (progress is saved after every answer)

Decisions are appended to the review file; prepare_splits_gen.py applies them
(drop removes the row, fix replaces its text). Rows already reviewed are not
shown again. At the end you get the keep/fix/drop rate for that language: if
more than ~30% needed fixing or dropping, tell Claude before training.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from model_b_generation.prepare_splits_gen import load_review


def pending(silver: pd.DataFrame, reviewed: dict, language: str | None, n: int,
            seed: int = 0) -> pd.DataFrame:
    df = silver if language is None else silver[silver["language"] == language]
    df = df[~df["silver_id"].isin(reviewed)]
    return df.sample(min(n, len(df)), random_state=seed) if len(df) else df


def record(path: Path, silver_id: str, decision: str, **fix) -> dict:
    rec = {"silver_id": silver_id, "decision": decision,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fix}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def rates(review: dict, silver: pd.DataFrame, language: str | None) -> dict:
    ids = set(silver["silver_id"] if language is None
              else silver.loc[silver["language"] == language, "silver_id"])
    dec = [r["decision"] for sid, r in review.items() if sid in ids]
    return {d: dec.count(d) for d in ("keep", "fix", "drop")} | {"reviewed": len(dec)}


def main(argv=None, input_fn=input):
    p = argparse.ArgumentParser(description="keep / fix / drop silver translations")
    p.add_argument("--silver", default="data/gen/silver_pairs.parquet")
    p.add_argument("--review", default="data/gen/silver_review.jsonl")
    p.add_argument("--language", default=None, help="hi | ta | te | ml | ur_roman (default: all)")
    p.add_argument("--n", type=int, default=100)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    silver = pd.read_parquet(a.silver)
    path = Path(a.review)
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = pending(silver, load_review(path), a.language, a.n)
    print(f"{len(todo)} pairs to review -> decisions saved to {path}\n")
    for i, r in enumerate(todo.itertuples(index=False), 1):
        print(f"--- {i}/{len(todo)}  [{r.task} | {r.language} | from {r.origin_source}]")
        print(f"  EN input : {r.en_source_text}")
        print(f"  EN output: {r.en_target_text}")
        print(f"  -> input : {r.source_text}")
        print(f"  -> output: {r.target_text}")
        while True:
            c = input_fn("  [k]eep [d]rop [f]ix [s]kip [q]uit > ").strip().lower()[:1]
            if c in ("k", "d", "s", "q", "f"):
                break
        if c == "q":
            break
        if c == "k":
            record(path, r.silver_id, "keep")
        elif c == "d":
            record(path, r.silver_id, "drop")
        elif c == "f":
            src = input_fn("  corrected input  (Enter = unchanged): ").strip()
            tgt = input_fn("  corrected output (Enter = unchanged): ").strip()
            record(path, r.silver_id, "fix", **{k: v for k, v in
                                                (("source_text", src), ("target_text", tgt)) if v})
    print("\nreview so far:", rates(load_review(path), silver, a.language))


if __name__ == "__main__":
    main()
