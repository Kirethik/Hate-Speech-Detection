"""
HateCheck functional evaluation (Röttger et al., ACL 2021).

WHY THIS AND NOT JUST MACRO-F1
------------------------------
Macro-F1 on a held-out split says how well the model fits the same distribution
it trained on. It cannot say *which capability* is missing, because every error
is worth the same one point. HateCheck is a suite of hand-built FUNCTIONAL
tests: ~3.7k cases grouped into ~29 functionalities, each isolating one
behaviour. A model can score 0.85 macro-F1 and still fail 100% of
"non-hateful use of a slur" or "counter-speech quoting hate" — and those two
failures are what make a moderation model unusable in practice.

The functionalities suffixed `_h` expect HATEFUL, `_nh` expect NON-HATEFUL. The
`_nh` families are the interesting ones here: they are precisely the
false-positive traps (identity mentions, reclaimed slurs, counter-speech,
negated hate) that a profanity-keyed model walks into.

Read the per-functionality table, not the headline. That table is the artefact
worth putting in the report.

Usage:
    python -m eval.eval_hatecheck --checkpoint artifacts/model_a/best_model.pt
    python -m eval.eval_hatecheck --compare artifacts/model_a/best_model_baseline.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import XLMRobertaTokenizerFast

from infer import load_model, score_batch

REPO, FILENAME = "Paul/hatecheck", "test.csv"


def evaluate(path, df, device):
    m = load_model(path, device)
    threshold = m.threshold
    probs = np.array([r["hate_prob"] for r in score_batch(m, df["test_case"].tolist(), 64)])
    pred = probs >= threshold
    gold = (df["label_gold"].astype(str).str.strip().str.casefold() == "hateful").to_numpy()
    return pred, gold, threshold, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/model_a/best_model.pt")
    ap.add_argument("--compare", default=None,
                    help="second checkpoint to diff against (e.g. the baseline)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="use cpu when a training run owns the GPU")
    ap.add_argument("--out", default="artifacts/model_a/hatecheck.json")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )

    df = pd.read_csv(hf_hub_download(REPO, FILENAME, repo_type="dataset"))
    df = df.dropna(subset=["test_case", "label_gold", "functionality"])
    print(f"HateCheck: {len(df)} cases across {df['functionality'].nunique()} functionalities "
          f"| device={device}\n")

    pred, gold, threshold, _ = evaluate(args.checkpoint, df, device)
    df["correct"] = pred == gold
    df["pred_hateful"] = pred

    base_correct = None
    if args.compare and Path(args.compare).exists():
        base_pred, _, _, _ = evaluate(args.compare, df, device)
        base_correct = base_pred == gold
        df["baseline_correct"] = base_correct

    rows = []
    for func, g in df.groupby("functionality"):
        expects_hateful = str(g["label_gold"].iloc[0]).strip().casefold() == "hateful"
        entry = {
            "functionality": func,
            "expects": "hateful" if expects_hateful else "non-hateful",
            "n": len(g),
            "accuracy": float(g["correct"].mean()),
        }
        if base_correct is not None:
            entry["baseline_accuracy"] = float(g["baseline_correct"].mean())
            entry["delta"] = entry["accuracy"] - entry["baseline_accuracy"]
        rows.append(entry)

    rows.sort(key=lambda r: (r["expects"], r["accuracy"]))
    header = f"{'functionality':<32} {'expects':<12} {'n':>5} {'acc':>7}"
    if base_correct is not None:
        header += f" {'base':>7} {'delta':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['functionality']:<32} {r['expects']:<12} {r['n']:>5} {r['accuracy']:>7.1%}"
        if base_correct is not None:
            line += f" {r['baseline_accuracy']:>7.1%} {r['delta']:>+7.1%}"
        print(line)

    overall = float(df["correct"].mean())
    hateful = df[df["label_gold"].astype(str).str.strip().str.casefold() == "hateful"]
    non_hateful = df[df["label_gold"].astype(str).str.strip().str.casefold() != "hateful"]
    print(f"\noverall {overall:.1%}   "
          f"hateful cases {hateful['correct'].mean():.1%}   "
          f"non-hateful cases {non_hateful['correct'].mean():.1%}")
    if base_correct is not None:
        print(f"baseline {df['baseline_correct'].mean():.1%}   "
              f"hateful {hateful['baseline_correct'].mean():.1%}   "
              f"non-hateful {non_hateful['baseline_correct'].mean():.1%}")
    print("\nThe non-hateful number is the false-positive rate in disguise: a model "
          "that flags everything scores 100% on hateful and 0% here.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "threshold": threshold, "overall": overall,
        "hateful_accuracy": float(hateful["correct"].mean()),
        "non_hateful_accuracy": float(non_hateful["correct"].mean()),
        "per_functionality": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
