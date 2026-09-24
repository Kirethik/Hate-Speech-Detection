"""
Fit the NLI stage's decision threshold on val, instead of shipping a guess.

The placeholder 0.70 in nli_stage.py was wrong: measured entailment on the probe
cases separates at 0.50-0.56 (hate) vs 0.003-0.15 (benign), so 0.70 would fire
almost never. This fits the threshold the same way train.py fits the detector's
own — on val, applied unchanged thereafter.

TARGET CHOICE. Calibrating against `hate_label` would be wrong: that label is
offensive-or-worse and includes personal abuse and profanity, which the
hypotheses deliberately do NOT describe (they are about GROUP-directed
dehumanisation). Calibrating against it would push the threshold down to chase
examples the stage was never meant to catch. So this fits against
`severity_label`: hate (2) as positive, normal (0) as negative, skipping
offensive_profanity (1) as the deliberately-ambiguous middle.

Runs on CPU by default so it can share a machine with a training run.

Usage:
    python -m eval.calibrate_nli --n 400 --device cpu
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nli_stage import NLIStage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--n", type=int, default=400, help="rows per class")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="checkpoints/model_a/nli_calibration.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.val_csv)
    # English only: the hypotheses are English and mDeBERTa's NLI training is
    # English-dominated, so a threshold fitted on Romanised Indic text would be
    # fitting noise. The stage is English-first and this makes that explicit.
    df = df[df["language"] == "en"]

    pos = df[df["severity_label"] == 2]
    neg = df[df["severity_label"] == 0]
    n = min(args.n, len(pos), len(neg))
    if n < 20:
        raise SystemExit(f"not enough English severity-labelled val rows (pos={len(pos)}, neg={len(neg)})")
    pos = pos.sample(n=n, random_state=args.seed)
    neg = neg.sample(n=n, random_state=args.seed)

    texts = pos["text"].astype(str).tolist() + neg["text"].astype(str).tolist()
    labels = np.array([1] * n + [0] * n)
    print(f"calibrating on {len(texts)} English val rows ({n} hate / {n} normal) on {args.device}")

    nli = NLIStage(device=torch.device(args.device))
    scores = []
    for i, t in enumerate(texts):
        scores.append(nli.max_entailment(t)[0])
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(texts)}", flush=True)
    scores = np.array(scores)

    best = {"threshold": 0.5, "f1": -1.0}
    for th in np.arange(0.05, 0.96, 0.01):
        pred = scores >= th
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"threshold": round(float(th), 2), "f1": float(f1),
                    "precision": float(precision), "recall": float(recall)}

    result = {
        **best,
        "n_per_class": n,
        "mean_entailment_hate": float(scores[labels == 1].mean()),
        "mean_entailment_normal": float(scores[labels == 0].mean()),
        "note": "fitted on English val rows, severity hate(2) vs normal(0)",
    }
    print(f"\nfitted threshold {result['threshold']:.2f}  "
          f"F1={result['f1']:.3f} P={result['precision']:.3f} R={result['recall']:.3f}")
    print(f"mean entailment: hate {result['mean_entailment_hate']:.3f} "
          f"vs normal {result['mean_entailment_normal']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
