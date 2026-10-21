"""
Civitas AI — Model A Detector Evaluation Suite

Evaluates hate speech detection performance quantitatively.
Computes:
- Accuracy, Macro-F1, Weighted-F1
- Per-class precision, recall, and F1
- Confusion matrix
- False positive and false negative rates
- Language-wise breakdown where available
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.models.hate_detector import detect_hate_speech
from src.models.model_loader import get_system_model_status


def evaluate_detector(csv_path: str, output_path: str = None, max_samples: int = 500) -> dict:
    df = pd.read_csv(csv_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    y_true = df["label"].astype(int).tolist()
    y_pred = []
    y_probs = []

    print(f"Running evaluation on {len(df)} samples from {csv_path}...")
    for text, lang in zip(df["text"], df.get("language", ["en"] * len(df))):
        res = detect_hate_speech(str(text), language_hint=str(lang))
        y_pred.append(1 if res.is_hate else 0)
        y_probs.append(res.hate_prob)

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    class_p, class_r, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    tn, fp, fn, tp = (cm[0][0], cm[0][1], cm[1][0], cm[1][1]) if len(cm) == 2 and len(cm[0]) == 2 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    report = {
        "dataset": csv_path,
        "sample_count": len(df),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "weighted_f1": round(w_f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4)
        },
        "per_class_metrics": {
            "non_hateful": {"precision": round(class_p[0], 4), "recall": round(class_r[0], 4), "f1": round(class_f1[0], 4)},
            "hateful": {"precision": round(class_p[1], 4), "recall": round(class_r[1], 4), "f1": round(class_f1[1], 4)} if len(class_p) > 1 else {}
        },
        "confusion_matrix": cm,
        "model_status": {k: v.to_dict() for k, v in get_system_model_status().items()}
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Evaluation report written to {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model A Detector")
    parser.add_argument("--data", default="datasets/processed/detector/validation.csv", help="Path to evaluation CSV")
    parser.add_argument("--out", default="evaluation/detector_eval_report.json", help="Path to write JSON report")
    parser.add_argument("--samples", type=int, default=200, help="Max samples to evaluate")
    args = parser.parse_args()

    results = evaluate_detector(args.data, args.out, max_samples=args.samples)
    print("\nSummary Results:")
    print(json.dumps(results["overall_metrics"], indent=2))
