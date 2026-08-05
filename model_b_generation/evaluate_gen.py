"""
Evaluation for Model B — counter-narrative generation quality.

Computes:
  1. BERTScore (semantic alignment with reference human responses) using
     bert-base-multilingual-cased as the scorer backbone (not English-only).
  2. Distinct-2 (lexical diversity across generated outputs).
  3. BLEU (secondary, logged but not optimized for — rigid n-gram overlap
     is explicitly the wrong metric for this task per the blueprint).

Results are written to checkpoints_gen/test_metrics_gen.json in the same
shape as Model A's test_metrics.json so both can be presented side by side.

Usage:
    python -m model_b_generation.evaluate_gen \\
        --test_csv data/test_gen.csv \\
        --checkpoint_dir checkpoints_gen \\
        --output_path checkpoints_gen/test_metrics_gen.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Ensure we can import from the package
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


def distinct_2(texts: list[str]) -> float:
    """Compute Distinct-2: ratio of unique bigrams to total bigrams."""
    if not texts:
        return 0.0
    unique_bigrams = set()
    total_bigrams = 0
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            unique_bigrams.add(bigram)
            total_bigrams += 1
    return len(unique_bigrams) / total_bigrams if total_bigrams > 0 else 0.0


def compute_bleu(candidates: list[str], references: list[str]) -> float:
    """Compute corpus BLEU using nltk as a fallback-safe approach."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[ref.split()] for ref in references]
        hyps = [cand.split() for cand in candidates]
        smoothie = SmoothingFunction().method1
        return corpus_bleu(refs, hyps, smoothing_function=smoothie)
    except ImportError:
        logger.warning("nltk not available — BLEU will be 0.0")
        return 0.0
    except Exception as e:
        logger.warning("BLEU computation failed: %s", e)
        return 0.0


def evaluate(
    test_csv: str,
    checkpoint_dir: str,
    output_path: str,
    device: str = "auto",
    batch_size: int = 16,
):
    """Run evaluation on test_gen.csv and write metrics."""
    from infer_gen import load_model_for_inference, generate_alternatives
    from bert_score import score as bert_score_fn

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(test_csv)
    logger.info("Loaded %d test examples from %s", len(df), test_csv)

    model, tokenizer = load_model_for_inference(checkpoint_dir, device)

    # Generate counter-narratives for each test example
    cands_by_lang = defaultdict(list)
    refs_by_lang = defaultdict(list)

    for idx, row in df.iterrows():
        hate_text = str(row["hate_text"])
        lang = str(row.get("language", "en"))
        ref = str(row["response_text"])

        cands = generate_alternatives(
            hate_text, lang, n=1, model=model, tokenizer=tokenizer
        )
        cand = cands[0] if cands else ""

        cands_by_lang[lang].append(cand)
        refs_by_lang[lang].append(ref)

        if (idx + 1) % 50 == 0:
            logger.info("  evaluated %d / %d", idx + 1, len(df))

    # ── Compute metrics per language ──
    metrics = {"n_evaluated": len(df), "by_language": {}}
    all_cands, all_refs = [], []

    for lang in sorted(cands_by_lang.keys()):
        c = cands_by_lang[lang]
        r = refs_by_lang[lang]
        all_cands.extend(c)
        all_refs.extend(r)

        # BERTScore with multilingual model
        P, R, F1 = bert_score_fn(
            c, r,
            model_type="bert-base-multilingual-cased",
            device=device,
            verbose=False,
        )

        d2 = distinct_2(c)
        bleu = compute_bleu(c, r)

        metrics["by_language"][lang] = {
            "bertscore_f1": F1.mean().item(),
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "distinct_2": d2,
            "bleu": bleu,
            "n": len(c),
        }
        logger.info(
            "  %s: BERTScore-F1=%.4f  Distinct-2=%.4f  BLEU=%.4f  (n=%d)",
            lang, F1.mean().item(), d2, bleu, len(c),
        )

    # ── Aggregate metrics ──
    if all_cands:
        P, R, F1 = bert_score_fn(
            all_cands, all_refs,
            model_type="bert-base-multilingual-cased",
            device=device,
            verbose=False,
        )
        metrics["bertscore_f1"] = F1.mean().item()
        metrics["bertscore_precision"] = P.mean().item()
        metrics["bertscore_recall"] = R.mean().item()
        metrics["distinct_2"] = distinct_2(all_cands)
        metrics["bleu"] = compute_bleu(all_cands, all_refs)

    # ── Write ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_path}")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Model B on held-out test set"
    )
    parser.add_argument("--test_csv", type=str, default="data/test_gen.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_gen")
    parser.add_argument("--output_path", type=str,
                        default="checkpoints_gen/test_metrics_gen.json")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    evaluate(args.test_csv, args.checkpoint_dir, args.output_path,
             args.device, args.batch_size)


if __name__ == "__main__":
    main()
