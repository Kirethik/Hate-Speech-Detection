"""
Evaluate the pretrained detector on the project's labelled data and show it in the
Model Evaluation tab.

    python scripts/evaluate.py --split val --per-language 300              # classifiers only (minutes)
    python scripts/evaluate.py --split val --per-language 300 --with-llm   # + LLM judge (slower)
    python scripts/evaluate.py --split test --per-language 500 --with-llm  # final numbers, run once

On val the classifier threshold is tuned for macro-F1 and saved; test reuses it.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.evaluation import run  # noqa: E402
from voice.registry import Registry  # noqa: E402
from voice.settings import get_settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--per-language", type=int, default=300)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    a = ap.parse_args()
    s = get_settings()
    rep = run(Registry(s), a.split, Path(a.data_dir), Path(s.data_dir), a.per_language, a.with_llm, a.threshold)
    o = rep["overall"]
    print(f"\n{rep['split']}: n={o['n']} acc={o['accuracy']:.3f} macroF1={o['macro_f1']:.3f} "
          f"recall={o['recall']:.3f} FPR={o['fpr']:.3f} (threshold {rep['threshold']:.2f})")
    for k, m in rep["per_language"].items():
        print(f"  {k:9s} n={m['n']:4d} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} recall={m['recall']:.3f}")
    if rep["rewrite"]:
        print(f"  rewrites passing safety gate: {rep['rewrite']['safe_rate']:.1%} of {rep['rewrite']['n']}")
    print("Saved; open the Model Evaluation tab.")


if __name__ == "__main__":
    main()
