"""
Final Model B report on the TEST split. Run once, after tuning is finished
(notebook 03, "RUN ONCE" cell). Writes <run_dir>/results/test_report.json
and results/test_samples.jsonl.

    python -m model_b_generation.evaluate_gen --data_dir /content/gen_data \
        --run_dir /content/drive/MyDrive/civitas/model_b_v1 [--model_a_ckpt best_model.pt]
"""

import argparse
import sys
from pathlib import Path

import results_io
from model_b_generation import gen_metrics
from model_b_generation.dataset_gen import read_pairs
from model_b_generation.model_gen import DEFAULT_BASE, load_for_inference
from model_b_generation.prompts import build_prompt
from model_b_generation.train_gen import generate, model_a_scorer


def main(argv=None):
    p = argparse.ArgumentParser(description="Model B test report (RUN ONCE)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--run_dir", required=True, help="train_gen output_dir (uses run_dir/best)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--model_a_ckpt", default="")
    p.add_argument("--max_output_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = Path(a.data_dir)
    path = next(q for q in (data / f"{a.split}.parquet", data / f"{a.split}.csv") if q.exists())
    df = read_pairs(path)
    mb = load_for_inference(str(Path(a.run_dir) / "best"), a.base, merge=True)
    prompts = [build_prompt(r.task, r.language, r.source_text, r.target) for r in df.itertuples()]
    outs = generate(mb.model, mb.tokenizer, prompts, mb.device, a.max_output_length, a.batch_size)
    score_fn, thr = model_a_scorer(a.model_a_ckpt, mb.device)
    groups = (df["task"] + "|" + df["language"]).tolist()
    rep = gen_metrics.report(outs, df["target_text"].tolist(), df["source_text"].tolist(),
                             groups, score_fn, thr)
    rep.update(split=a.split, n_rows=len(df), adapter=str(Path(a.run_dir) / "best"))
    if "source" in df.columns:  # silver (machine-translated) references vs human ones
        is_silver = df["source"].eq("silver_mt").tolist()
        rep["silver_vs_gold_chrf"] = {
            "silver": gen_metrics.corpus_chrf([o for o, s in zip(outs, is_silver) if s],
                                              [r for r, s in zip(df["target_text"], is_silver) if s]),
            "gold": gen_metrics.corpus_chrf([o for o, s in zip(outs, is_silver) if not s],
                                            [r for r, s in zip(df["target_text"], is_silver) if not s]),
        }
    name = "test_report.json" if a.split == "test" else "val_report.json"
    results_io.write_json(a.run_dir, name, rep)
    results_io.write_jsonl(a.run_dir, f"{a.split}_samples.jsonl", [
        {"group": g, "prompt": pr, "output": o, "reference": r}
        for g, pr, o, r in list(zip(groups, prompts, outs, df["target_text"]))[:300]])
    print({k: rep[k] for k in ("n", "chrf", "bleu", "copy_rate", "safety_rate")})
    return rep


if __name__ == "__main__":
    main()
