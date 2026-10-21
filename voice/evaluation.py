"""
Measure the pretrained detector (and optionally the LLM judge + rewrite safety) on the
project's labelled splits (data/{val,test}.csv: text, language, hate_label, ...).

Use --split val while tuning; run --split test ONCE for the numbers you report.
"""

from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


def metrics(y: list[int], yhat: list[int]) -> dict:
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    n = len(y)

    def f1(p_tp, p_fp, p_fn):
        d = 2 * p_tp + p_fp + p_fn
        return 2 * p_tp / d if d else 0.0

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": n, "accuracy": round((tp + tn) / n, 4) if n else 0.0,
            "macro_f1": round((f1(tp, fp, fn) + f1(tn, fn, fp)) / 2, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def load_split(path: Path, per_lang: int, seed: int = 13) -> list[dict]:
    rows: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("hate_label") not in ("0", "1") or not (r.get("text") or "").strip():
                continue
            rows.setdefault(r["language"], []).append(r)
    rng = random.Random(seed)
    out = []
    for lang, rs in sorted(rows.items()):
        rng.shuffle(rs)
        # balanced sample where possible
        pos = [r for r in rs if r["hate_label"] == "1"][: per_lang // 2]
        neg = [r for r in rs if r["hate_label"] == "0"][: per_lang - len(pos)]
        out += pos + neg
    return out


def best_threshold(y: list[int], p: list[float]) -> float:
    best, bt = -1.0, 0.5
    for t in [i / 100 for i in range(5, 96)]:
        m = metrics(y, [int(x >= t) for x in p])["macro_f1"]
        if m > best:
            best, bt = m, t
    return bt


def run(reg, split: str, data_dir: Path, out_dir: Path, per_lang: int = 300, with_llm: bool = False,
        threshold: float | None = None, rewrite_sample: int = 40, progress=print) -> dict:
    t0 = time.time()
    s = reg.s
    rows = load_split(data_dir / f"{split}.csv", per_lang)
    texts = [r["text"] for r in rows]
    y = [int(r["hate_label"]) for r in rows]
    langs = [("ur" if r["language"] == "ur_roman" else r["language"]) for r in rows]
    progress(f"{len(rows)} examples from {split}; scoring with classifiers...")
    det = reg.detector()
    p: list[float] = []
    for i in range(0, len(texts), 64):
        p += det.score(texts[i:i + 64])
        progress(f"  classifiers {min(i + 64, len(texts))}/{len(texts)}")
    if threshold is None:
        threshold = best_threshold(y, p) if split == "val" else _saved_threshold(out_dir)
    yhat = [int(x >= threshold) for x in p]
    llm_model = None
    rewrite_stats = None
    if with_llm:
        llm = reg.llm()
        llm_model = s.llm_model
        unsafe, total, tox = 0, 0, []
        n_llm = 0
        for i, (t, lang) in enumerate(zip(texts, langs)):
            if p[i] < s.llm_gate and lang in s.clf_langs:
                yhat[i] = 0
                continue
            v = llm.judge(t, t, lang, "rewrite")
            n_llm += 1
            if v is None:
                continue  # keep classifier decision
            yhat[i] = int(v.severity != "none")
            if v.severity != "none" and v.rewrite and total < rewrite_sample:
                q = det.score([v.rewrite])[0]
                tox.append(q)
                total += 1
                unsafe += int(q >= s.safety_threshold)
            if n_llm % 20 == 0:
                progress(f"  LLM judged {n_llm} (row {i + 1}/{len(texts)})")
        if total:
            rewrite_stats = {"n": total, "safe_rate": round(1 - unsafe / total, 4),
                             "avg_hate_prob": round(sum(tox) / total, 4)}
    per = {}
    for lang in sorted({r["language"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        per[lang] = metrics([y[i] for i in idx], [yhat[i] for i in idx])
    rep = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split, "per_language_limit": per_lang, "threshold": threshold,
        "with_llm": with_llm, "llm_model": llm_model, "classifiers": list(getattr(det, "names", [])),
        "overall": metrics(y, yhat),
        "classifier_only": metrics(y, [int(x >= threshold) for x in p]),
        "per_language": per, "rewrite": rewrite_stats, "seconds": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{split}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out_dir / "eval_latest.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if split == "val":
        (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    return rep


def _saved_threshold(out_dir: Path) -> float:
    p = out_dir / "threshold.json"
    return json.loads(p.read_text())["threshold"] if p.exists() else 0.5
