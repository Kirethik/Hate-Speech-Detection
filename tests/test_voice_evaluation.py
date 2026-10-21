import csv
import json

from tests.voice_fakes import FakeDetector, FakeLlm, fake_registry
from voice.evaluation import best_threshold, load_split, metrics, run


def test_metrics():
    m = metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["macro_f1"] == 0.5 and m["fpr"] == 0.5


def test_best_threshold_separates():
    assert 0.3 < best_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9]) <= 0.7


def _write(tmp_path):
    rows = [("they are vermin", "en", 1), ("nice weather", "en", 0), ("a plague on us", "ta", 1),
            ("good food", "ta", 0), ("hello", "ur_roman", 0), ("bad", "en", "")]
    with open(tmp_path / "val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "language", "hate_label", "target_label", "severity_label", "rationale_spans", "source", "script"])
        for t, l, y in rows:
            w.writerow([t, l, y, -1, -1, "[]", "x", "latin"])


def test_run_with_llm(tmp_path):
    _write(tmp_path)
    assert len(load_split(tmp_path / "val.csv", 10)) == 5  # unlabeled row dropped
    reg = fake_registry(tmp_path)
    rep = run(reg, "val", tmp_path, tmp_path / "out", per_lang=10, with_llm=True, progress=lambda *a: None)
    assert rep["overall"]["accuracy"] == 1.0 and rep["overall"]["n"] == 5
    assert set(rep["per_language"]) == {"en", "ta", "ur_roman"}
    assert rep["rewrite"]["n"] == 2 and rep["rewrite"]["safe_rate"] == 1.0
    saved = json.loads((tmp_path / "out" / "eval_latest.json").read_text())
    assert saved["split"] == "val" and (tmp_path / "out" / "threshold.json").exists()
