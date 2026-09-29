import json
import zipfile

import results_io


def test_write_and_zip_results(tmp_path):
    run = tmp_path / "run"
    results_io.write_json(run, "metrics.json", {"f1": 0.8, "path": tmp_path})
    results_io.append_jsonl(run, "metrics_history.jsonl", {"step": 1})
    results_io.append_jsonl(run, "metrics_history.jsonl", {"step": 2})
    results_io.write_config(run, {"lr": 1e-4})
    results_io.write_env(run)
    (run / "best_model.pt").write_bytes(b"weights")  # must not be zipped
    assert results_io.missing_files(run / "results") == []

    z = results_io.zip_results(run, tmp_path / "out.zip")
    names = zipfile.ZipFile(z).namelist()
    assert "results/metrics.json" in names and "best_model.pt" not in " ".join(names)
    hist = (run / "results" / "metrics_history.jsonl").read_text().splitlines()
    assert [json.loads(l)["step"] for l in hist] == [1, 2]


def test_collect_model_a(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_metrics.json").write_text(json.dumps({"hate_macro_f1": 0.9}))
    (run / "metrics_history.csv").write_text("step,hate_macro_f1\n1,0.8\n2,0.9\n")
    results_io.collect_model_a(run)
    res = run / "results"
    assert json.loads((res / "metrics.json").read_text())["hate_macro_f1"] == 0.9
    assert len((res / "metrics_history.jsonl").read_text().splitlines()) == 2
    assert results_io.missing_files(res) == []


def test_ingest_extracts_and_summarizes(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ingest_results", Path(__file__).resolve().parent.parent / "scripts" / "ingest_results.py")
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)

    run = tmp_path / "run"
    results_io.write_json(run, "metrics.json", {"chrf": 41.234567, "by_group": {"rewrite|ta": {"chrf": 30.0}}})
    results_io.write_config(run, {})
    results_io.write_env(run)
    z = results_io.zip_results(run, tmp_path / "civitas_results_model_b.zip")
    assert ingest.main([str(z), "--dest", str(tmp_path / "out")]) == 0
    res = tmp_path / "out" / "model_b" / "results"
    s = ingest.summarize(res)
    assert s["missing"] == [] and s["metrics.json"]["chrf"] == 41.2346
    assert s["metrics.json:by_group"]["rewrite|ta"]["chrf"] == 30.0


def test_cli_zips(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_metrics.json").write_text("{}")
    results_io.main(["--run_dir", str(run), "--kind", "model_a", "--zip", str(tmp_path / "r.zip")])
    assert (tmp_path / "r.zip").exists()
