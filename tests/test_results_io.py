import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

import results_io

REPO = Path(__file__).resolve().parent.parent


def _install_models():
    spec = importlib.util.spec_from_file_location("install_models", REPO / "scripts" / "install_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _fake_model_a_run(run: Path):
    torch = pytest.importorskip("torch")
    run.mkdir(parents=True)
    torch.save({"model_state_dict": {"w": torch.ones(4, 4), "step": torch.tensor([3])},
                "threshold": 0.42, "args": {"encoder_name": "x"}, "optimizer": {"huge": 1}},
               run / "best_model.pt")
    (run / "best_metrics.json").write_text(json.dumps({"hate_macro_f1": 0.9, "threshold": 0.42}))
    return torch


def test_trained_zip_model_a_is_fp16_and_repo_relative(tmp_path):
    run = tmp_path / "run"
    torch = _fake_model_a_run(run)
    results_io.main(["--run_dir", str(run), "--kind", "model_a"])
    z = run / results_io.TRAINED_ZIP["model_a"]
    names = zipfile.ZipFile(z).namelist()
    assert all(n.startswith("artifacts/") for n in names)
    assert {"artifacts/model_a/best_model.pt", "artifacts/model_a/MODEL_INFO.json",
            "artifacts/results/model_a/results/metrics.json"} <= set(names)
    zipfile.ZipFile(z).extractall(tmp_path / "x")
    ck = torch.load(tmp_path / "x/artifacts/model_a/best_model.pt", weights_only=False)
    assert ck["model_state_dict"]["w"].dtype == torch.float16
    assert ck["model_state_dict"]["step"].dtype == torch.int64
    assert "optimizer" not in ck and ck["threshold"] == 0.42
    info = json.loads((tmp_path / "x/artifacts/model_a/MODEL_INFO.json").read_text())
    assert info["headline_val_metrics"]["hate_macro_f1"] == 0.9


def test_trained_zip_model_b_and_install(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "best").mkdir(parents=True)
    (run / "best" / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "bigscience/mt0-base"}))
    (run / "best" / "adapter_model.safetensors").write_bytes(b"x")
    results_io.write_json(run, "metrics.json", {"chrf": 40.0})
    results_io.write_config(run, {})
    results_io.write_env(run)
    z = results_io.package_trained("model_b", run, tmp_path / "civitas_model_b_trained.zip")

    im = _install_models()
    dest = tmp_path / "project"
    names = im.install(z, dest)
    assert (dest / "artifacts/model_b/adapter/adapter_config.json").exists()
    assert len(names) >= 5
    s = im.summarize("model_b", dest)
    assert s["base_model"] == "bigscience/mt0-base" and s["missing_results"] == []

    # finds the zip by name in a search folder, newest copy wins
    monkeypatch.setattr(im, "SEARCH", (tmp_path,))
    assert im.find_zips()["model_b"] == z


def test_install_rejects_foreign_zip(tmp_path):
    bad = tmp_path / "civitas_model_a_trained.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        _install_models().install(bad, tmp_path / "p")


def test_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        results_io.package_trained("model_b", tmp_path / "empty")
