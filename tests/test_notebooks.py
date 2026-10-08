"""Committed notebooks must match scripts/build_notebooks.py and be valid Python."""

import ast
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebooks_are_generated_and_parse(tmp_path):
    b = _load("build_notebooks")
    b.NB_DIR = tmp_path
    for name, cells in b.NOTEBOOKS.items():
        b.write(name, cells)
        fresh = (tmp_path / name).read_text(encoding="utf-8")
        committed = (REPO / "notebooks" / name).read_text(encoding="utf-8")
        assert fresh == committed, f"{name} is stale: run python scripts/build_notebooks.py"
        for c in json.loads(fresh)["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(l for l in c["source"] if not l.lstrip().startswith("!")))


def test_every_notebook_writes_a_results_zip():
    b = _load("build_notebooks")
    for cells in b.NOTEBOOKS.values():
        src = "".join("".join(c["source"]) for c in cells)
        assert "_trained.zip" in src and "results_io" in src and "civitas_results_" not in src


def test_bundle_has_code_data_and_notebook(tmp_path, monkeypatch):
    mb = _load("make_bundles")
    repo = tmp_path / "repo"
    (repo / "data" / "gen").mkdir(parents=True)
    (repo / "notebooks").mkdir()
    for f in ("config.py", "train.py", "devices.py"):
        (repo / f).write_text("# stub")
    (repo / "eval").mkdir()
    (repo / "eval" / "probe_model.py").write_text("# stub")
    (repo / "model_b_generation" / "tests").mkdir(parents=True)
    (repo / "model_b_generation" / "train_gen.py").write_text("# stub")
    (repo / "model_b_generation" / "tests" / "test_x.py").write_text("# stub")
    (repo / "converters").mkdir()
    (repo / "notebooks" / "train_model_a.ipynb").write_text("{}")
    import pandas as pd
    df = pd.DataFrame({"text": ["a", "b"], "language": ["en", "ta"], "hate_label": [0, 1], "source": ["x", "y"]})
    for s in ("train", "val", "test"):
        df.to_csv(repo / "data" / f"{s}.csv", index=False)
    monkeypatch.setattr(mb, "REPO", repo)
    z = mb.build("a", tmp_path / "out")
    import zipfile
    names = set(zipfile.ZipFile(z).namelist())
    assert {"civitas_model_a/train_model_a.ipynb", "civitas_model_a/README.md",
            "civitas_model_a/code/train.py", "civitas_model_a/code/eval/probe_model.py",
            "civitas_model_a/code/model_b_generation/train_gen.py",
            "civitas_model_a/data/train.parquet", "civitas_model_a/data/manifest.json"} <= names
    assert not any("/tests/" in n for n in names)
    m = json.loads(zipfile.ZipFile(z).read("civitas_model_a/data/manifest.json"))
    assert m["splits"]["val"]["rows"] == 2 and "train.parquet" in m["files"]
