"""Committed notebooks must match scripts/build_notebooks.py and be valid Python."""

import ast
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebooks", REPO / "scripts" / "build_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebooks_are_generated_and_parse(tmp_path):
    b = _builder()
    b.NB_DIR = tmp_path
    for name, cells in (("02_train_model_a.ipynb", b.NB02), ("03a_translate_silver.ipynb", b.NB03A),
                        ("03_train_model_b.ipynb", b.NB03)):
        b.write(name, cells)
        fresh = (tmp_path / name).read_text(encoding="utf-8")
        committed = (REPO / "notebooks" / name).read_text(encoding="utf-8")
        assert fresh == committed, f"{name} is stale: run python scripts/build_notebooks.py"
        for c in json.loads(fresh)["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(l for l in c["source"] if not l.lstrip().startswith("!")))


def test_every_notebook_writes_a_results_zip():
    b = _builder()
    for cells in (b.NB02, b.NB03A, b.NB03):
        src = "".join("".join(c["source"]) for c in cells)
        assert "civitas_results_" in src and ("results_io" in src)
