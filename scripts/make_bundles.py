"""
Builds the two Colab bundles. Each one is self-contained: training code,
every dataset that model needs, and its notebook.

    python scripts/make_bundles.py            # both
    python scripts/make_bundles.py --only a   # just Model A

Prerequisites (on this PC, repo root):
    Model A: python build_dataset.py && python prepare_splits.py
             -> data/{train,val,test}.csv
    Model B: python -m model_b_generation.build_dataset_gen && python -m model_b_generation.prepare_splits_gen
             -> data/gen/pairs.parquet + data/gen/{train,val,test}.parquet

Output:
    artifacts/civitas_model_a.zip   ->  civitas_model_a/{train_model_a.ipynb, README.md, code/, data/}
    artifacts/civitas_model_b.zip   ->  civitas_model_b/{train_model_b.ipynb, README.md, code/, data/}
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"

# training-side code only: no server, speech pipeline, frontend or tests
CODE_DIRS = ("converters", "eval", "model_b_generation")
CODE_EXCLUDE_PARTS = {"tests", "__pycache__"}
CODE_ROOT_FILES = ("config.py", "label_maps.py", "model.py", "dataset.py", "text_norm.py",
                   "train.py", "infer.py", "devices.py", "results_io.py", "dehumanization.py",
                   "nli_stage.py", "identity_augment.py", "script_augment.py",
                   "spelling_augment.py", "data_config.yaml", "requirements.txt")

README = {
    "a": """# Civitas: Model A bundle (hate speech detector)

Everything needed to train Model A in Google Colab: the code (`code/`), the prepared
train/val/test data (`data/`) and the notebook.

1. In Colab: File -> Upload notebook -> `train_model_a.ipynb` (no need to unzip anything).
2. Runtime -> Change runtime type -> **T4 GPU**, then **Runtime -> Run all**. Allow Drive access.
3. When "Choose files" appears, pick `civitas_model_a.zip`.
4. Disconnected? Reconnect and Run all again: training resumes.
5. Download My Drive -> `civitas_runs/civitas_model_a/civitas_model_a_trained.zip`.

Data: {summary}
""",
    "b": """# Civitas: Model B bundle (alternate speech generator)

Everything needed to train Model B in Google Colab: the code (`code/`), the English/Hindi
Rewrite + Respond pairs (`data/`) and the notebook. The notebook adds Tamil, Telugu,
Malayalam, Hindi and Roman-Urdu "silver" translations itself (IndicTrans2) before training.

1. In Colab: File -> Upload notebook -> `train_model_b.ipynb` (no need to unzip anything).
2. Optional: add a Hugging Face read token as Colab secret `HF_TOKEN` (for IndicTrans2).
3. Runtime -> Change runtime type -> **T4 GPU**, then **Runtime -> Run all**. Allow Drive access.
4. When "Choose files" appears, pick `civitas_model_b.zip`.
5. Download My Drive -> `civitas_runs/civitas_model_b/civitas_model_b_trained.zip`.

Data: {summary}
""",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return (r.stdout.strip() or "unknown") + ("+dirty" if dirty else "")


def copy_code(dest: Path):
    for name in CODE_ROOT_FILES:
        src = REPO / name
        if src.exists():
            shutil.copy2(src, dest / name)
    for d in CODE_DIRS:
        for f in (REPO / d).rglob("*"):
            rel = f.relative_to(REPO)
            if f.is_file() and not (CODE_EXCLUDE_PARTS & set(rel.parts)) and f.suffix in (".py", ".txt", ".yaml"):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest / rel)


def read_any(path_no_ext: Path) -> pd.DataFrame:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = path_no_ext.with_suffix(ext)
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"{path_no_ext}.parquet/.csv not found - build the data first (see --help)")


def write_data(dest: Path, frames: dict, extra_files=(), kind="a") -> dict:
    manifest = {"git_commit": git_commit(), "created_at": datetime.now(timezone.utc).isoformat(),
                "kind": kind, "splits": {}, "files": {}}
    for name, df in frames.items():
        out = dest / f"{name}.parquet"
        df.to_parquet(out, index=False, compression="zstd")
        info = {"rows": int(len(df)),
                "language_counts": {k: int(v) for k, v in df["language"].value_counts().items()},
                "source_counts": {k: int(v) for k, v in df["source"].value_counts().items()}}
        if "hate_label" in df:
            info["hate_label_counts"] = {str(k): int(v) for k, v in df["hate_label"].value_counts().items()}
        if "task" in df:
            info["task_counts"] = {k: int(v) for k, v in df["task"].value_counts().items()}
        manifest["splits"][name] = info
        manifest["files"][out.name] = sha256(out)
    for src in extra_files:
        src = Path(src)
        if src.exists():
            shutil.copy2(src, dest / src.name)
            manifest["files"][src.name] = sha256(dest / src.name)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def zip_dir(folder: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent).as_posix())


def build(which: str, out_dir: Path) -> Path:
    name = f"civitas_model_{which}"
    root = out_dir / name
    if root.exists():
        shutil.rmtree(root)
    (root / "code").mkdir(parents=True)
    (root / "data").mkdir()
    copy_code(root / "code")

    if which == "a":
        data = REPO / "data"
        frames = {s: read_any(data / s) for s in ("train", "val", "test")}
        manifest = write_data(root / "data", frames,
                              [REPO / "data_config.yaml", data / "DATA_REPORT.md"], "a")
    else:
        gen = REPO / "data" / "gen"
        frames = {s: read_any(gen / s) for s in ("pairs", "train", "val", "test")}
        manifest = write_data(root / "data", frames, [gen / "SPLITS_REPORT.json"], "b")

    shutil.copy2(REPO / "notebooks" / f"train_model_{which}.ipynb", root)
    summary = ", ".join(f"{k} {v['rows']:,} rows" for k, v in manifest["splits"].items())
    (root / "README.md").write_text(README[which].format(summary=summary), encoding="utf-8")
    zip_path = out_dir / f"{name}.zip"
    zip_dir(root, zip_path)
    print(f"{zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)  {summary}")
    return zip_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the self-contained Colab bundles")
    p.add_argument("--only", choices=["a", "b"], default=None)
    p.add_argument("--out_dir", default=str(ARTIFACTS))
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for which in ([a.only] if a.only else ["a", "b"]):
        build(which, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
