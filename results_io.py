"""
One place that writes and packs training results, so every Colab run leaves
the same files behind and `scripts/ingest_results.py` can read them locally.

Layout of <run_dir>/results/:
    metrics.json            final/best val metrics
    metrics_history.jsonl   one JSON line per eval
    config.json             hyperparameters + git commit + data manifest hash
    env.json                GPU, library versions, wall time
    samples.jsonl           (Model B) generations for eyeballing
    test_report.json        written only by the RUN ONCE test cell
    hatecheck_report.json   (Model A) written only by the RUN ONCE cell

Checkpoints never go in results/ so the zip stays small.
"""

import hashlib
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path

RESULTS_DIRNAME = "results"
REQUIRED_FILES = ("metrics.json", "config.json", "env.json")


def results_dir(run_dir) -> Path:
    d = Path(run_dir) / RESULTS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return str(obj)


def write_json(run_dir, name: str, data) -> Path:
    path = results_dir(run_dir) / name
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_jsonl(run_dir, name: str, row: dict) -> Path:
    path = results_dir(run_dir) / name
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    return path


def write_jsonl(run_dir, name: str, rows) -> Path:
    path = results_dir(run_dir) / name
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    return path


def git_commit(repo_dir=".") -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def file_sha256(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_config(run_dir, args: dict, data_manifest=None, repo_dir=".") -> Path:
    return write_json(run_dir, "config.json", {
        "args": args,
        "git_commit": git_commit(repo_dir),
        "data_manifest_sha256": file_sha256(data_manifest) if data_manifest else None,
    })


def env_info(started_at: float | None = None) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    for lib in ("transformers", "peft"):
        try:
            info[lib] = __import__(lib).__version__
        except ImportError:
            pass
    if started_at is not None:
        info["wall_time_min"] = round((time.time() - started_at) / 60, 1)
    return info


def write_env(run_dir, started_at: float | None = None) -> Path:
    return write_json(run_dir, "env.json", env_info(started_at))


def zip_results(run_dir, zip_path) -> Path:
    """Zip results/ (never checkpoints). Paths inside the zip start at results/."""
    src = results_dir(run_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f"{RESULTS_DIRNAME}/{f.relative_to(src).as_posix()}")
    return zip_path


def missing_files(results_path) -> list[str]:
    p = Path(results_path)
    return [n for n in REQUIRED_FILES if not (p / n).exists()]


# train.py (Model A) writes its own files into output_dir; this maps them onto
# the results/ layout. Model B's train_gen.py writes results/ directly.
_MODEL_A_FILES = {
    "best_metrics.json": "metrics.json",
    "test_report.json": "test_report.json",
    "hatecheck.json": "hatecheck_report.json",
    "robustness.json": "robustness.json",
}


def collect_model_a(run_dir, data_manifest=None, repo_dir=".") -> Path:
    import csv
    import shutil
    run_dir = Path(run_dir)
    out = results_dir(run_dir)
    for src, dst in _MODEL_A_FILES.items():
        if (run_dir / src).exists():
            shutil.copyfile(run_dir / src, out / dst)
    hist = run_dir / "metrics_history.csv"
    if hist.exists():
        with hist.open(encoding="utf-8") as f:
            write_jsonl(run_dir, "metrics_history.jsonl", csv.DictReader(f))
    args = {}
    best = run_dir / "best_model.pt"
    if best.exists():
        try:
            import torch
            args = torch.load(best, map_location="cpu", weights_only=False).get("args", {})
        except Exception as e:  # config.json still gets written, just without args
            args = {"error": f"could not read args from best_model.pt: {e}"}
    write_config(run_dir, args, data_manifest, repo_dir)
    write_env(run_dir)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Collect a run's results/ folder and zip it")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--kind", choices=["model_a", "none"], default="none",
                   help="model_a: first gather train.py's files into results/")
    p.add_argument("--data_manifest", default=None)
    p.add_argument("--zip", required=True, help="where to write civitas_results_<run>.zip")
    a = p.parse_args(argv)
    if a.kind == "model_a":
        collect_model_a(a.run_dir, a.data_manifest)
    missing = missing_files(results_dir(a.run_dir))
    if missing:
        print("WARNING: results/ is missing", missing)
    print("wrote", zip_results(a.run_dir, a.zip))


if __name__ == "__main__":
    main()
