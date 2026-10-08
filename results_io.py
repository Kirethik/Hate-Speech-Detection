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
        import devices
        info.update(devices.describe())
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


# --------------------------------------------------------------------------- #
# The one file a Colab run hands back: civitas_model_{a,b}_trained.zip
# --------------------------------------------------------------------------- #
# Paths inside the zip are relative to the repo root, so "extract here" in the
# project folder (or scripts/install_models.py) puts every file where config.py
# already looks for it:
#   model_a: artifacts/model_a/best_model.pt          (config.MODEL_A_PT_PATH)
#   model_b: artifacts/model_b/adapter/...            (config.MODEL_B_ADAPTER_DIR)
#   both:    artifacts/results/<kind>/results/...     metrics, history, config, env
#            artifacts/<kind>/MODEL_INFO.json         what is inside + headline metrics
TRAINED_ZIP = {"model_a": "civitas_model_a_trained.zip", "model_b": "civitas_model_b_trained.zip"}
_HEADLINE = {
    "model_a": ("threshold", "hate_macro_f1", "hate_roc_auc", "severity_macro_f1",
                "target_macro_f1", "rationale_token_f1"),
    "model_b": ("chrf", "copy_rate", "safety_rate", "selection_score", "val_loss"),
}


def _half_checkpoint(src: Path, dst: Path) -> None:
    """best_model.pt with fp16 weights (half the download; infer.load_model upcasts)."""
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    for k in [k for k in ckpt if k not in ("model_state_dict", "threshold", "args", "target_classes",
                                            "severity_classes", "normalization_version",
                                            "val_metrics", "epoch", "global_step")]:
        ckpt.pop(k)  # optimizer / scheduler state is only needed for resuming
    ckpt["model_state_dict"] = {k: (v.half() if v.is_floating_point() else v)
                                for k, v in ckpt["model_state_dict"].items()}
    ckpt["stored_dtype"] = "float16"
    torch.save(ckpt, dst)


def package_trained(kind: str, run_dir, zip_path=None) -> Path:
    import tempfile
    run_dir = Path(run_dir)
    res = results_dir(run_dir)
    zip_path = Path(zip_path or run_dir / TRAINED_ZIP[kind])
    metrics = json.loads((res / "metrics.json").read_text(encoding="utf-8")) \
        if (res / "metrics.json").exists() else {}
    info = {"kind": kind, "packaged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir), "has_test_report": (res / "test_report.json").exists(),
            "headline_val_metrics": {k: metrics.get(k) for k in _HEADLINE[kind]}}
    cfg = res / "config.json"
    if cfg.exists():
        info["git_commit"] = json.loads(cfg.read_text(encoding="utf-8")).get("git_commit")

    with tempfile.TemporaryDirectory() as tmp, \
            zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        if kind == "model_a":
            best = run_dir / "best_model.pt"
            if not best.exists():
                raise FileNotFoundError(f"{best} not found - has training finished an eval?")
            small = Path(tmp) / "best_model.pt"
            _half_checkpoint(best, small)
            z.write(small, "artifacts/model_a/best_model.pt", compress_type=zipfile.ZIP_STORED)
            info["files"] = ["artifacts/model_a/best_model.pt"]
        else:
            best = run_dir / "best"
            if not (best / "adapter_config.json").exists():
                raise FileNotFoundError(f"{best}/adapter_config.json not found - has training finished an eval?")
            info["files"] = []
            for f in sorted(best.rglob("*")):
                if f.is_file():
                    arc = f"artifacts/model_b/adapter/{f.relative_to(best).as_posix()}"
                    z.write(f, arc)
                    info["files"].append(arc)
            info["base_model"] = json.loads((best / "adapter_config.json").read_text()).get(
                "base_model_name_or_path")
        for f in sorted(res.rglob("*")):
            if f.is_file():
                z.write(f, f"artifacts/results/{kind}/results/{f.relative_to(res).as_posix()}")
        z.writestr(f"artifacts/{kind}/MODEL_INFO.json", json.dumps(info, indent=2))
    return zip_path


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Collect a run's results/ and build the trained-model zip")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--kind", choices=["model_a", "model_b"], required=True)
    p.add_argument("--data_manifest", default=None)
    p.add_argument("--zip", default=None, help=f"default: <run_dir>/{TRAINED_ZIP['model_a']} (or _b)")
    a = p.parse_args(argv)
    if a.kind == "model_a":
        collect_model_a(a.run_dir, a.data_manifest)
    missing = missing_files(results_dir(a.run_dir))
    if missing:
        print("WARNING: results/ is missing", missing)
    out = package_trained(a.kind, a.run_dir, a.zip)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
