"""
Install the trained models that Colab hands back. Put the files in the project
folder (or leave them in Downloads) and run:

    python scripts/install_models.py            # finds civitas_model_{a,b}_trained.zip
    python scripts/install_models.py --check    # also loads each model once to prove it works

Looks in: the project folder, artifacts/, and your Downloads folder.
Each zip only contains paths under artifacts/, so files land exactly where
config.py looks for them:
    artifacts/model_a/best_model.pt     artifacts/model_b/adapter/
    artifacts/results/<model>/results/  (metrics, history, config, env)
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import results_io  # noqa: E402

SEARCH = (REPO, REPO / "artifacts", Path.home() / "Downloads")


def find_zips(extra=()) -> dict[str, Path]:
    found = {}
    for kind, name in results_io.TRAINED_ZIP.items():
        cands = [Path(p) for p in extra if Path(p).name.startswith(name.removesuffix(".zip"))]
        cands += [d / name for d in SEARCH]
        # newest wins (Downloads may hold "civitas_model_a_trained (1).zip" too)
        cands += [p for d in SEARCH if d.exists() for p in d.glob(name.replace(".zip", "*.zip"))]
        cands = [p for p in cands if p.exists()]
        if cands:
            found[kind] = max(cands, key=lambda p: p.stat().st_mtime)
    return found


def install(zip_path: Path, dest: Path = REPO) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in names:
            if not n.startswith("artifacts/") or ".." in Path(n).parts:
                raise ValueError(f"{zip_path.name}: unexpected path {n!r} - not a Civitas trained-model zip")
        z.extractall(dest)
    return names


def summarize(kind: str, dest: Path = REPO) -> dict:
    out = {}
    info = dest / "artifacts" / kind / "MODEL_INFO.json"
    if info.exists():
        out.update(json.loads(info.read_text(encoding="utf-8")))
    res = dest / "artifacts" / "results" / kind / "results"
    out["missing_results"] = results_io.missing_files(res)
    for name in ("test_report.json", "hatecheck_report.json"):
        if (res / name).exists():
            data = json.loads((res / name).read_text(encoding="utf-8"))
            out[name] = {k: v for k, v in data.items() if isinstance(v, (int, float, str)) or v is None}
    m = res / "metrics.json"
    if m.exists():
        data = json.loads(m.read_text(encoding="utf-8"))
        for k, v in data.items():
            if isinstance(v, dict) and (k.startswith("by_") or k.startswith("hate_by_")):
                out[f"val:{k}"] = {g: (round(x.get("macro_f1", x.get("chrf", 0)) or 0, 3)
                                       if isinstance(x, dict) else x) for g, x in v.items()}
    return out


def check(kind: str, dest: Path = REPO) -> str:
    """Load the installed model once on CPU."""
    if kind == "model_a":
        from infer import load_model, score_batch
        m = load_model(str(dest / "artifacts/model_a/best_model.pt"), "cpu")
        r = score_batch(m, ["have a lovely day"])[0]
        return f"Model A loads (threshold {m.threshold:.2f}); 'have a lovely day' -> hate_prob {r['hate_prob']:.3f}"
    from model_b_generation.model_gen import load_for_inference
    mb = load_for_inference(str(dest / "artifacts/model_b/adapter"), device="cpu")
    return f"Model B loads (base {mb.base}, adapter {'found' if mb.adapter_dir else 'MISSING'})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="install civitas_model_{a,b}_trained.zip")
    p.add_argument("zips", nargs="*", help="explicit zip paths (default: search)")
    p.add_argument("--check", action="store_true", help="load each model once after installing")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    found = find_zips(a.zips)
    if not found:
        print("No civitas_model_a_trained.zip / civitas_model_b_trained.zip found in:")
        for d in SEARCH:
            print("  ", d)
        return 1
    for kind, z in sorted(found.items()):
        names = install(z)
        print(f"===== installed {z.name} ({len(names)} files)")
        print(json.dumps(summarize(kind), indent=1, ensure_ascii=False))
        if a.check:
            print(check(kind))
    return 0


if __name__ == "__main__":
    sys.exit(main())
