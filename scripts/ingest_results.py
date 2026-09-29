"""
Unpack a Colab results zip and print a compact summary, so results never have
to be copy-pasted out of notebook cells.

    python scripts/ingest_results.py                       # every artifacts/civitas_results_*.zip
    python scripts/ingest_results.py artifacts/civitas_results_model_b.zip

Each zip is extracted to artifacts/results/<run>/ (e.g. artifacts/results/model_b/).
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import results_io  # noqa: E402

ARTIFACTS = REPO / "artifacts"


def extract(zip_path: Path, dest_root: Path) -> Path:
    run = zip_path.stem.removeprefix("civitas_results_")
    dest = dest_root / run
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in zip: {name}")
        z.extractall(dest)
    return dest / results_io.RESULTS_DIRNAME


def _scalars(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
            if isinstance(v, (int, float, str, bool)) or v is None}


def summarize(res: Path) -> dict:
    out = {"results_dir": str(res), "missing": results_io.missing_files(res)}
    for name in ("metrics.json", "test_report.json", "hatecheck_report.json"):
        p = res / name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            out[name] = _scalars(data)
            for k, v in data.items():
                if isinstance(v, dict) and (k.startswith("by_") or k.startswith("hate_by_")):
                    out[f"{name}:{k}"] = {g: (_scalars(m) if isinstance(m, dict) else m)
                                          for g, m in v.items()}
    hist = res / "metrics_history.jsonl"
    if hist.exists():
        out["evals_logged"] = sum(1 for line in hist.read_text(encoding="utf-8").splitlines() if line)
    env = res / "env.json"
    if env.exists():
        out["env"] = json.loads(env.read_text(encoding="utf-8"))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="unpack + summarize civitas_results_*.zip")
    p.add_argument("zips", nargs="*")
    p.add_argument("--dest", default=str(ARTIFACTS / "results"))
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    zips = [Path(z) for z in a.zips] or sorted(ARTIFACTS.glob("civitas_results_*.zip"))
    if not zips:
        print(f"no civitas_results_*.zip in {ARTIFACTS}")
        return 1
    for z in zips:
        res = extract(z, Path(a.dest))
        print(f"===== {z.name}")
        print(json.dumps(summarize(res), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
