"""
Package processed data splits for Colab training.

Produces artifacts/civitas_data_v2.zip containing:
  - train.parquet, val.parquet, test.parquet
  - label_maps.py
  - config.py
  - manifest.json  (row counts, git commit hash, creation timestamp)

Usage:
    python scripts/package_data.py                 # Model A -> civitas_data_v2.zip
    python scripts/package_data.py --kind gen      # Model B -> civitas_gen_data.zip
                                                   # (reads data/gen/{train,val,test}.parquet)
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data",
                        help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--out_dir", default="artifacts",
                        help="Output directory for the zip file")
    parser.add_argument("--kind", choices=["model_a", "gen"], default="model_a")
    args = parser.parse_args()
    if args.kind == "gen" and args.data_dir == "data":
        args.data_dir = "data/gen"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent

    # ── Load and convert splits to Parquet ─────────────────────────────────
    manifest = {
        "git_commit": _git_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "files": {},
    }

    tmp_dir = out_dir / "_tmp_package"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_path = next((p for p in (data_dir / f"{split}.parquet", data_dir / f"{split}.csv")
                         if p.exists()), None)
        if src_path is None:
            print(f"  WARNING: no {split}.parquet/.csv in {data_dir} — skipping {split} split")
            continue

        df = pd.read_parquet(src_path) if src_path.suffix == ".parquet" else pd.read_csv(src_path)
        parquet_path = tmp_dir / f"{split}.parquet"
        df.to_parquet(parquet_path, index=False, compression="zstd")

        info = {
            "rows": len(df),
            "language_counts": df["language"].value_counts().to_dict(),
            "source_counts": df["source"].value_counts().to_dict(),
        }
        if args.kind == "model_a":
            info["hate_label_counts"] = df["hate_label"].value_counts().to_dict()
            info["script_counts"] = df["script"].value_counts().to_dict() if "script" in df else {}
        else:
            info["task_counts"] = df["task"].value_counts().to_dict()
        manifest["splits"][split] = info
        manifest["files"][f"{split}.parquet"] = _file_sha256(parquet_path)
        print(f"  {split}: {len(df):>7} rows -> {parquet_path.name}")

    # ── Copy supporting files ───────────────────────────────────────────────
    support = (("label_maps.py", "config.py", "data_config.yaml") if args.kind == "model_a"
               else ("config.py",))
    for fname in support:
        src = repo_root / fname
        if src.exists():
            shutil.copy2(src, tmp_dir / fname)
            manifest["files"][fname] = _file_sha256(src)
        else:
            print(f"  WARNING: {fname} not found at {src}")

    # ── Write manifest ──────────────────────────────────────────────────────
    manifest_path = tmp_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ── Zip everything ──────────────────────────────────────────────────────
    zip_path = out_dir / ("civitas_data_v2.zip" if args.kind == "model_a" else "civitas_gen_data.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in tmp_dir.iterdir():
            zf.write(item, item.name)

    shutil.rmtree(tmp_dir)

    total_rows = sum(s["rows"] for s in manifest["splits"].values())
    print(f"\nOK {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Git commit: {manifest['git_commit']}")
    print(f"\nUpload this file to MyDrive/civitas/ in Google Drive before running Colab.")


if __name__ == "__main__":
    main()
