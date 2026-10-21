"""
Unit tests for dataset schemas, label mapping, and preprocessing logic.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"


def test_label_mapping_integrity():
    map_file = DATASETS_DIR / "label_mapping.json"
    assert map_file.exists(), "datasets/label_mapping.json must exist"
    with open(map_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "detector" in data
    assert "binary" in data["detector"]
    assert data["detector"]["binary"]["0"] == "non_hateful"
    assert data["detector"]["binary"]["1"] == "hateful"
    assert "generator" in data
    assert "supported_tones" in data["generator"]
    assert "calm" in data["generator"]["supported_tones"]
    assert "languages" in data
    assert "en" in data["languages"]


def test_detector_split_schemas():
    det_dir = DATASETS_DIR / "processed" / "detector"
    for split in ["train.csv", "validation.csv", "test.csv"]:
        split_file = det_dir / split
        if split_file.exists():
            df = pd.read_csv(split_file, nrows=50)
            required = ["text", "label", "label_name", "language", "source_dataset"]
            for col in required:
                assert col in df.columns, f"Column {col} missing in detector {split}"
            assert set(df["label"].unique()).issubset({0, 1}), f"Non-binary labels found in detector {split}"


def test_generator_split_schemas():
    gen_dir = DATASETS_DIR / "processed" / "generator"
    for split in ["train.csv", "validation.csv", "test.csv"]:
        split_file = gen_dir / split
        if split_file.exists():
            df = pd.read_csv(split_file, nrows=50)
            required = ["input_text", "target_text", "language", "source_dataset", "task", "tone"]
            for col in required:
                assert col in df.columns, f"Column {col} missing in generator {split}"
