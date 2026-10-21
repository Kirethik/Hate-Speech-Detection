"""
Civitas AI — Model Loader and Lifecycle Manager

Discovers trained artifacts from standard project paths:
- model_artifacts/hate_detector/
- model_artifacts/alternative_generator/
Manages singleton instances of Model A and Model B for inference.
Defaults safely to explicit Demo Mode until weights are installed, preventing
accidental large downloads from the Hugging Face Hub during local development.
"""

import os
from pathlib import Path
from typing import Optional, Dict

import torch
from src.models.interfaces import ModelStatus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard artifact directories per Master Prompt contract
DETECTOR_ARTIFACT_DIR = REPO_ROOT / "model_artifacts" / "hate_detector"
GENERATOR_ARTIFACT_DIR = REPO_ROOT / "model_artifacts" / "alternative_generator"

# Fallback legacy paths (if user used legacy scripts)
LEGACY_MODEL_A = REPO_ROOT / "artifacts" / "model_a" / "best_model.pt"
LEGACY_MODEL_B = REPO_ROOT / "artifacts" / "model_b" / "adapter"

_CACHED_MODEL_A = None
_CACHED_MODEL_B = None


def get_inference_mode() -> str:
    """Returns 'demo' or 'real'. Defaults to 'demo' unless artifacts exist and real mode requested."""
    forced = os.getenv("CIVITAS_MODE", "").lower()
    if forced in ("demo", "real"):
        return forced
    # If artifacts exist in model_artifacts, default to real, else demo
    if find_model_a_artifact() and find_model_b_artifact():
        return "real"
    return "demo"


def find_model_a_artifact() -> Optional[Path]:
    candidates = [
        DETECTOR_ARTIFACT_DIR / "best_model.pt",
        DETECTOR_ARTIFACT_DIR / "model.safetensors",
        DETECTOR_ARTIFACT_DIR / "pytorch_model.bin",
        LEGACY_MODEL_A,
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def find_model_b_artifact() -> Optional[Path]:
    candidates = [
        GENERATOR_ARTIFACT_DIR,
        GENERATOR_ARTIFACT_DIR / "adapter",
        LEGACY_MODEL_B,
    ]
    for p in candidates:
        if p.exists() and (
            (p / "adapter_config.json").exists() or 
            (p / "model.safetensors").exists()
        ):
            return p
    return None


def get_preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_detector_model(force_reload: bool = False):
    """
    Returns (model_instance, is_installed, status_msg).
    In Demo Mode, does not load heavy weights and returns None.
    """
    global _CACHED_MODEL_A

    if get_inference_mode() == "demo":
        return None, False, "Operating in Demo Mode. Model A weights not loaded."

    if _CACHED_MODEL_A is not None and not force_reload:
        return _CACHED_MODEL_A, True, "Loaded from memory cache"

    artifact_path = find_model_a_artifact()
    if not artifact_path:
        return (
            None,
            False,
            f"Model A weights not found in {DETECTOR_ARTIFACT_DIR.relative_to(REPO_ROOT)}. Please copy trained weights here."
        )

    try:
        from infer import load_model
        device = get_preferred_device()
        model_a = load_model(str(artifact_path), device=device)
        _CACHED_MODEL_A = model_a
        return model_a, True, f"Loaded Model A from {artifact_path}"
    except Exception as e:
        return None, False, f"Failed to load Model A from {artifact_path}: {str(e)}"


def get_generator_model(force_reload: bool = False):
    """
    Returns (model_b_instance, is_installed, status_msg).
    In Demo Mode, returns None to avoid Hub network downloads.
    """
    global _CACHED_MODEL_B

    if get_inference_mode() == "demo":
        return None, False, "Operating in Demo Mode. Model B adapter not loaded."

    if _CACHED_MODEL_B is not None and not force_reload:
        return _CACHED_MODEL_B, True, "Loaded from memory cache"

    artifact_dir = find_model_b_artifact()
    if not artifact_dir:
        return (
            None,
            False,
            f"Model B adapter not found in {GENERATOR_ARTIFACT_DIR.relative_to(REPO_ROOT)}. Please copy trained adapter here."
        )

    try:
        from model_b_generation.model_gen import load_for_inference
        device = get_preferred_device()
        # Ensure offline mode is respected if base model cached
        model_b = load_for_inference(str(artifact_dir), device=device)
        _CACHED_MODEL_B = model_b
        return model_b, True, f"Loaded Model B from {artifact_dir}"
    except Exception as e:
        return None, False, f"Failed to load Model B from {artifact_dir}: {str(e)}"


def get_system_model_status() -> Dict[str, ModelStatus]:
    dev_str = str(get_preferred_device())
    mode = get_inference_mode()

    art_a = find_model_a_artifact()
    a_installed = art_a is not None
    status_a = ModelStatus(
        name="Model A (Hate Speech Detector)",
        model_type="hate_detector",
        is_installed=a_installed,
        weights_path=str(art_a) if art_a else "model_artifacts/hate_detector/best_model.pt (MISSING)",
        device=dev_str,
        is_demo_mode=(mode == "demo" or not a_installed),
        status_message="Active (Real Model)" if (mode == "real" and a_installed) else "Demo Mode (Awaiting Google Colab export zip / weights in model_artifacts/hate_detector/)"
    )

    art_b = find_model_b_artifact()
    b_installed = art_b is not None
    status_b = ModelStatus(
        name="Model B (Constructive Alternative Generator)",
        model_type="alternative_generator",
        is_installed=b_installed,
        weights_path=str(art_b) if art_b else "model_artifacts/alternative_generator/ (MISSING)",
        device=dev_str,
        is_demo_mode=(mode == "demo" or not b_installed),
        status_message="Active (Real Model)" if (mode == "real" and b_installed) else "Demo Mode (Awaiting Google Colab export zip / adapter in model_artifacts/alternative_generator/)"
    )

    return {
        "model_a": status_a,
        "model_b": status_b,
    }
