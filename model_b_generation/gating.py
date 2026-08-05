"""
Model A → Model B gating — the single integration point between the
hate-speech detector and the counter-narrative generator.

This module is the ONLY place where Model A and Model B are coupled.
It consumes Model A's output (the prediction dict from infer.py) and
decides whether to invoke Model B's generate_alternatives().

Design constraints (from the blueprint):
  - Model B must NEVER run on non-hateful input (compute-saving gate).
  - The gating threshold is Model A's own checkpoint threshold, loaded
    dynamically — not a hardcoded value.
  - Both models are lazy-loaded and cached for re-use.

Usage:
    from model_b_generation.gating import moderate
    result = moderate("some text to check", "en")
"""

import sys
import logging
import torch
from pathlib import Path

# Add repo root to path for Model A imports
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from infer import load_model as load_model_a, predict as predict_a
    _MODEL_A_AVAILABLE = True
except ImportError:
    _MODEL_A_AVAILABLE = False

from model_b_generation.infer_gen import (
    generate_alternatives,
    load_model_for_inference as load_model_b,
)

logger = logging.getLogger(__name__)

# ── Lazy-loaded model caches ───────────────────────────────────────────────
# Single-threaded use assumed (documented). For multi-threaded serving,
# wrap in a threading.Lock.
_model_a_cache: tuple | None = None   # (model, tokenizer, threshold)
_model_b_cache: tuple | None = None   # (model, tokenizer)
_device: torch.device | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _ensure_model_a():
    """Lazy-load Model A's detector (once)."""
    global _model_a_cache
    if _model_a_cache is not None:
        return _model_a_cache

    if not _MODEL_A_AVAILABLE:
        raise RuntimeError(
            "Model A's infer.py could not be imported. "
            "Ensure the repo root is on sys.path and infer.py exists."
        )

    ckpt_path = str(Path(_REPO_ROOT) / "checkpoints" / "model_a" / "best_model.pt")
    device = _get_device()
    logger.info("Loading Model A from %s on %s", ckpt_path, device)
    _model_a_cache = load_model_a(ckpt_path, device)
    logger.info("Model A loaded (threshold=%.2f)", _model_a_cache[2])
    return _model_a_cache


def _ensure_model_b():
    """Lazy-load Model B's generator (once)."""
    global _model_b_cache
    if _model_b_cache is not None:
        return _model_b_cache

    ckpt_dir = str(Path(_REPO_ROOT) / "checkpoints_gen")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Model B from %s on %s", ckpt_dir, device)
    _model_b_cache = load_model_b(ckpt_dir, device)
    logger.info("Model B loaded")
    return _model_b_cache


def moderate(text: str, language: str) -> dict:
    """
    Full moderation pipeline: classify with Model A, gate, generate with Model B.

    Calls Model A's inference to classify `text`. If the prediction label
    is ABUSIVE (using Model A's own checkpoint threshold), calls Model B's
    generate_alternatives(). Otherwise returns a 'safe' status and skips
    Model B entirely — this is the compute-saving gate.

    Parameters
    ----------
    text : str
        The text to moderate.
    language : str
        ISO code: 'en', 'hi', or 'ta'.

    Returns
    -------
    dict
        {"status": "safe"} when Model A does not flag the text, or
        {"status": "flagged", "target": str, "severity": str,
         "p_abuse": float, "alternatives": list[str]} when flagged.
    """
    # ── Stage 1: classify with Model A ─────────────────────────────────
    mod_a, tok_a, thresh_a = _ensure_model_a()
    device_a = next(mod_a.parameters()).device

    prediction = predict_a(mod_a, tok_a, thresh_a, text, device_a)

    if prediction.get("label") != "ABUSIVE":
        return {"status": "safe"}

    # ── Stage 2: generate counter-narratives with Model B ──────────────
    mod_b, tok_b = _ensure_model_b()

    alternatives = generate_alternatives(
        text=text,
        language=language,
        style="empathetic",
        n=3,
        model=mod_b,
        tokenizer=tok_b,
        model_a=(mod_a, tok_a, thresh_a),  # pass for safety check
    )

    return {
        "status": "flagged",
        "target": prediction.get("target"),
        "severity": prediction.get("severity"),
        "p_abuse": prediction.get("p_abuse"),
        "alternatives": alternatives,
    }


def reset_cache():
    """Clear cached models (useful for testing)."""
    global _model_a_cache, _model_b_cache, _device
    _model_a_cache = None
    _model_b_cache = None
    _device = None
