"""
Tests for the Model A → Model B gating logic.

Validates:
  - Model B is NEVER invoked when Model A returns a non-hate prediction.
  - Model B IS invoked when Model A returns ABUSIVE, and alternatives are
    included in the response.
  - The return dict has the correct structure in both cases.

All tests use mocks — no actual model loading or GPU required.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add package to path
_PKG_DIR = str(Path(__file__).resolve().parent.parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


class TestGatingLogic:
    """Test the moderate() function's gating behavior."""

    def setup_method(self):
        """Reset the model caches before each test."""
        from model_b_generation.gating import reset_cache
        reset_cache()

    @patch("model_b_generation.gating.predict_a")
    @patch("model_b_generation.gating._ensure_model_a")
    def test_safe_text_skips_model_b(self, mock_ensure_a, mock_predict):
        """When Model A says 'clean', Model B must not be called."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_ensure_a.return_value = (mock_model, MagicMock(), 0.71)
        mock_predict.return_value = {
            "text": "hello world",
            "p_abuse": 0.12,
            "label": "clean",
            "threshold": 0.71,
            "severity": "normal",
            "target": "none",
            "reasons": [],
        }

        from model_b_generation.gating import moderate

        with patch("model_b_generation.gating._ensure_model_b") as mock_ensure_b:
            result = moderate("hello world", "en")

            # Model B should never be called
            mock_ensure_b.assert_not_called()
            assert result["status"] == "safe"

    @patch("model_b_generation.gating.generate_alternatives")
    @patch("model_b_generation.gating._ensure_model_b")
    @patch("model_b_generation.gating.predict_a")
    @patch("model_b_generation.gating._ensure_model_a")
    def test_abusive_text_invokes_model_b(
        self, mock_ensure_a, mock_predict, mock_ensure_b, mock_gen_alts
    ):
        """When Model A says ABUSIVE, Model B must be called."""
        mock_model_a = MagicMock()
        mock_model_a.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_ensure_a.return_value = (mock_model_a, MagicMock(), 0.71)
        mock_predict.return_value = {
            "text": "you are vermin",
            "p_abuse": 0.92,
            "label": "ABUSIVE",
            "threshold": 0.71,
            "severity": "hate",
            "target": "caste_ethnicity",
            "reasons": [],
        }
        mock_ensure_b.return_value = (MagicMock(), MagicMock())
        mock_gen_alts.return_value = [
            "Let's treat everyone with respect.",
            "Every person deserves dignity.",
        ]

        from model_b_generation.gating import moderate
        result = moderate("you are vermin", "en")

        assert result["status"] == "flagged"
        assert result["target"] == "caste_ethnicity"
        assert result["severity"] == "hate"
        assert len(result["alternatives"]) == 2
        mock_gen_alts.assert_called_once()

    @patch("model_b_generation.gating.generate_alternatives")
    @patch("model_b_generation.gating._ensure_model_b")
    @patch("model_b_generation.gating.predict_a")
    @patch("model_b_generation.gating._ensure_model_a")
    def test_flagged_result_structure(
        self, mock_ensure_a, mock_predict, mock_ensure_b, mock_gen_alts
    ):
        """The flagged dict must have all required keys."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_ensure_a.return_value = (mock_model, MagicMock(), 0.71)
        mock_predict.return_value = {
            "text": "hate text",
            "p_abuse": 0.85,
            "label": "ABUSIVE",
            "threshold": 0.71,
            "severity": "offensive_profanity",
            "target": "gender",
            "reasons": [],
        }
        mock_ensure_b.return_value = (MagicMock(), MagicMock())
        mock_gen_alts.return_value = ["alternative"]

        from model_b_generation.gating import moderate
        result = moderate("hate text", "en")

        required_keys = {"status", "target", "severity", "alternatives", "p_abuse"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    @patch("model_b_generation.gating.predict_a")
    @patch("model_b_generation.gating._ensure_model_a")
    def test_safe_result_is_minimal(self, mock_ensure_a, mock_predict):
        """A safe result should only have 'status': 'safe'."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_ensure_a.return_value = (mock_model, MagicMock(), 0.71)
        mock_predict.return_value = {"label": "clean", "p_abuse": 0.05}

        from model_b_generation.gating import moderate
        result = moderate("hello", "en")

        assert result == {"status": "safe"}
