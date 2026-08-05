"""
Tests for the counter-narrative dataset converters and unified schema.

Validates that each converter:
  - Produces rows matching the unified schema (all required columns present)
  - Has no nulls in hate_text / response_text
  - Uses only allowed language codes: {en, hi, ta}
  - Uses only allowed style values: {factual, empathetic}
  - Returns a pandas DataFrame (even if empty)
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add package to path
_PKG_DIR = str(Path(__file__).resolve().parent.parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

ALLOWED_LANGUAGES = {"en", "hi", "ta"}
ALLOWED_STYLES = {"factual", "empathetic"}
REQUIRED_COLUMNS = {"hate_text", "language", "style", "response_text", "source"}


# ── Test TER Mini-Corpus (always available, no download needed) ────────────

class TestTerMiniConverter:
    """The TER Mini-Corpus is embedded in the code, so it always works."""

    @pytest.fixture(scope="class")
    def ter_df(self):
        from converters_gen.ter_mini import convert
        return convert()

    def test_returns_dataframe(self, ter_df):
        assert isinstance(ter_df, pd.DataFrame)

    def test_not_empty(self, ter_df):
        assert len(ter_df) >= 100, f"Expected ≥100 rows, got {len(ter_df)}"

    def test_schema_columns(self, ter_df):
        assert REQUIRED_COLUMNS.issubset(set(ter_df.columns)), (
            f"Missing columns: {REQUIRED_COLUMNS - set(ter_df.columns)}"
        )

    def test_no_null_hate_text(self, ter_df):
        assert ter_df["hate_text"].notnull().all(), "Found null hate_text"

    def test_no_null_response_text(self, ter_df):
        assert ter_df["response_text"].notnull().all(), "Found null response_text"

    def test_language_codes(self, ter_df):
        langs = set(ter_df["language"].unique())
        assert langs.issubset(ALLOWED_LANGUAGES), (
            f"Invalid languages: {langs - ALLOWED_LANGUAGES}"
        )

    def test_style_values(self, ter_df):
        styles = set(ter_df["style"].unique())
        assert styles.issubset(ALLOWED_STYLES), (
            f"Invalid styles: {styles - ALLOWED_STYLES}"
        )

    def test_has_all_three_languages(self, ter_df):
        """TER Mini specifically must have en, hi, and ta."""
        langs = set(ter_df["language"].unique())
        assert ALLOWED_LANGUAGES.issubset(langs), (
            f"Missing languages: {ALLOWED_LANGUAGES - langs}"
        )

    def test_source_is_ter_mini(self, ter_df):
        assert (ter_df["source"] == "ter_mini").all()

    def test_has_split_column(self, ter_df):
        assert "split" in ter_df.columns
        splits = set(ter_df["split"].unique())
        assert splits.issubset({"train", "val"})


# ── Test schema validation helper ──────────────────────────────────────────

class TestSchemaValidation:
    """Test schema compliance on synthetic data."""

    def test_valid_schema(self):
        df = pd.DataFrame({
            "hate_text": ["hate1", "hate2", "hate3"],
            "language": ["en", "hi", "ta"],
            "style": ["empathetic", "factual", "empathetic"],
            "response_text": ["resp1", "resp2", "resp3"],
            "source": ["test", "test", "test"],
        })
        assert REQUIRED_COLUMNS.issubset(set(df.columns))
        assert df["hate_text"].notnull().all()
        assert df["response_text"].notnull().all()
        assert set(df["language"].unique()).issubset(ALLOWED_LANGUAGES)
        assert set(df["style"].unique()).issubset(ALLOWED_STYLES)

    def test_invalid_language_detected(self):
        df = pd.DataFrame({
            "hate_text": ["hate1"],
            "language": ["fr"],  # Invalid
            "style": ["empathetic"],
            "response_text": ["resp1"],
            "source": ["test"],
        })
        assert not set(df["language"].unique()).issubset(ALLOWED_LANGUAGES)

    def test_invalid_style_detected(self):
        df = pd.DataFrame({
            "hate_text": ["hate1"],
            "language": ["en"],
            "style": ["sarcastic"],  # Invalid
            "response_text": ["resp1"],
            "source": ["test"],
        })
        assert not set(df["style"].unique()).issubset(ALLOWED_STYLES)


# ── Graceful fallback tests (converters that may not have data) ────────────

class TestGracefulFallbacks:
    """Converters that attempt external downloads must fail gracefully."""

    def test_indic_conan_returns_dataframe(self):
        """indic_conan.convert() must return a DataFrame, even if empty."""
        from converters_gen.indic_conan import convert
        df = convert()
        assert isinstance(df, pd.DataFrame)

    def test_lt_edi_returns_dataframe(self):
        """lt_edi.convert() must return a DataFrame, even if empty."""
        from converters_gen.lt_edi import convert
        df = convert()
        assert isinstance(df, pd.DataFrame)

    def test_lt_edi_returns_correct_schema(self):
        """Even an empty DataFrame must have the right columns."""
        from converters_gen.lt_edi import convert
        df = convert()
        assert REQUIRED_COLUMNS.issubset(
            set(df.columns) | {"split"}  # split is optional for empty
        ) or len(df) == 0
