"""
Single source of truth for runtime settings. Every value can be overridden by
an environment variable of the same name. Import from here; do not redefine
settings in sub-packages.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Model checkpoints and names
# --------------------------------------------------------------------------- #
# Model A: the .pt written by train.py (model_state_dict + threshold + class lists)
MODEL_A_PT_PATH = os.getenv("MODEL_A_PT_PATH", "artifacts/model_a/best_model.pt")
MODEL_A_ONNX_PATH = os.getenv("MODEL_A_ONNX_PATH", "artifacts/model_a/model_a_int8.onnx")
MODEL_A_NAME = os.getenv("MODEL_A_NAME", "xlm-roberta-base")

# NLI second stage
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
NLI_CALIBRATION_PATH = os.getenv("NLI_CALIBRATION_PATH", "artifacts/model_a/nli_calibration.json")

# Model B: LoRA adapter directory + the base model it was trained on
MODEL_B_CKPT_DIR = os.getenv("MODEL_B_CKPT_DIR", "artifacts/model_b")
MODEL_B_BASE = os.getenv("MODEL_B_BASE", "bigscience/mt0-base")

# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
SUPPORTED_LANGUAGES = ["en", "hi", "ur_roman", "ta", "te", "ml"]

# --------------------------------------------------------------------------- #
# Model A decision thresholds (fallbacks; calibrated values override them)
# --------------------------------------------------------------------------- #
NLI_BAND_LOW = float(os.getenv("NLI_BAND_LOW", "0.25"))
NLI_BAND_HIGH = float(os.getenv("NLI_BAND_HIGH", "0.85"))
NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", "0.40"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# --------------------------------------------------------------------------- #
# Model B decoding + safety gate
# --------------------------------------------------------------------------- #
MODEL_B_TEMPERATURE = float(os.getenv("MODEL_B_TEMPERATURE", "0.65"))
MODEL_B_TOP_P = float(os.getenv("MODEL_B_TOP_P", "0.9"))
MODEL_B_REPETITION_PENALTY = float(os.getenv("MODEL_B_REPETITION_PENALTY", "1.2"))
MODEL_B_MAX_OUTPUT_LENGTH = int(os.getenv("MODEL_B_MAX_OUTPUT_LENGTH", "64"))
MODEL_B_SAFETY_THRESHOLD = float(os.getenv("MODEL_B_SAFETY_THRESHOLD", "0.5"))
MODEL_B_CANDIDATES = int(os.getenv("MODEL_B_CANDIDATES", "4"))

# --------------------------------------------------------------------------- #
# Speech
# --------------------------------------------------------------------------- #
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.7"))
FASTTEXT_LID_MODEL = os.getenv("FASTTEXT_LID_MODEL", "")

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
VRAM_LIMIT_GB = float(os.getenv("VRAM_LIMIT_GB", "5.5"))


def mock_mode() -> bool:
    """Read at call time so tests (and the server) can toggle it via env."""
    return os.getenv("CIVITAS_MOCK", "0") == "1"


MOCK_MODE = mock_mode()

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "25.0"))
MAX_AUDIO_DURATION_S = float(os.getenv("MAX_AUDIO_DURATION_S", "300.0"))
STORE_RAW_AUDIO = os.getenv("STORE_RAW_AUDIO", "0") == "1"
DB_PATH = os.getenv("DB_PATH", "data/civitas.db")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "50"))
FEEDBACK_TABLE = "feedback"
HISTORY_TABLE = "analysis_history"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
DATA_CONFIG_PATH = os.getenv("DATA_CONFIG_PATH", "data_config.yaml")
