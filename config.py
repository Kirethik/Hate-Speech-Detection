import os

# Model Checkpoints
MODEL_A_CHECKPOINT = os.getenv("MODEL_A_CHECKPOINT", "civitas_model_a")
MODEL_B_CHECKPOINT_DIR = os.getenv("MODEL_B_CHECKPOINT_DIR", "civitas_model_b")

# Model Names
MODEL_A_NAME = os.getenv("MODEL_A_NAME", "xlm-roberta-base")
MODEL_NLI_NAME = os.getenv("MODEL_NLI_NAME", "mDeBERTa-v3-base-mnli-xnli")
MODEL_B_NAME = os.getenv("MODEL_B_NAME", "mt0-base")

# Supported Languages
SUPPORTED_LANGUAGES = ["en", "hi", "ur_roman", "ta", "te", "ml"]

# Thresholds
NLI_BAND_LOW = float(os.getenv("NLI_BAND_LOW", "0.25"))
NLI_BAND_HIGH = float(os.getenv("NLI_BAND_HIGH", "0.85"))
NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", "0.40"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.71"))

# Server Config
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Limits
MAX_AUDIO_FILE_SIZE_MB = int(os.getenv("MAX_AUDIO_FILE_SIZE_MB", "10"))
MAX_AUDIO_DURATION_SECONDS = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "30"))

# Data Config
DATA_CONFIG_PATH = os.getenv("DATA_CONFIG_PATH", "data")

# Hardware/VRAM limits
VRAM_HEADROOM_LIMIT_MB = int(os.getenv("VRAM_HEADROOM_LIMIT_MB", "1024"))

import os
MODEL_B_TEMPERATURE = float(os.getenv("MODEL_B_TEMPERATURE", "0.65"))
MODEL_B_TOP_P = float(os.getenv("MODEL_B_TOP_P", "0.9"))
MODEL_B_REPETITION_PENALTY = float(os.getenv("MODEL_B_REPETITION_PENALTY", "1.2"))
MODEL_B_MAX_OUTPUT_LENGTH = int(os.getenv("MODEL_B_MAX_OUTPUT_LENGTH", "64"))
MODEL_B_SAFETY_THRESHOLD = float(os.getenv("MODEL_B_SAFETY_THRESHOLD", "0.5"))
MODEL_B_CANDIDATES = int(os.getenv("MODEL_B_CANDIDATES", "4"))
