
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "25.0"))
MAX_AUDIO_DURATION_S = float(os.getenv("MAX_AUDIO_DURATION_S", "300.0"))
STORE_RAW_AUDIO = os.getenv("STORE_RAW_AUDIO", "0") == "1"
DB_PATH = os.getenv("DB_PATH", "data/civitas.db")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "50"))
FEEDBACK_TABLE = "feedback"
HISTORY_TABLE = "analysis_history"
