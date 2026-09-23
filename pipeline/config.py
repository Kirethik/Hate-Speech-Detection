
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

MODEL_A_ONNX_PATH = os.getenv("MODEL_A_ONNX_PATH", "artifacts/model_a/model_a_int8.onnx")
MODEL_A_PT_PATH = os.getenv("MODEL_A_PT_PATH", "checkpoints/model_a/best_model.pt")
MODEL_A_TOKENIZER = os.getenv("MODEL_A_TOKENIZER", "xlm-roberta-base")
MODEL_B_CKPT_DIR = os.getenv("MODEL_B_CKPT_DIR", "checkpoints_gen")
MODEL_B_BASE = os.getenv("MODEL_B_BASE", "bigscience/mt0-small")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
VRAM_LIMIT_GB = float(os.getenv("VRAM_LIMIT_GB", "5.5"))
FASTTEXT_LID_MODEL = os.getenv("FASTTEXT_LID_MODEL", "")
MOCK_MODE = os.getenv("CIVITAS_MOCK", "0") == "1"
ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.7"))
