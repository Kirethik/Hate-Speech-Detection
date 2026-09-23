
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_B_SAFETY_THRESHOLD, MODEL_B_CANDIDATES

def generate_suggestions(text: str, language: str, target_group: str,
                          model_a_tuple, model_b_tuple, cfg=None) -> dict:
    return {
        "rewrite": [],
        "respond":  [],
        "fallback": True
    }
