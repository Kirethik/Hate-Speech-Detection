
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_B_TEMPERATURE, MODEL_B_TOP_P, MODEL_B_REPETITION_PENALTY, MODEL_B_MAX_OUTPUT_LENGTH
LANGUAGE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil',
    'te': 'Telugu', 'ml': 'Malayalam', 'ur_roman': 'Urdu'
}
FALLBACK_RESPONSES = {
    'en': 'Fallback english', 'hi': 'Fallback hindi', 'ta': 'Fallback tamil',
    'te': 'Fallback telugu', 'ml': 'Fallback malayalam', 'ur_roman': 'Fallback urdu'
}

def generate_alternatives(text, lang, task='respond'):
    lang_name = LANGUAGE_MAP.get(lang, 'English')
    if task == 'rewrite':
        prompt = f"rewrite in {lang_name}, make non-hateful: {text}"
    else:
        prompt = f"generate empathetic counter-narrative in {lang_name}: {text}"
    return prompt
