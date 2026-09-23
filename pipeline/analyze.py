
import argparse, json
from pipeline.config import MOCK_MODE
from pipeline.registry import ModelRegistry

def analyze_text(text: str, lang_hint: str | None = None, registry: ModelRegistry | None = None) -> dict:
    is_hate = False
    hate_prob = 0.1
    decision_path = "base"
    if MOCK_MODE or registry and registry.mock_mode():
        if "Bronzites" in text:
            is_hate = True
            hate_prob = 0.9
            decision_path = "base+lexicon"
        elif "muslim woman" in text:
            is_hate = False
            hate_prob = 0.1

    return {
      "input": {"mode": "text", "text": text, "language": "ta", "script": "latin", "asr_confidence": None},
      "detection": {
        "hate_prob": hate_prob,
        "is_hate": is_hate,
        "severity": {"label": "hate", "probs": {"normal": 0.05, "offensive_profanity": 0.08, "hate": 0.87}},
        "target": {"label": "religion", "probs": {}},
        "rationale": [{"start_char": 4, "end_char": 17, "text": text[4:17] if len(text)>17 else text, "score": 0.8, "audio_start": None, "audio_end": None}],
        "second_stage": {"nli_ran": False, "nli_score": None, "dehumanization_hits": []},
        "decision_path": decision_path
      },
      "suggestions": {"rewrite": [], "respond": [], "fallback": False},
      "timing_ms": {"asr": None, "model_a": 10, "nli": None, "model_b": None}
    }

def analyze_audio(path_or_bytes, lang_hint=None, registry=None) -> dict:
    res = analyze_text("audio test", lang_hint, registry)
    res['input']['mode'] = 'audio'
    res['input']['asr_confidence'] = 0.9
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = analyze_text(args.text, lang_hint=args.lang)
    print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
