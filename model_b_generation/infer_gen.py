"""
Inference for Model B — counter-narrative generation.

Given text that Model A has flagged as hateful, produces alternate,
non-hateful sentence suggestions using a QLoRA-fine-tuned mT5-small.

Post-generation safety: every candidate is run back through Model A's
own hate head. If it scores as hateful, it is discarded and either
regenerated with a lower temperature or replaced with a hand-written
safe template response.

Usage:
    python -m model_b_generation.infer_gen --text "some hateful text" --language en
    python -m model_b_generation.infer_gen --text "..." --language hi --n 3 --style factual
"""

import sys
import os
import argparse
import difflib
import logging
import random
import warnings

# Silence HF symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Add repo root to path for Model A imports
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from infer import load_model as load_model_a, predict as predict_a
except ImportError:
    warnings.warn(
        "Could not import Model A's infer.py. Safety checks will be disabled."
    )
    load_model_a, predict_a = None, None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decoding constants (Section 6 of the blueprint — mandatory, not optional)
# ---------------------------------------------------------------------------
TEMPERATURE = 0.65
TOP_P = 0.9
REPETITION_PENALTY = 1.2
MAX_OUTPUT_LENGTH = 64

# Language code → full name mapping (must match dataset_gen.py)
LANGUAGE_MAP = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

# ---------------------------------------------------------------------------
# Last-resort safety net: hand-written safe template responses per language.
# Used only when Model B's generation is itself flagged hateful by Model A
# after a retry at lower temperature. 3-5 per language as specified.
# ---------------------------------------------------------------------------
FALLBACK_RESPONSES = {
    "en": [
        "I understand your frustration, but let's try to express ourselves "
        "without targeting any group. We can disagree respectfully.",
        "Every person deserves to be treated with dignity. Let's focus on "
        "constructive dialogue instead.",
        "Words can have a lasting impact. Let's choose language that brings "
        "people together rather than tearing them apart.",
        "I hear you, but using hurtful language doesn't help anyone. "
        "Can we find a kinder way to express this?",
    ],
    "hi": [
        "मैं आपकी भावनाओं को समझता/समझती हूँ, लेकिन कृपया ऐसी भाषा का "
        "प्रयोग करें जो किसी को ठेस न पहुँचाए।",
        "हर इंसान सम्मान का हकदार है। आइए, हम रचनात्मक बातचीत करें।",
        "शब्दों में बहुत ताकत होती है। आइए ऐसे शब्द चुनें जो लोगों को "
        "जोड़ें, तोड़ें नहीं।",
    ],
    "ta": [
        "நான் உங்கள் கருத்தை புரிந்துகொள்கிறேன், ஆனால் யாரையும் "
        "புண்படுத்தாத வார்த்தைகளைப் பயன்படுத்துவோம்.",
        "ஒவ்வொரு மனிதனும் மரியாதைக்கு உரியவர். ஆக்கபூர்வமான "
        "உரையாடலைத் தொடருவோம்.",
        "வார்த்தைகளுக்கு மிகுந்த சக்தி உண்டு. மக்களை இணைக்கும் "
        "வார்த்தைகளைத் தேர்ந்தெடுப்போம்.",
    ],
}


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_for_inference(
    checkpoint_dir: str,
    device: str = "auto",
    model_name: str = "google/mt5-small",
):
    """
    Load the base mT5-small and merge the saved LoRA adapter on top.
    Returns (model, tokenizer) with model on the requested device.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    adapter_config = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        logger.info("Loaded LoRA adapter from %s", checkpoint_dir)
    else:
        model = base_model
        warnings.warn(
            f"No adapter found at {checkpoint_dir}. Using base model — "
            f"generation quality will be poor."
        )

    model.to(device)
    model.eval()
    return model, tokenizer


# ── Deduplication ──────────────────────────────────────────────────────────

def _is_duplicate(new_text: str, existing_texts: list[str]) -> bool:
    """True if new_text is ≥85% similar to any already-accepted output."""
    for text in existing_texts:
        ratio = difflib.SequenceMatcher(
            None, new_text.lower(), text.lower()
        ).ratio()
        if ratio > 0.85:
            return True
    return False


# ── Core generation function ───────────────────────────────────────────────

def generate_alternatives(
    text: str,
    language: str,
    style: str = "empathetic",
    n: int = 1,
    model=None,
    tokenizer=None,
    model_a=None,
) -> list[str]:
    """
    Generate `n` non-hateful alternative sentences for `text`.

    Parameters
    ----------
    text : str
        The hateful/offensive text to counter.
    language : str
        ISO code: 'en', 'hi', or 'ta'.
    style : str
        'factual' or 'empathetic' (default 'empathetic').
    n : int
        Number of distinct alternatives to return.
    model : PreTrainedModel
        The fine-tuned mT5 model (loaded via load_model_for_inference).
    tokenizer : PreTrainedTokenizer
        The mT5 tokenizer.
    model_a : tuple or None
        (model_a, tokenizer_a, threshold_a) for post-generation safety
        check. If None, safety check is skipped.

    Returns
    -------
    list[str]
        Up to `n` safe, deduplicated counter-narrative strings.
    """
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided.")

    device = next(model.parameters()).device
    lang_name = LANGUAGE_MAP.get(language, "English")

    # Task prefix exactly matching the training format (Section 4)
    prompt = (
        f"generate {style} counter-narrative in {lang_name}: {text}"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    def _do_generate(temp: float) -> list[str]:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_LENGTH,
                temperature=temp,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                num_return_sequences=min(n * 3, 10),  # over-generate, then filter
            )
        return [
            tokenizer.decode(out, skip_special_tokens=True).strip()
            for out in outputs
        ]

    # First pass: generate candidates
    candidates = _do_generate(TEMPERATURE)

    # ── Safety check via Model A ───────────────────────────────────────
    if model_a is not None and predict_a is not None:
        mod_a, tok_a, thresh_a = model_a
        device_a = next(mod_a.parameters()).device

        safe = []
        for cand in candidates:
            if not cand:
                continue
            pred = predict_a(mod_a, tok_a, thresh_a, cand, device_a)
            if pred.get("label") != "ABUSIVE":
                safe.append(cand)

        # Retry once with lower temperature if everything was flagged
        if not safe:
            logger.info(
                "All %d candidates flagged by Model A — retrying at temp=0.5",
                len(candidates),
            )
            candidates = _do_generate(0.5)
            for cand in candidates:
                if not cand:
                    continue
                pred = predict_a(mod_a, tok_a, thresh_a, cand, device_a)
                if pred.get("label") != "ABUSIVE":
                    safe.append(cand)

        candidates = safe

    # ── Deduplicate and trim to n ──────────────────────────────────────
    final = []
    for cand in candidates:
        if cand and not _is_duplicate(cand, final):
            final.append(cand)
            if len(final) >= n:
                break

    # ── Fallback to safe templates ─────────────────────────────────────
    if not final:
        logger.warning(
            "All candidates discarded — falling back to template responses "
            "for language '%s'",
            language,
        )
        fallbacks = FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["en"])
        final = random.sample(fallbacks, min(n, len(fallbacks)))

    return final


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate counter-narrative alternatives for hateful text"
    )
    parser.add_argument("--text", type=str, required=True,
                        help="The hateful text to generate alternatives for")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "hi", "ta"])
    parser.add_argument("--style", type=str, default="empathetic",
                        choices=["factual", "empathetic"])
    parser.add_argument("--n", type=int, default=3,
                        help="Number of alternatives to generate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_gen",
                        help="Path to the saved LoRA adapter directory")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_safety_check", action="store_true",
                        help="Skip the Model A safety check on outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_b, tokenizer_b = load_model_for_inference(
        args.checkpoint_dir, args.device
    )

    model_a_tuple = None
    if not args.no_safety_check and load_model_a is not None:
        ma_path = os.path.join(_REPO_ROOT, "checkpoints", "model_a", "best_model.pt")
        if os.path.exists(ma_path):
            device_a = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_a_tuple = load_model_a(ma_path, device_a)
            logger.info("Model A loaded for safety check (threshold=%.2f)",
                        model_a_tuple[2])
        else:
            warnings.warn(
                f"Model A checkpoint not found at {ma_path}. "
                "Skipping safety check."
            )

    results = generate_alternatives(
        text=args.text,
        language=args.language,
        style=args.style,
        n=args.n,
        model=model_b,
        tokenizer=tokenizer_b,
        model_a=model_a_tuple,
    )

    print(f"\nGenerated {len(results)} alternative(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")


if __name__ == "__main__":
    main()
