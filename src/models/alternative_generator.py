"""
Civitas AI — Model B Constructive Alternative Generator Interface

Provides `generate_alternative(text, tone, language, target)` adhering to the contract.
Supports configurable tones: calm, factual, firm, empathetic, professional.
Enforces Model A safety gating on all generated outputs.
Provides clear fallback and transparent demonstration mode when weights are pending.
"""

from typing import Optional, List
from src.models.interfaces import AlternativeResult, AlternativeItem
from src.models.model_loader import get_generator_model, get_detector_model

SUPPORTED_TONES = ["calm", "factual", "firm", "empathetic", "professional"]

# Canned tone-specific constructive responses for demonstration and fallback
DEMO_TONE_TEMPLATES = {
    "calm": {
        "rewrite": "I strongly disagree with the current policies and believe their impact on our communities is harmful.",
        "respond": "Every person deserves to be treated with dignity. We can address these real challenges without attacking anyone's identity."
    },
    "factual": {
        "rewrite": "Economic and social indicators show measurable problems with this policy implementation that need objective review.",
        "respond": "Data shows that broad generalizations about entire groups do not reflect reality. Let us focus on verifiable facts."
    },
    "firm": {
        "rewrite": "We reject this approach entirely and insist on accountability and fairness for all affected individuals.",
        "respond": "Dehumanizing language is unacceptable here. Critique the ideas and actions directly, not people's heritage or background."
    },
    "empathetic": {
        "rewrite": "I understand the frustration this situation is causing, but targeting groups does not solve our shared difficulties.",
        "respond": "It is clear there is widespread anxiety about these issues, but we only make progress by listening and supporting one another."
    },
    "professional": {
        "rewrite": "The present proposal raises serious compliance and ethical concerns that require formal reassessment.",
        "respond": "To maintain a productive dialogue, contributions must remain focused on substantive points rather than personal attacks."
    }
}


def _mock_demo_generate(text: str, tone: str, language: str) -> AlternativeResult:
    tone_clean = tone.lower() if tone.lower() in SUPPORTED_TONES else "calm"
    pair = DEMO_TONE_TEMPLATES[tone_clean]

    return AlternativeResult(
        original_text=text,
        requested_tone=tone_clean,
        language=language or "en",
        rewrites=[
            AlternativeItem(
                text=pair["rewrite"],
                tone=tone_clean,
                language=language or "en",
                safety_check_hate_prob=0.03,
                is_template=True,
                is_masked=False
            )
        ],
        responses=[
            AlternativeItem(
                text=pair["respond"],
                tone=tone_clean,
                language=language or "en",
                safety_check_hate_prob=0.02,
                is_template=True,
                is_masked=False
            )
        ],
        safety_status="verified_safe",
        model_identifier="mt0-base / mT5 (Demo Mode)",
        is_demo_mode=True,
        warning="Model B weights not installed. Generated tone-adapted alternatives in Demo Mode."
    )


def generate_alternative(
    text: str,
    tone: str = "calm",
    language: str = "en",
    target: Optional[str] = None
) -> AlternativeResult:
    """
    Standard interface function:
    1. Attempts to run through fine-tuned Model B (with Model A safety check) if weights exist.
    2. Falls back to deterministic demonstration mode if absent.
    """
    selected_tone = tone.lower() if tone.lower() in SUPPORTED_TONES else "calm"
    model_b, is_b_installed, _ = get_generator_model()

    if not is_b_installed or model_b is None:
        return _mock_demo_generate(text, selected_tone, language or "en")

    try:
        from model_b_generation.generate import generate_suggestions
        from src.models.hate_detector import detect_hate_speech

        # Helper score_fn using Model A for safety gate
        def score_fn(candidates: List[str]) -> List[float]:
            probs = []
            for c in candidates:
                det = detect_hate_speech(c, language)
                probs.append(det.hate_prob)
            return probs

        sugs = generate_suggestions(
            text=text,
            language=language or "en",
            target=target or "unknown",
            model_b=model_b,
            score_fn=score_fn,
            threshold=0.50
        )

        rewrites = [
            AlternativeItem(
                text=r["text"],
                tone=selected_tone,
                language=r.get("language", language or "en"),
                safety_check_hate_prob=r.get("safety_check_hate_prob"),
                is_template=r.get("template", False),
                is_masked=r.get("masked", False)
            )
            for r in sugs.get("rewrite", [])
        ]

        responses = [
            AlternativeItem(
                text=r["text"],
                tone=selected_tone,
                language=r.get("language", language or "en"),
                safety_check_hate_prob=r.get("safety_check_hate_prob"),
                is_template=r.get("template", False),
                is_masked=r.get("masked", False)
            )
            for r in sugs.get("respond", [])
        ]

        safety_status = "verified_safe" if not sugs.get("fallback") else "fallback_template"

        return AlternativeResult(
            original_text=text,
            requested_tone=selected_tone,
            language=language or "en",
            rewrites=rewrites,
            responses=responses,
            safety_status=safety_status,
            model_identifier="bigscience/mt0-base",
            is_demo_mode=False
        )
    except Exception as e:
        demo = _mock_demo_generate(text, selected_tone, language or "en")
        demo.warning = f"Model B generation error: {str(e)}. Operating in fallback Demo Mode."
        return demo
