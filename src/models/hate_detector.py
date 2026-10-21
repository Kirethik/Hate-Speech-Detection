"""
Civitas AI — Model A Hate Speech Detector Interface

Provides `detect_hate_speech(text)` adhering to the canonical contract.
Seamlessly routes to the trained XLM-RoBERTa multi-task detector when installed,
or provides a deterministic, transparent demonstration result when weights are pending.
"""

from typing import Optional, List
from src.models.interfaces import DetectionResult, SeverityResult, TargetResult, RationaleSpan
from src.models.model_loader import get_detector_model


def _mock_demo_detect(text: str, language_hint: str) -> DetectionResult:
    """
    Deterministic demonstration detection for benchmark cases when weights are pending.
    Clearly flags is_demo_mode=True so mock outputs are never misrepresented.
    """
    text_lower = text.lower()
    
    # 1. Obvious identity attack / dehumanization benchmark pattern
    if any(h in text_lower for h in ["plague", "vermin", "infestation", "cockroaches", "scum", "subhuman"]):
        rationale_spans = []
        for word in ["plague on this country", "plague", "vermin", "cockroaches", "scum", "subhuman"]:
            idx = text_lower.find(word)
            if idx != -1:
                rationale_spans.append(RationaleSpan(
                    start_char=idx,
                    end_char=idx + len(word),
                    text=text[idx:idx + len(word)],
                    score=0.94
                ))
                break

        return DetectionResult(
            text=text,
            is_hate=True,
            hate_prob=0.89,
            threshold=0.50,
            severity=SeverityResult(
                label="hate",
                probs={"normal": 0.03, "offensive_profanity": 0.08, "hate": 0.89}
            ),
            target=TargetResult(
                label="ethnicity",
                probs={"ethnicity": 0.82, "religion": 0.10, "none": 0.08}
            ),
            rationale=rationale_spans,
            decision_path="base+lexicon",
            second_stage={"nli_ran": True, "nli_score": 0.86, "dehumanization_hits": ["plague"]},
            language=language_hint or "en",
            model_identifier="xlm-roberta-base (Demo Mode)",
            is_demo_mode=True,
            warning="Model A weights not installed. Displaying calibrated demonstration result."
        )

    # 2. Profanity / strong criticism WITHOUT hate (HateCheck style counter-test)
    if any(p in text_lower for p in ["damn", "hell", "fuck", "annoying", "pissed", "ridiculous", "bad", "terrible", "disagree"]):
        profanity_span = []
        for w in ["damn", "fuck", "ridiculous", "annoying"]:
            idx = text_lower.find(w)
            if idx != -1:
                profanity_span.append(RationaleSpan(
                    start_char=idx,
                    end_char=idx + len(w),
                    text=text[idx:idx + len(w)],
                    score=0.62
                ))
        return DetectionResult(
            text=text,
            is_hate=False,
            hate_prob=0.15,
            threshold=0.50,
            severity=SeverityResult(
                label="offensive_profanity",
                probs={"normal": 0.20, "offensive_profanity": 0.65, "hate": 0.15}
            ),
            target=TargetResult(
                label="none",
                probs={"none": 0.90, "other": 0.10}
            ),
            rationale=profanity_span,
            decision_path="base",
            second_stage={"nli_ran": False, "nli_score": None, "dehumanization_hits": []},
            language=language_hint or "en",
            model_identifier="xlm-roberta-base (Demo Mode)",
            is_demo_mode=True,
            warning="Model A weights not installed. Profanity / disagreement classified as non-hate in Demo Mode."
        )

    # 3. Safe / constructive input
    return DetectionResult(
        text=text,
        is_hate=False,
        hate_prob=0.04,
        threshold=0.50,
        severity=SeverityResult(
            label="normal",
            probs={"normal": 0.93, "offensive_profanity": 0.05, "hate": 0.02}
        ),
        target=TargetResult(
            label="none",
            probs={"none": 0.97}
        ),
        rationale=[],
        decision_path="base",
        second_stage={"nli_ran": False, "nli_score": None, "dehumanization_hits": []},
        language=language_hint or "en",
        model_identifier="xlm-roberta-base (Demo Mode)",
        is_demo_mode=True,
        warning="Model A weights not installed. Safe input classified as non-hate in Demo Mode."
    )


def detect_hate_speech(text: str, language_hint: Optional[str] = "en") -> DetectionResult:
    """
    Standard interface function:
    1. Attempts to run through real Model A if weights are present.
    2. Falls back to explicit, deterministic demonstration mode if absent.
    """
    model_a, is_installed, status_msg = get_detector_model()

    if not is_installed or model_a is None:
        return _mock_demo_detect(text, language_hint or "en")

    try:
        from infer import score_batch
        from dehumanization import is_dehumanizing_claim, explain as explain_dehumanization

        batch_res = score_batch(model_a, [text])[0]
        spans = [
            RationaleSpan(
                start_char=s["start_char"],
                end_char=s["end_char"],
                text=s["text"],
                score=s["score"]
            )
            for s in batch_res.get("rationale", [])
        ]

        decision_path = "base"
        dehum_hits = []
        is_hate = batch_res["is_hate"]
        hate_prob = batch_res["hate_prob"]

        # Check dehumanization lexicon
        if is_dehumanizing_claim(text) or is_dehumanizing_claim(batch_res["normalized_text"]):
            decision_path = "base+lexicon"
            dehum_hits.append("lexicon_match")
            is_hate = True
            hate_prob = max(hate_prob, 0.85)

        return DetectionResult(
            text=text,
            is_hate=is_hate,
            hate_prob=hate_prob,
            threshold=model_a.threshold,
            severity=SeverityResult(
                label=batch_res["severity"]["label"],
                probs=batch_res["severity"]["probs"]
            ),
            target=TargetResult(
                label=batch_res["target"]["label"],
                probs=batch_res["target"]["probs"]
            ),
            rationale=spans,
            decision_path=decision_path,
            second_stage={"nli_ran": False, "nli_score": None, "dehumanization_hits": dehum_hits},
            language=language_hint or "en",
            model_identifier="xlm-roberta-base",
            is_demo_mode=False
        )
    except Exception as e:
        demo = _mock_demo_detect(text, language_hint or "en")
        demo.warning = f"Model A error: {str(e)}. Operating in fallback Demo Mode."
        return demo
