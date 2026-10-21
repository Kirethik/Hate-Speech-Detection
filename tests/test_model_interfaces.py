"""
Unit tests for Model A and Model B interfaces and contracts.
Verifies that both models adhere to the canonical schema independently of trained weights.
"""

import pytest
from src.models.interfaces import DetectionResult, AlternativeResult, ModelStatus
from src.models.hate_detector import detect_hate_speech
from src.models.alternative_generator import generate_alternative, SUPPORTED_TONES
from src.models.model_loader import get_system_model_status


def test_detector_contract_hateful():
    res = detect_hate_speech("you people are a plague on this country", language_hint="en")
    assert isinstance(res, DetectionResult)
    assert res.is_hate is True
    assert res.hate_prob > 0.5
    assert res.severity.label in ["hate", "offensive_profanity", "normal"]
    assert res.target.label in ["ethnicity", "religion", "gender", "caste", "none", "other"]
    assert isinstance(res.rationale, list)
    assert len(res.rationale) > 0
    assert res.rationale[0].start_char >= 0
    assert res.rationale[0].end_char > res.rationale[0].start_char
    assert res.is_demo_mode is True or res.is_demo_mode is False


def test_detector_contract_profanity_counter_test():
    # Profanity without identity-directed hate must not be classified as hate speech
    res = detect_hate_speech("this is damn annoying and ridiculous", language_hint="en")
    assert isinstance(res, DetectionResult)
    assert res.is_hate is False
    assert res.severity.label == "offensive_profanity"


def test_detector_contract_safe():
    res = detect_hate_speech("have a wonderful day and thank you", language_hint="en")
    assert isinstance(res, DetectionResult)
    assert res.is_hate is False
    assert res.hate_prob < 0.2


def test_generator_contract_all_tones():
    test_input = "you people are ruining everything"
    for tone in SUPPORTED_TONES:
        res = generate_alternative(test_input, tone=tone, language="en")
        assert isinstance(res, AlternativeResult)
        assert res.original_text == test_input
        assert res.requested_tone == tone
        assert len(res.rewrites) > 0
        assert res.rewrites[0].tone == tone
        assert len(res.responses) > 0
        assert res.safety_status in ["verified_safe", "fallback_template", "fallback_masked"]


def test_model_loader_status():
    status = get_system_model_status()
    assert "model_a" in status
    assert "model_b" in status
    assert isinstance(status["model_a"], ModelStatus)
    assert isinstance(status["model_b"], ModelStatus)
    assert status["model_a"].model_type == "hate_detector"
    assert status["model_b"].model_type == "alternative_generator"
