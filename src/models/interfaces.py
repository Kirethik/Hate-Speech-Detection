"""
Civitas AI — Core Model Interfaces and Contracts

Defines the canonical data transfer objects and stable interfaces for Model A
(Hate Speech Detector) and Model B (Constructive Alternative Generator).
All application layers (API, CLI, UI, and evaluations) adhere to this contract.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class RationaleSpan:
    start_char: int
    end_char: int
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "score": round(self.score, 4),
        }


@dataclass
class SeverityResult:
    label: str  # "normal", "offensive_profanity", "hate"
    probs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "probs": {k: round(v, 4) for k, v in self.probs.items()},
        }


@dataclass
class TargetResult:
    label: str  # e.g., "religion", "gender", "caste", "ethnicity", "disability", "none"
    probs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "probs": {k: round(v, 4) for k, v in self.probs.items()},
        }


@dataclass
class DetectionResult:
    text: str
    is_hate: bool
    hate_prob: float
    threshold: float
    severity: SeverityResult
    target: TargetResult
    rationale: List[RationaleSpan] = field(default_factory=list)
    decision_path: str = "base"  # "base", "base+lexicon", "base+nli", "mock_demo"
    second_stage: Dict[str, Any] = field(default_factory=lambda: {
        "nli_ran": False,
        "nli_score": None,
        "dehumanization_hits": []
    })
    language: str = "en"
    model_identifier: str = "xlm-roberta-base"
    is_demo_mode: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "is_hate": self.is_hate,
            "hate_prob": round(self.hate_prob, 4),
            "threshold": round(self.threshold, 4),
            "severity": self.severity.to_dict(),
            "target": self.target.to_dict(),
            "rationale": [r.to_dict() for r in self.rationale],
            "decision_path": self.decision_path,
            "second_stage": self.second_stage,
            "language": self.language,
            "model_identifier": self.model_identifier,
            "is_demo_mode": self.is_demo_mode,
            "warning": self.warning,
        }


@dataclass
class AlternativeItem:
    text: str
    tone: str
    language: str
    safety_check_hate_prob: Optional[float] = None
    is_template: bool = False
    is_masked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tone": self.tone,
            "language": self.language,
            "safety_check_hate_prob": round(self.safety_check_hate_prob, 4) if self.safety_check_hate_prob is not None else None,
            "is_template": self.is_template,
            "is_masked": self.is_masked,
        }


@dataclass
class AlternativeResult:
    original_text: str
    requested_tone: str
    language: str
    rewrites: List[AlternativeItem] = field(default_factory=list)
    responses: List[AlternativeItem] = field(default_factory=list)
    safety_status: str = "verified_safe"  # "verified_safe", "fallback_template", "fallback_masked", "review_required"
    model_identifier: str = "bigscience/mt0-base"
    is_demo_mode: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "requested_tone": self.requested_tone,
            "language": self.language,
            "rewrites": [r.to_dict() for r in self.rewrites],
            "responses": [r.to_dict() for r in self.responses],
            "safety_status": self.safety_status,
            "model_identifier": self.model_identifier,
            "is_demo_mode": self.is_demo_mode,
            "warning": self.warning,
        }


@dataclass
class ModelStatus:
    name: str
    model_type: str  # "hate_detector" or "alternative_generator"
    is_installed: bool
    weights_path: str
    device: str
    is_demo_mode: bool
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "is_installed": self.is_installed,
            "weights_path": self.weights_path,
            "device": self.device,
            "is_demo_mode": self.is_demo_mode,
            "status_message": self.status_message,
        }
