from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RationaleSpan(BaseModel):
    start_char: int
    end_char: int
    text: str
    score: float
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None


class SeverityResult(BaseModel):
    label: str
    probs: Dict[str, float] = Field(default_factory=dict)


class TargetResult(BaseModel):
    label: str
    probs: Dict[str, float] = Field(default_factory=dict)


class SecondStageResult(BaseModel):
    nli_ran: bool = False
    nli_score: Optional[float] = None
    dehumanization_hits: List[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    hate_prob: float
    is_hate: bool
    severity: SeverityResult
    target: TargetResult
    rationale: List[RationaleSpan] = Field(default_factory=list)
    second_stage: SecondStageResult = Field(default_factory=SecondStageResult)
    decision_path: str = "base"
    is_demo_mode: Optional[bool] = None
    warning: Optional[str] = None
    reason: Optional[str] = None


class Suggestion(BaseModel):
    text: str
    language: str
    tone: Optional[str] = "calm"
    safety_check_hate_prob: Optional[float] = None
    is_template: Optional[bool] = None
    is_masked: Optional[bool] = None


class Suggestions(BaseModel):
    rewrite: List[Suggestion] = Field(default_factory=list)
    respond: List[Suggestion] = Field(default_factory=list)
    fallback: bool = False
    safety_status: Optional[str] = None


class InputInfo(BaseModel):
    mode: str
    text: str
    language: str
    script: str = "latin"
    asr_confidence: Optional[float] = None


class TimingMs(BaseModel):
    asr: Optional[float] = None
    model_a: Optional[float] = None
    nli: Optional[float] = None
    model_b: Optional[float] = None


class AnalysisResult(BaseModel):
    input: InputInfo
    detection: DetectionResult
    suggestions: Suggestions
    timing_ms: TimingMs
    analysis_id: Optional[str] = None
    system_status: Optional[Dict[str, Any]] = None
    sentences: Optional[List[Dict[str, Any]]] = None  # audio: per-sentence transcript + scores


class TextAnalysisRequest(BaseModel):
    text: str
    lang_hint: Optional[str] = None
    tone: Optional[str] = "calm"


class RewriteRequest(BaseModel):
    text: str
    tone: Optional[str] = "calm"
    language: Optional[str] = "en"
    target: Optional[str] = None


class SuggestRequest(BaseModel):
    text: str
    task: str = "rewrite"
    lang: str = "en"
    tone: Optional[str] = "calm"


class FeedbackRequest(BaseModel):
    analysis_id: str
    correct_label: Optional[bool] = None
    bad_suggestion: Optional[str] = None
    note: Optional[str] = None
    rating: Optional[str] = "helpful"
    accepted_rewrite: Optional[str] = None
