
from pydantic import BaseModel, Field
from typing import Optional

class RationaleSpan(BaseModel):
    start_char: int
    end_char: int
    text: str
    score: float
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None

class SeverityResult(BaseModel):
    label: str
    probs: dict[str, float]

class TargetResult(BaseModel):
    label: str
    probs: dict[str, float]

class SecondStageResult(BaseModel):
    nli_ran: bool
    nli_score: Optional[float] = None
    dehumanization_hits: list[str] = Field(default_factory=list)

class DetectionResult(BaseModel):
    hate_prob: float
    is_hate: bool
    severity: SeverityResult
    target: TargetResult
    rationale: list[RationaleSpan] = Field(default_factory=list)
    second_stage: SecondStageResult
    decision_path: str

class Suggestion(BaseModel):
    text: str
    language: str
    safety_check_hate_prob: float

class Suggestions(BaseModel):
    rewrite: list[Suggestion] = Field(default_factory=list)
    respond: list[Suggestion] = Field(default_factory=list)
    fallback: bool = False

class InputInfo(BaseModel):
    mode: str
    text: str
    language: str
    script: str
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

class TextAnalysisRequest(BaseModel):
    text: str
    lang_hint: Optional[str] = None

class SuggestRequest(BaseModel):
    text: str
    task: str
    lang: str

class FeedbackRequest(BaseModel):
    analysis_id: str
    correct_label: Optional[bool] = None
    bad_suggestion: Optional[str] = None
    note: Optional[str] = None
