export interface RationaleSpan {
  start_char: number;
  end_char: number;
  text: string;
  score: number;
  audio_start: number | null;
  audio_end: number | null;
}

export interface SeverityResult {
  label: string;
  probs: Record<string, number>;
}

export interface TargetResult {
  label: string;
  probs: Record<string, number>;
}

export interface SecondStageResult {
  nli_ran: boolean;
  nli_score: number | null;
  dehumanization_hits: string[];
}

export interface DetectionResult {
  hate_prob: number;
  is_hate: boolean;
  severity: SeverityResult;
  target: TargetResult;
  rationale: RationaleSpan[];
  second_stage: SecondStageResult;
  decision_path: string;
}

export interface Suggestion {
  text: string;
  language: string;
  safety_check_hate_prob: number;
}

export interface Suggestions {
  rewrite: Suggestion[];
  respond: Suggestion[];
  fallback: boolean;
}

export interface InputInfo {
  mode: string;
  text: string;
  language: string;
  script: string;
  asr_confidence: number | null;
}

export interface TimingMs {
  asr: number | null;
  model_a: number | null;
  nli: number | null;
  model_b: number | null;
}

export interface AnalysisResult {
  analysis_id: string;
  input: InputInfo;
  detection: DetectionResult;
  suggestions: Suggestions;
  timing_ms: TimingMs;
}
