export interface RationaleSpan {
  start_char: number;
  end_char: number;
  text: string;
  score: number;
  audio_start?: number | null;
  audio_end?: number | null;
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
  is_demo_mode?: boolean;
  warning?: string | null;
  reason?: string;
}

export interface Suggestion {
  text: string;
  language: string;
  tone?: string;
  safety_check_hate_prob?: number | null;
  is_template?: boolean;
  is_masked?: boolean;
}

export interface Suggestions {
  rewrite: Suggestion[];
  respond: Suggestion[];
  fallback: boolean;
  safety_status?: string;
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
  analysis_id?: string;
  input: InputInfo;
  detection: DetectionResult;
  suggestions: Suggestions;
  timing_ms: TimingMs;
  system_status?: Record<string, ModelStatus>;
}

export interface ModelStatus {
  name: string;
  model_type: string;
  is_installed: boolean;
  weights_path: string;
  device: string;
  is_demo_mode: boolean;
  status_message: string;
}

export interface RewriteResult {
  original_text: string;
  requested_tone: string;
  language: string;
  rewrites: Suggestion[];
  responses: Suggestion[];
  safety_status: string;
  model_identifier: string;
  is_demo_mode: boolean;
  warning?: string | null;
}

export interface AnalyticsData {
  total_analyses: number;
  hateful_detected: number;
  clean_detected: number;
  hate_ratio: number;
  language_distribution: Record<string, number>;
  total_feedback_submissions: number;
}

export interface FeedbackSubmission {
  analysis_id: string;
  correct_label?: boolean;
  bad_suggestion?: string;
  note?: string;
  rating?: string;
  accepted_rewrite?: string;
}

export interface HistoryItem {
  id: string;
  text: string;
  language: string;
  verdict: string;
  hate_prob: number;
  created_at: string;
  result?: AnalysisResult;
}
