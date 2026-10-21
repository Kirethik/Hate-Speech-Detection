import {
  AnalysisResult, RewriteResult, AnalyticsData, FeedbackSubmission, ModelStatus, HistoryItem
} from './types';

// Same-origin by default: Vite proxies /api in development; one domain serves both in production.
// Set VITE_API_URL only when the API lives on a different host.
const API_BASE: string = (import.meta as any).env?.VITE_API_URL ?? '';

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error('Cannot reach the Civitas server. Start the API (scripts\\run_dev.ps1) and try again.');
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
    } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const apiClient = {
  analyzeText: (text: string, langHint?: string, tone?: string): Promise<AnalysisResult> =>
    call('/api/analyze/text', json({ text, lang_hint: langHint, tone: tone || 'calm' })),

  analyzeAudio: (file: File, langHint?: string, tone?: string): Promise<AnalysisResult> => {
    const fd = new FormData();
    fd.append('file', file);
    if (langHint) fd.append('lang_hint', langHint);
    if (tone) fd.append('tone', tone);
    return call('/api/analyze/audio', { method: 'POST', body: fd });
  },

  rewriteText: (text: string, tone = 'calm', language?: string, target?: string): Promise<RewriteResult> =>
    call('/api/rewrite', json({ text, tone, language, target })),

  submitFeedback: (feedback: FeedbackSubmission): Promise<{ feedback_id: string }> =>
    call('/api/feedback', json(feedback)),

  fetchAnalytics: (): Promise<AnalyticsData> => call('/api/analytics'),

  fetchModelStatus: (): Promise<Record<string, ModelStatus>> => call('/api/model-status'),

  fetchHistory: (limit = 20): Promise<HistoryItem[]> => call(`/api/history?limit=${limit}`),

  fetchEvaluation: (): Promise<EvalReport | null> =>
    call<EvalReport>('/api/v1/eval').catch(() => null),
};

export interface EvalMetrics {
  n: number;
  accuracy: number;
  macro_f1: number;
  precision: number;
  recall: number;
  fpr: number;
  tp: number; fp: number; tn: number; fn: number;
}

export interface EvalReport {
  created_at: string;
  split: string;
  per_language_limit: number;
  threshold: number;
  with_llm: boolean;
  llm_model?: string | null;
  classifiers: string[];
  overall: EvalMetrics;
  per_language: Record<string, EvalMetrics>;
  rewrite?: { n: number; safe_rate: number; avg_hate_prob: number } | null;
  seconds: number;
}
