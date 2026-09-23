import { AnalysisResult } from './types';

// Mock mode fixture (matches canonical JSON exactly)
const MOCK_RESPONSE: AnalysisResult = {
  analysis_id: 'mock-123',
  input: {
    mode: 'text',
    text: 'You people are a plague on this country.',
    language: 'en',
    script: 'latin',
    asr_confidence: null,
  },
  detection: {
    hate_prob: 0.87,
    is_hate: true,
    severity: {
      label: 'hate',
      probs: { normal: 0.05, offensive_profanity: 0.08, hate: 0.87 },
    },
    target: { label: 'other', probs: {} },
    rationale: [
      {
        start_char: 17,
        end_char: 23,
        text: 'plague',
        score: 0.92,
        audio_start: null,
        audio_end: null,
      },
    ],
    second_stage: {
      nli_ran: true,
      nli_score: 0.89,
      dehumanization_hits: ['plague'],
    },
    decision_path: 'base+nli+lexicon',
  },
  suggestions: {
    rewrite: [
      { text: 'I strongly disagree with your impact on this country.', language: 'en', safety_check_hate_prob: 0.02 },
    ],
    respond: [
      { text: 'It is important to remember that people are not diseases. We should discuss issues without resorting to dehumanizing language.', language: 'en', safety_check_hate_prob: 0.01 },
    ],
    fallback: false,
  },
  timing_ms: { asr: null, model_a: 45, nli: 120, model_b: 450 },
};

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const IS_MOCK = import.meta.env.VITE_MOCK === '1';

export const apiClient = {
  analyzeText: async (text: string, langHint?: string): Promise<AnalysisResult> => {
    if (IS_MOCK) {
      await new Promise(r => setTimeout(r, 800)); // simulate latency
      return { ...MOCK_RESPONSE, input: { ...MOCK_RESPONSE.input, text, language: langHint || 'en' } };
    }

    const res = await fetch(`${API_BASE}/api/analyze/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lang_hint: langHint }),
    });

    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  analyzeAudio: async (file: File, langHint?: string): Promise<AnalysisResult> => {
    if (IS_MOCK) {
      await new Promise(r => setTimeout(r, 1500));
      return { ...MOCK_RESPONSE, input: { ...MOCK_RESPONSE.input, mode: 'audio' } };
    }

    const formData = new FormData();
    formData.append('file', file);
    if (langHint) formData.append('lang_hint', langHint);

    const res = await fetch(`${API_BASE}/api/analyze/audio`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
};
