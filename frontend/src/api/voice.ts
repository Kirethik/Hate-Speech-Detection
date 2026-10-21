// Client for the Civitas Voice API (/api/v1). Same-origin by default: the Vite dev
// server proxies /api to the backend, and in production one domain serves both.

const BASE: string = (import.meta as any).env?.VITE_VOICE_API ?? '';

export type Mode = 'rewrite' | 'counter';
export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface JobOptions {
  mode: Mode;
  language: string; // 'auto' or ISO code
  voiceId?: string;
  replaceOffensive: boolean;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  message?: string;
  error?: string | null;
  filename?: string;
  created_at?: number;
  position?: number;
  options?: Record<string, unknown>;
  summary?: { language?: string; duration_s?: number; edits?: number; is_hate?: boolean };
}

export interface Edit {
  seg_index: number;
  start: number;
  end: number;
  new_start: number;
  new_end: number;
  original_text: string;
  replacement_text: string;
  severity: 'offensive' | 'hate';
  target: string;
  spans: string[];
  p_classifier: number;
  method: 'tts' | 'muted';
  reason: string;
  tts_engine?: string | null;
  safety_p?: number | null;
}

export interface TranscriptLine {
  index: number;
  start: number;
  end: number;
  text: string;
  conf: number;
  p_toxic: number;
  flagged: boolean;
}

export interface JobResult {
  job_id: string;
  input: { text: string; language: string; script: string; asr_confidence: number | null };
  detection: { is_hate: boolean; hate_prob: number; flagged_sentences: number };
  audio_edit: {
    original_url: string;
    cleaned_url: string;
    duration_s: number;
    cleaned_duration_s: number;
    language: string;
    mode: Mode;
    voice_source: string | null;
    asr_model: string;
    edits: Edit[];
    transcript: TranscriptLine[];
  };
  timing_ms: Record<string, number>;
}

export interface Voice {
  id: string;
  language: string;
  text: string;
  duration_s: number;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, init);
  } catch {
    throw new Error('Cannot reach the Civitas server. Is the API running?');
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

export const voiceApi = {
  media: (url: string) => (url.startsWith('http') ? url : BASE + url),

  createJob(file: Blob, filename: string, o: JobOptions) {
    const fd = new FormData();
    fd.append('file', file, filename);
    fd.append('mode', o.mode);
    fd.append('language', o.language);
    fd.append('replace_offensive', String(o.replaceOffensive));
    if (o.voiceId) fd.append('voice_id', o.voiceId);
    return req<{ job_id: string; position: number }>('/api/v1/jobs', { method: 'POST', body: fd });
  },

  job: (id: string) => req<Job>(`/api/v1/jobs/${id}`),
  jobs: (limit = 20) => req<Job[]>(`/api/v1/jobs?limit=${limit}`),
  result: (id: string) => req<JobResult>(`/api/v1/jobs/${id}/result`),

  /** Subscribe to progress. Falls back to polling if SSE is unavailable. Returns an unsubscribe fn. */
  watch(id: string, onEvent: (j: Partial<Job>) => void): () => void {
    let closed = false;
    let timer: number | undefined;
    const poll = async () => {
      if (closed) return;
      try {
        const j = await voiceApi.job(id);
        onEvent(j);
        if (j.status === 'done' || j.status === 'failed') return;
      } catch { /* retry */ }
      timer = window.setTimeout(poll, 1000);
    };
    if (typeof EventSource === 'undefined') {
      poll();
      return () => { closed = true; clearTimeout(timer); };
    }
    const es = new EventSource(`${BASE}/api/v1/jobs/${id}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data) as Partial<Job>;
      onEvent(ev);
      if (ev.status === 'done' || ev.status === 'failed') es.close();
    };
    es.onerror = () => {
      es.close();
      if (!closed) poll();
    };
    return () => { closed = true; es.close(); clearTimeout(timer); };
  },

  prompt: (language: string) =>
    req<{ language: string; text: string }>(`/api/v1/voices/prompt?language=${language}`),

  enroll(file: Blob, filename: string, language: string, text: string) {
    const fd = new FormData();
    fd.append('file', file, filename);
    fd.append('language', language);
    fd.append('text', text);
    return req<Voice>('/api/v1/voices/enroll', { method: 'POST', body: fd });
  },

  voice: (id: string) => req<Voice>(`/api/v1/voices/${id}`),
  deleteVoice: (id: string) => req<{ deleted: string }>(`/api/v1/voices/${id}`, { method: 'DELETE' }),
  system: () => req<Record<string, any>>('/api/v1/system'),
};

export const LANGUAGES: { code: string; label: string }[] = [
  { code: 'auto', label: 'Detect automatically' },
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
  { code: 'ur', label: 'Urdu' },
  { code: 'ta', label: 'Tamil' },
  { code: 'te', label: 'Telugu' },
  { code: 'ml', label: 'Malayalam' },
  { code: 'kn', label: 'Kannada' },
  { code: 'bn', label: 'Bengali' },
  { code: 'mr', label: 'Marathi' },
];

export const langLabel = (c?: string) => LANGUAGES.find((l) => l.code === c)?.label ?? c ?? '';

export const fmtTime = (s: number) => {
  if (!isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, '0')}`;
};
