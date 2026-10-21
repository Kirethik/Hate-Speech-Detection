import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Job, JobResult, LANGUAGES, Mode, Voice, fmtTime, langLabel, voiceApi,
} from '../../api/voice';
import EnrollVoiceDialog from './EnrollVoiceDialog';
import { useRecorder } from './useRecorder';
import Waveform, { Region } from './Waveform';
import './VoiceStudio.css';

const STAGES: { key: string; label: string }[] = [
  { key: 'decode', label: 'Read audio' },
  { key: 'asr', label: 'Transcribe' },
  { key: 'detect', label: 'Scan' },
  { key: 'judge', label: 'Review & rewrite' },
  { key: 'voice', label: 'Learn voice' },
  { key: 'tts', label: 'Re-voice' },
  { key: 'splice', label: 'Stitch' },
];
const stageIndex = (s?: string) => {
  const i = STAGES.findIndex((x) => x.key === s);
  return s === 'done' ? STAGES.length : i;
};
const TARGET_LABEL: Record<string, string> = {
  none: 'No target', religion: 'Religion', gender: 'Gender', caste_ethnicity: 'Caste / ethnicity',
  disability: 'Disability', political: 'Political', nationality_migrant: 'Nationality / migrants', other: 'Other',
};
const ACCEPT = 'audio/*,video/mp4,.wav,.mp3,.m4a,.webm,.ogg,.opus,.flac,.aac';

const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string | null) => {
    try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* private mode */ }
  },
};

type Source = { blob: Blob; filename: string; label: string; url: string };

export default function VoiceStudioView() {
  const [tab, setTab] = useState<'upload' | 'record'>('upload');
  const [source, setSource] = useState<Source | null>(null);
  const [mode, setMode] = useState<Mode>('rewrite');
  const [language, setLanguage] = useState('auto');
  const [replaceOffensive, setReplaceOffensive] = useState(true);
  const [useEnrolled, setUseEnrolled] = useState(false);
  const [voice, setVoice] = useState<Voice | null>(null);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [consent, setConsent] = useState(false);
  const [dragging, setDragging] = useState(false);

  const [job, setJob] = useState<Partial<Job> | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Job[]>([]);
  const [seek, setSeek] = useState<{ which: 'o' | 'c'; t: number; n: number } | null>(null);
  const [pauseO, setPauseO] = useState(0);
  const [pauseC, setPauseC] = useState(0);
  const unwatch = useRef<() => void>(() => {});
  const fileInput = useRef<HTMLInputElement>(null);
  const outRef = useRef<HTMLElement>(null);
  const rec = useRecorder(300);

  const refreshHistory = useCallback(() => {
    voiceApi.jobs(12).then(setHistory).catch(() => {});
  }, []);

  useEffect(() => {
    refreshHistory();
    const vid = store.get('civitas.voiceId');
    if (vid) voiceApi.voice(vid).then((v) => { setVoice(v); setUseEnrolled(true); })
      .catch(() => store.set('civitas.voiceId', null));
    return () => unwatch.current();
  }, [refreshHistory]);

  useEffect(() => () => { if (source) URL.revokeObjectURL(source.url); }, [source]);

  const pickFile = (f: File | undefined) => {
    if (!f) return;
    if (!(f.type.startsWith('audio/') || f.type.startsWith('video/') || /\.(wav|mp3|m4a|webm|ogg|opus|flac|aac)$/i.test(f.name))) {
      setError('That does not look like an audio file.');
      return;
    }
    setError(null);
    setSource({ blob: f, filename: f.name, label: `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`, url: URL.createObjectURL(f) });
  };

  const stopRecording = async () => {
    const r = await rec.stop();
    if (r) setSource({ blob: r.blob, filename: r.filename, label: `Recording · ${fmtTime(r.seconds)}`, url: URL.createObjectURL(r.blob) });
  };

  const loadResult = useCallback(async (id: string) => {
    try {
      setResult(await voiceApi.result(id));
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  const follow = useCallback((id: string) => {
    unwatch.current();
    unwatch.current = voiceApi.watch(id, (ev) => {
      setJob((j) => ({ ...j, ...ev, id }));
      if (ev.status === 'done') { loadResult(id); refreshHistory(); }
      if (ev.status === 'failed') { setError(ev.error || 'Processing failed.'); refreshHistory(); }
    });
  }, [loadResult, refreshHistory]);

  const submit = async () => {
    if (!source) return;
    setError(null);
    setResult(null);
    setJob({ status: 'queued', stage: 'queued', progress: 0 });
    if (window.innerWidth < 960) outRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const r = await voiceApi.createJob(source.blob, source.filename, {
        mode, language, replaceOffensive, voiceId: useEnrolled && voice ? voice.id : undefined,
      });
      setJob({ id: r.job_id, status: 'queued', stage: 'queued', progress: 0, position: r.position });
      follow(r.job_id);
      refreshHistory();
    } catch (e: any) {
      setJob(null);
      setError(e.message);
    }
  };

  const openJob = (j: Job) => {
    setError(null);
    setResult(null);
    setJob(j);
    if (j.status === 'done') loadResult(j.id);
    else if (j.status === 'failed') setError(j.error || 'Processing failed.');
    else follow(j.id);
  };

  const removeVoice = async () => {
    if (!voice) return;
    try { await voiceApi.deleteVoice(voice.id); } catch { /* already gone */ }
    store.set('civitas.voiceId', null);
    setVoice(null);
    setUseEnrolled(false);
  };

  const busy = job?.status === 'queued' || job?.status === 'running';
  const ae = result?.audio_edit;
  const origRegions: Region[] = useMemo(() => (ae?.edits ?? []).map((e) => ({
    start: e.start, end: e.end, kind: 'removed', label: e.original_text,
  })), [ae]);
  const cleanRegions: Region[] = useMemo(() => (ae?.edits ?? []).map((e) => ({
    start: e.new_start, end: e.new_end, kind: e.method === 'tts' ? 'replaced' : 'muted', label: e.replacement_text || 'Muted',
  })), [ae]);
  const editBySeg = useMemo(() => new Map((ae?.edits ?? []).map((e) => [e.seg_index, e])), [ae]);
  const totalSec = result ? (result.timing_ms.total ?? 0) / 1000 : 0;

  const downloadReport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `civitas-report-${result.job_id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  return (
    <div className="vs">
      <section className="vs-hero">
        <div>
          <h2>Voice Studio</h2>
          <p>
            Upload or record speech. Civitas transcribes it, finds hate speech, and re-voices only those
            sentences in the speaker's own voice. Everything else stays exactly as recorded.
          </p>
        </div>
        <ol className="vs-how" aria-label="How it works">
          <li><b>1</b>Transcribe</li>
          <li><b>2</b>Detect</li>
          <li><b>3</b>Rewrite</li>
          <li><b>4</b>Re-voice</li>
        </ol>
      </section>

      <div className="vs-grid">
        {/* ------------------------------------------------ input */}
        <section className="vs-card vs-input" aria-labelledby="vs-src">
          <h3 id="vs-src">Source audio</h3>
          <div className="vs-seg" role="tablist">
            <button role="tab" aria-selected={tab === 'upload'} className={tab === 'upload' ? 'on' : ''}
                    onClick={() => setTab('upload')} disabled={rec.recording}>Upload file</button>
            <button role="tab" aria-selected={tab === 'record'} className={tab === 'record' ? 'on' : ''}
                    onClick={() => setTab('record')}>Record</button>
          </div>

          {tab === 'upload' ? (
            <div
              className={`vs-drop ${dragging ? 'drag' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
              onClick={() => fileInput.current?.click()}
              role="button" tabIndex={0}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInput.current?.click()}
            >
              <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M12 16V4m0 0-4 4m4-4 4 4M5 16v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
              <span><b>Drop an audio file</b> or click to browse</span>
              <small>WAV, MP3, M4A, WEBM, OGG · up to 5 minutes</small>
              <input ref={fileInput} type="file" accept={ACCEPT} hidden onChange={(e) => pickFile(e.target.files?.[0])} />
            </div>
          ) : (
            <div className="vs-recorder">
              {!rec.recording ? (
                <button className="vs-btn vs-btn-primary vs-btn-lg" onClick={rec.start}>
                  <span className="vs-dot" aria-hidden="true" /> Start recording
                </button>
              ) : (
                <button className="vs-btn vs-btn-danger vs-btn-lg" onClick={stopRecording}>
                  <span className="vs-square" aria-hidden="true" /> Stop · {fmtTime(rec.seconds)}
                </button>
              )}
              <div className="vs-meter" aria-hidden="true"><span style={{ width: `${rec.level * 100}%` }} /></div>
              {rec.error && <p className="vs-error" role="alert">{rec.error}</p>}
            </div>
          )}

          {source && (
            <div className="vs-source">
              <div className="vs-source-row">
                <span className="vs-source-label">{source.label}</span>
                <button className="vs-link" onClick={() => setSource(null)}>Remove</button>
              </div>
              <audio controls src={source.url} />
            </div>
          )}

          <h3 className="vs-sub">What should replace hate speech?</h3>
          <div className="vs-choice">
            <label className={mode === 'rewrite' ? 'on' : ''}>
              <input type="radio" name="mode" checked={mode === 'rewrite'} onChange={() => setMode('rewrite')} />
              <span><b>Rewrite</b><small>Same point, without the hate. Recommended.</small></span>
            </label>
            <label className={mode === 'counter' ? 'on' : ''}>
              <input type="radio" name="mode" checked={mode === 'counter'} onChange={() => setMode('counter')} />
              <span><b>Counter-narrative</b><small>A first-person statement that challenges the hateful idea.</small></span>
            </label>
          </div>

          <div className="vs-row2">
            <label className="vs-field">
              <span>Spoken language</span>
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                {LANGUAGES.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
              </select>
            </label>
            <label className="vs-field">
              <span>Voice for replacements</span>
              <select value={useEnrolled ? 'enrolled' : 'auto'}
                      onChange={(e) => (e.target.value === 'enrolled' && !voice ? setEnrollOpen(true) : setUseEnrolled(e.target.value === 'enrolled'))}>
                <option value="auto">Learn from this recording</option>
                <option value="enrolled">{voice ? `My enrolled voice (${langLabel(voice.language)})` : 'Enroll my voice…'}</option>
              </select>
            </label>
          </div>
          {voice && (
            <p className="vs-muted vs-small">
              Enrolled voice saved · <button className="vs-link" onClick={removeVoice}>delete it</button>
            </p>
          )}

          <label className="vs-check">
            <input type="checkbox" checked={replaceOffensive} onChange={(e) => setReplaceOffensive(e.target.checked)} />
            <span>Also replace insults and profanity that are not hate speech</span>
          </label>
          <label className="vs-check">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>I am the speaker, or I have the speaker's consent to re-voice this recording.</span>
          </label>

          <button className="vs-btn vs-btn-primary vs-btn-block" disabled={!source || !consent || busy || rec.recording} onClick={submit}>
            {busy ? 'Processing…' : 'Clean this audio'}
          </button>
          {error && !busy && <p className="vs-error" role="alert">{error}</p>}
        </section>

        {/* ------------------------------------------------ output */}
        <section className="vs-card vs-output" aria-live="polite" ref={outRef}>
          {!job && !result && (
            <div className="vs-empty">
              <div className="vs-empty-art" aria-hidden="true">
                {Array.from({ length: 28 }, (_, i) => (
                  <span key={i} className={i >= 11 && i <= 16 ? 'hot' : ''}
                        style={{ height: `${18 + Math.abs(Math.sin(i * 1.7)) * 60}%` }} />
                ))}
              </div>
              <h3>Your cleaned audio appears here</h3>
              <p className="vs-muted">
                Replaced sentences are marked on both waveforms, so listeners can always tell what was edited.
              </p>
            </div>
          )}

          {busy && (
            <div className="vs-progress">
              <div className="vs-progress-head">
                <h3>{job?.status === 'queued' ? 'Waiting in queue' : 'Processing'}</h3>
                <span>{Math.round((job?.progress ?? 0) * 100)}%</span>
              </div>
              <div className="vs-bar"><span style={{ width: `${(job?.progress ?? 0) * 100}%` }} /></div>
              <p className="vs-muted">
                {job?.status === 'queued' && job.position ? `Position ${job.position} in queue. ` : ''}
                {job?.message || 'Starting…'}
              </p>
              <ol className="vs-steps">
                {STAGES.map((s, i) => {
                  const cur = stageIndex(job?.stage);
                  return (
                    <li key={s.key} className={i < cur ? 'done' : i === cur ? 'now' : ''}>
                      <span className="vs-step-dot" aria-hidden="true" />{s.label}
                    </li>
                  );
                })}
              </ol>
              <p className="vs-muted vs-small">The first run downloads and loads the models, which takes longer.</p>
            </div>
          )}

          {job?.status === 'failed' && !busy && (
            <div className="vs-failed">
              <h3>Processing failed</h3>
              <p>{error || job.error}</p>
              <button className="vs-btn vs-btn-ghost" onClick={submit} disabled={!source}>Try again</button>
            </div>
          )}

          {result && ae && (
            <div className="vs-result">
              <div className="vs-stats">
                <div><span>Sentences replaced</span><b className={ae.edits.length ? 'hot' : 'ok'}>{ae.edits.length}</b></div>
                <div><span>Language</span><b>{langLabel(ae.language)}</b></div>
                <div><span>Duration</span><b>{fmtTime(ae.duration_s)}</b></div>
                <div><span>Processed in</span><b>{totalSec.toFixed(1)}s</b></div>
              </div>

              {ae.edits.length === 0 && (
                <div className="vs-clean-banner">No hate speech found. The audio was left unchanged.</div>
              )}

              <Waveform src={voiceApi.media(ae.original_url)} title="Original" subtitle="As recorded"
                        regions={origRegions} tone="original" pauseSignal={pauseO}
                        seekTo={seek?.which === 'o' ? seek : null} onPlay={() => setPauseC((n) => n + 1)} />
              <Waveform src={voiceApi.media(ae.cleaned_url)} title="Cleaned"
                        subtitle={ae.voice_source === 'enrolled' ? 'Replacements in your enrolled voice' : 'Replacements in the speaker’s voice'}
                        regions={cleanRegions} tone="cleaned" pauseSignal={pauseC}
                        seekTo={seek?.which === 'c' ? seek : null} onPlay={() => setPauseO((n) => n + 1)} />
              <div className="vs-legend">
                <span><i className="lg-removed" />Removed</span>
                <span><i className="lg-replaced" />Re-voiced</span>
                <span><i className="lg-muted" />Muted (no safe replacement)</span>
              </div>

              <h3 className="vs-sub">Transcript</h3>
              <ol className="vs-transcript">
                {ae.transcript.map((line) => {
                  const e = editBySeg.get(line.index);
                  return (
                    <li key={line.index} className={e ? `edited sev-${e.severity}` : ''}>
                      <button className="vs-ts" onClick={() => setSeek({ which: 'o', t: line.start, n: Date.now() })}
                              aria-label={`Play original from ${fmtTime(line.start)}`}>{fmtTime(line.start)}</button>
                      <div className="vs-line" lang={ae.language} data-script={result.input.script}>
                        {e ? (
                          <>
                            <del>{line.text}</del>
                            <ins>{e.method === 'tts' ? e.replacement_text : 'Muted: no safe replacement could be generated.'}</ins>
                            <div className="vs-chips">
                              <span className={`chip chip-${e.severity}`}>{e.severity === 'hate' ? 'Hate speech' : 'Offensive'}</span>
                              {e.target !== 'none' && <span className="chip">{TARGET_LABEL[e.target] ?? e.target}</span>}
                              <button className="vs-link" onClick={() => setSeek({ which: 'c', t: e.new_start, n: Date.now() })}>
                                Play cleaned
                              </button>
                            </div>
                            {e.reason && <p className="vs-reason">{e.reason}</p>}
                          </>
                        ) : line.text}
                      </div>
                    </li>
                  );
                })}
              </ol>

              <div className="vs-actions">
                <a className="vs-btn vs-btn-primary" href={voiceApi.media(ae.cleaned_url)} download>Download cleaned MP3</a>
                <button className="vs-btn vs-btn-ghost" onClick={downloadReport}>Download edit report</button>
              </div>
              <p className="vs-muted vs-small">
                ASR: {ae.asr_model} · Mode: {ae.mode === 'rewrite' ? 'Rewrite' : 'Counter-narrative'}
              </p>
            </div>
          )}
        </section>
      </div>

      {history.length > 0 && (
        <section className="vs-card vs-history" aria-labelledby="vs-hist">
          <h3 id="vs-hist">Recent</h3>
          <ul>
            {history.map((h) => (
              <li key={h.id}>
                <button onClick={() => openJob(h)} className={job?.id === h.id ? 'on' : ''}>
                  <span className={`vs-status st-${h.status}`}>{h.status}</span>
                  <span className="vs-h-name">{h.filename}</span>
                  <span className="vs-h-meta">
                    {h.summary ? `${langLabel(h.summary.language)} · ${fmtTime(h.summary.duration_s ?? 0)} · ${h.summary.edits} replaced` : ''}
                    {h.created_at ? ` · ${new Date(h.created_at * 1000).toLocaleString()}` : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {enrollOpen && (
        <EnrollVoiceDialog
          onClose={() => setEnrollOpen(false)}
          onEnrolled={(v) => { setVoice(v); setUseEnrolled(true); store.set('civitas.voiceId', v.id); setEnrollOpen(false); }}
        />
      )}
    </div>
  );
}
