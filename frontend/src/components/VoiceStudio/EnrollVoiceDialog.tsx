import { useEffect, useRef, useState } from 'react';
import { LANGUAGES, Voice, voiceApi } from '../../api/voice';
import { useRecorder } from './useRecorder';

interface Props {
  onClose: () => void;
  onEnrolled: (v: Voice) => void;
}

/** Records the user reading a fixed sentence: a clean voice sample with an exact transcript. */
export default function EnrollVoiceDialog({ onClose, onEnrolled }: Props) {
  const [lang, setLang] = useState('en');
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const rec = useRecorder(20);
  const dialog = useRef<HTMLDivElement>(null);

  useEffect(() => {
    voiceApi.prompt(lang).then((p) => setPrompt(p.text)).catch(() => setPrompt(''));
  }, [lang]);

  useEffect(() => {
    dialog.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, [onClose]);

  const finish = async () => {
    const r = await rec.stop();
    if (!r) return;
    if (r.seconds < 3) { setErr('That was too short. Read the whole sentence at a normal pace.'); return; }
    setBusy(true);
    setErr(null);
    try {
      onEnrolled(await voiceApi.enroll(r.blob, r.filename, lang, prompt));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="vs-modal-backdrop" onClick={onClose}>
      <div className="vs-modal" role="dialog" aria-modal="true" aria-labelledby="enroll-title" tabIndex={-1}
           ref={dialog} onClick={(e) => e.stopPropagation()}>
        <h3 id="enroll-title">Enroll your voice</h3>
        <p className="vs-muted">
          Read the sentence below in a quiet room. Civitas uses this 5–15 second sample only to re-voice
          your own recordings. You can delete it at any time.
        </p>
        <label className="vs-field">
          <span>Language you will speak</span>
          <select value={lang} onChange={(e) => setLang(e.target.value)} disabled={rec.recording || busy}>
            {LANGUAGES.filter((l) => l.code !== 'auto').map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
        </label>
        <blockquote className="vs-prompt" lang={lang} data-script={lang === 'en' ? 'latin' : 'native'}>
          {prompt || '…'}
        </blockquote>
        <div className="vs-rec-row">
          {!rec.recording ? (
            <button className="vs-btn vs-btn-primary" onClick={rec.start} disabled={busy || !prompt}>
              <span className="vs-dot" aria-hidden="true" /> Start reading
            </button>
          ) : (
            <button className="vs-btn vs-btn-danger" onClick={finish}>
              Stop · {rec.seconds.toFixed(0)}s
            </button>
          )}
          <div className="vs-meter" aria-hidden="true"><span style={{ width: `${rec.level * 100}%` }} /></div>
        </div>
        {(err || rec.error) && <p className="vs-error" role="alert">{err || rec.error}</p>}
        {busy && <p className="vs-muted">Saving your voice sample…</p>}
        <div className="vs-modal-actions">
          <button className="vs-btn vs-btn-ghost" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}
