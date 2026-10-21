import { useEffect, useMemo, useRef, useState } from 'react';
import { LANGUAGES, Voice, voiceApi } from '../../api/voice';
import EnrollVoiceDialog from '../VoiceStudio/EnrollVoiceDialog';
import { useLiveRoom } from './useLiveRoom';
import '../VoiceStudio/VoiceStudio.css';
import './LiveRoom.css';

const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const newCode = () => Math.random().toString(36).slice(2, 8);
const initials = (n: string) => n.split(/\s+/).map((p) => p[0]).join('').slice(0, 2).toUpperCase() || '?';

export default function LiveRoomView() {
  const params = new URLSearchParams(location.search);
  const [room, setRoom] = useState(params.get('room') ?? newCode());
  const [name, setName] = useState(store.get('civitas.name') ?? '');
  const [language, setLanguage] = useState('auto');
  const [mode, setMode] = useState<'rewrite' | 'counter'>('rewrite');
  const [voice, setVoice] = useState<Voice | null>(null);
  const [useVoice, setUseVoice] = useState(true);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [consent, setConsent] = useState(false);
  const [copied, setCopied] = useState(false);
  const live = useLiveRoom();
  const feed = useRef<HTMLOListElement>(null);

  useEffect(() => {
    const vid = store.get('civitas.voiceId');
    if (vid) voiceApi.voice(vid).then(setVoice).catch(() => {});
  }, []);

  useEffect(() => {
    feed.current?.scrollTo({ top: feed.current.scrollHeight, behavior: 'smooth' });
  }, [live.captions.length]);

  const inRoom = live.status === 'live';
  const validCode = /^[A-Za-z0-9_-]{3,32}$/.test(room);
  const invite = useMemo(() => `${location.origin}${location.pathname}?room=${encodeURIComponent(room)}&tab=live`, [room]);

  const join = () => {
    store.set('civitas.name', name);
    live.join({ room, name: name.trim() || 'Guest', language, mode, voiceId: useVoice && voice ? voice.id : undefined });
  };

  const copyInvite = async () => {
    try { await navigator.clipboard.writeText(invite); setCopied(true); setTimeout(() => setCopied(false), 1800); }
    catch { /* clipboard blocked */ }
  };

  if (!inRoom) {
    return (
      <div className="vs lr">
        <section className="vs-hero">
          <div>
            <h2>Live Room</h2>
            <p>
              Talk with others in real time. Civitas listens to each sentence and, if it contains hate speech,
              everyone else hears a rewritten version in your voice instead. Expect a short delay of a few
              seconds per sentence.
            </p>
          </div>
        </section>

        <div className="lr-lobby">
          <section className="vs-card" aria-labelledby="lr-join">
            <h3 id="lr-join">Join a room</h3>
            <div className="vs-row2">
              <label className="vs-field">
                <span>Room code</span>
                <div className="lr-code">
                  <input value={room} onChange={(e) => setRoom(e.target.value.trim())} aria-invalid={!validCode} />
                  <button className="vs-btn vs-btn-ghost" onClick={() => setRoom(newCode())} type="button">New</button>
                </div>
              </label>
              <label className="vs-field">
                <span>Your name</span>
                <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Asha" maxLength={40} />
              </label>
            </div>
            <div className="vs-row2">
              <label className="vs-field">
                <span>You will speak</span>
                <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                  {LANGUAGES.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
                </select>
              </label>
              <label className="vs-field">
                <span>Replace hate speech with</span>
                <select value={mode} onChange={(e) => setMode(e.target.value as 'rewrite' | 'counter')}>
                  <option value="rewrite">Rewrite (recommended)</option>
                  <option value="counter">Counter-narrative</option>
                </select>
              </label>
            </div>
            <div className="lr-voice">
              {voice ? (
                <label className="vs-check">
                  <input type="checkbox" checked={useVoice} onChange={(e) => setUseVoice(e.target.checked)} />
                  <span>Use my enrolled voice for replacements (best quality from the first sentence)</span>
                </label>
              ) : (
                <p className="vs-muted vs-small">
                  Without an enrolled voice, Civitas learns your voice from your first clean sentences; until then,
                  replaced sentences are muted. <button className="vs-link" onClick={() => setEnrollOpen(true)}>Enroll my voice</button>
                </p>
              )}
            </div>
            <label className="vs-check">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>I agree that my hateful sentences may be replaced with AI speech in my voice for other listeners.</span>
            </label>
            <button className="vs-btn vs-btn-primary vs-btn-block" onClick={join}
                    disabled={!consent || !validCode || live.status === 'connecting'}>
              {live.status === 'connecting' ? 'Connecting…' : 'Join room'}
            </button>
            {live.error && <p className="vs-error" role="alert">{live.error}</p>}
          </section>

          <aside className="vs-card lr-tips">
            <h3>Before you join</h3>
            <ul>
              <li><b>Use headphones.</b> Otherwise your mic picks up other people's audio.</li>
              <li><b>Speak in sentences</b> and pause briefly between them. Civitas processes one sentence at a time.</li>
              <li><b>Invite others</b> with the room code. Up to 8 people per room.</li>
              <li>You never hear yourself. Your captions show when one of your sentences was replaced.</li>
            </ul>
          </aside>
        </div>
        {enrollOpen && (
          <EnrollVoiceDialog onClose={() => setEnrollOpen(false)}
                             onEnrolled={(v) => { setVoice(v); setUseVoice(true); store.set('civitas.voiceId', v.id); setEnrollOpen(false); }} />
        )}
      </div>
    );
  }

  return (
    <div className="vs lr">
      <section className="lr-bar vs-card">
        <div className="lr-room">
          <span className="vs-muted vs-small">Room</span>
          <b>{room}</b>
          <button className="vs-link" onClick={copyInvite}>{copied ? 'Link copied' : 'Copy invite link'}</button>
        </div>
        <span className="lr-delay" title="Time from the end of a sentence to listeners hearing it">
          {live.latency ? `Moderation delay ≈ ${(live.latency / 1000).toFixed(1)} s` : 'Moderation delay: a few seconds'}
        </span>
        <div className="lr-controls">
          <div className="vs-meter lr-meter" aria-hidden="true"><span style={{ width: `${live.level * 100}%` }} /></div>
          <button className={`vs-btn ${live.muted ? 'vs-btn-danger' : 'vs-btn-ghost'}`} onClick={live.toggleMute}
                  aria-pressed={live.muted}>{live.muted ? 'Unmute' : 'Mute'}</button>
          <button className="vs-btn vs-btn-ghost" onClick={live.leave}>Leave</button>
        </div>
      </section>

      <div className="lr-main">
        <section className="vs-card" aria-label="Participants">
          <h3>People ({live.members.length})</h3>
          <ul className="lr-people">
            {live.members.map((m) => {
              const st = live.playing === m.id ? 'talking' : live.states[m.id] ?? 'idle';
              return (
                <li key={m.id} className={`lr-person st-${st}`}>
                  <span className="lr-avatar" aria-hidden="true">{initials(m.name)}</span>
                  <span className="lr-name">{m.name}{m.id === live.me ? ' (you)' : ''}</span>
                  <span className="lr-state">
                    {m.muted ? 'Muted' : st === 'talking' ? 'Playing' : st === 'speaking' ? 'Speaking…'
                      : st === 'processing' ? 'Checking…' : m.voice === 'enrolled' ? 'Voice enrolled' : 'Learning voice'}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="vs-card lr-feed-card" aria-label="Captions">
          <h3>Conversation</h3>
          {live.captions.length === 0 ? (
            <p className="vs-muted">Start talking. Captions of what others hear appear here.</p>
          ) : (
            <ol className="lr-feed" ref={feed} aria-live="polite">
              {live.captions.map((c) => (
                <li key={c.key + (c.self ? '-s' : '')} className={`${c.self ? 'self' : ''} ${c.edited ? 'edited' : ''}`}>
                  <div className="lr-meta">
                    <b>{c.self ? 'You' : c.name}</b>
                    {c.edited && <span className="chip chip-hate">{c.method === 'tts' ? 'Re-voiced' : 'Removed'}</span>}
                    <span className="vs-muted">{new Date(c.at).toLocaleTimeString()}</span>
                  </div>
                  <p>{c.text}</p>
                  {c.self && c.edited && (
                    <p className="lr-note">
                      Others heard this instead of: <del>{c.originalText}</del>
                    </p>
                  )}
                </li>
              ))}
            </ol>
          )}
        </section>
      </div>
    </div>
  );
}
