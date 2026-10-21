import { useEffect, useMemo, useRef, useState } from 'react';
import { fmtTime } from '../../api/voice';

export interface Region {
  start: number;
  end: number;
  kind: 'removed' | 'replaced' | 'muted';
  label?: string;
}

interface Props {
  src: string;
  title: string;
  subtitle?: string;
  regions: Region[];
  tone: 'original' | 'cleaned';
  seekTo?: { t: number; n: number } | null; // n increments so repeated seeks to the same time still fire
  onPlay?: () => void;
  pauseSignal?: number;
}

const BARS = 220;

/** Decodes the audio in the browser, draws peak bars, overlays edit regions, plays with an <audio>. */
export default function Waveform({ src, title, subtitle, regions, tone, seekTo, onPlay, pauseSignal }: Props) {
  const audio = useRef<HTMLAudioElement>(null);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [dur, setDur] = useState(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let dead = false;
    setPeaks(null);
    setErr(null);
    (async () => {
      try {
        const buf = await (await fetch(src)).arrayBuffer();
        const ac = new (window.AudioContext || (window as any).webkitAudioContext)();
        const data = await ac.decodeAudioData(buf);
        ac.close();
        const ch = data.getChannelData(0);
        const step = Math.max(1, Math.floor(ch.length / BARS));
        const p: number[] = [];
        for (let i = 0; i < BARS; i++) {
          let m = 0;
          for (let j = i * step; j < Math.min(ch.length, (i + 1) * step); j += 8) m = Math.max(m, Math.abs(ch[j]));
          p.push(m);
        }
        const mx = Math.max(...p, 1e-3);
        if (!dead) {
          setPeaks(p.map((v) => Math.max(0.04, v / mx)));
          setDur(data.duration);
        }
      } catch {
        if (!dead) setErr('Could not draw the waveform (audio still plays).');
      }
    })();
    return () => { dead = true; };
  }, [src]);

  useEffect(() => {
    if (seekTo && audio.current) {
      audio.current.currentTime = seekTo.t;
      audio.current.play().catch(() => {});
    }
  }, [seekTo]);

  useEffect(() => {
    if (pauseSignal) audio.current?.pause();
  }, [pauseSignal]);

  const total = dur || audio.current?.duration || 1;
  const progress = Math.min(1, t / total);
  const regionEls = useMemo(() => regions.map((r, i) => (
    <div
      key={i}
      className={`wf-region wf-region-${r.kind}`}
      style={{ left: `${(r.start / total) * 100}%`, width: `${Math.max(0.4, ((r.end - r.start) / total) * 100)}%` }}
      title={r.label}
    />
  )), [regions, total]);

  const toggle = () => {
    const a = audio.current;
    if (!a) return;
    if (a.paused) { a.play().catch(() => {}); } else { a.pause(); }
  };

  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    const a = audio.current;
    if (!a) return;
    const box = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - box.left) / box.width) * total;
  };

  return (
    <div className={`wf wf-${tone}`}>
      <div className="wf-head">
        <button className="wf-play" onClick={toggle} aria-label={playing ? `Pause ${title}` : `Play ${title}`}>
          {playing ? (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.87l11-6.5a1 1 0 0 0 0-1.74l-11-6.5A1 1 0 0 0 8 5.5z" /></svg>
          )}
        </button>
        <div className="wf-titles">
          <span className="wf-title">{title}</span>
          {subtitle && <span className="wf-sub">{subtitle}</span>}
        </div>
        <span className="wf-time">{fmtTime(t)} / {fmtTime(total)}</span>
      </div>
      <div className="wf-track" onClick={seek} role="slider" aria-label={`${title} position`}
           aria-valuemin={0} aria-valuemax={Math.round(total)} aria-valuenow={Math.round(t)} tabIndex={0}
           onKeyDown={(e) => {
             const a = audio.current; if (!a) return;
             if (e.key === 'ArrowRight') a.currentTime = Math.min(total, a.currentTime + 5);
             if (e.key === 'ArrowLeft') a.currentTime = Math.max(0, a.currentTime - 5);
             if (e.key === ' ') { e.preventDefault(); toggle(); }
           }}>
        {regionEls}
        <div className="wf-bars">
          {(peaks ?? Array.from({ length: BARS }, () => 0.06)).map((v, i) => (
            <span key={i} className={i / BARS < progress ? 'played' : ''} style={{ height: `${v * 100}%` }} />
          ))}
        </div>
        <div className="wf-cursor" style={{ left: `${progress * 100}%` }} />
      </div>
      {err && <p className="wf-err">{err}</p>}
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onTimeUpdate={(e) => setT(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => !dur && setDur(e.currentTarget.duration)}
        onPlay={() => { setPlaying(true); onPlay?.(); }}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
    </div>
  );
}
