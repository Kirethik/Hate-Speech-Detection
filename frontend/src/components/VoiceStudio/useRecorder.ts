import { useCallback, useEffect, useRef, useState } from 'react';

export interface Recording {
  blob: Blob;
  filename: string;
  seconds: number;
}

/** Microphone recorder (MediaRecorder). Works on https:// or localhost only. */
export function useRecorder(maxSeconds = 300) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stream = useRef<MediaStream | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const raf = useRef<number | undefined>(undefined);
  const ctx = useRef<AudioContext | null>(null);
  const resolver = useRef<((r: Recording | null) => void) | null>(null);
  const started = useRef(0);

  const cleanup = useCallback(() => {
    clearInterval(timer.current);
    if (raf.current) cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    ctx.current?.close().catch(() => {});
    ctx.current = null;
    setLevel(0);
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const start = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone needs a secure page (https:// or localhost).');
      return;
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      stream.current = s;
      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg']
        .find((m) => (window as any).MediaRecorder?.isTypeSupported?.(m)) ?? '';
      const r = new MediaRecorder(s, mime ? { mimeType: mime } : undefined);
      chunks.current = [];
      r.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
      r.onstop = () => {
        const type = r.mimeType || 'audio/webm';
        const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
        const secs = (performance.now() - started.current) / 1000;
        const out = { blob: new Blob(chunks.current, { type }), filename: `recording.${ext}`, seconds: secs };
        cleanup();
        resolver.current?.(out);
        resolver.current = null;
      };
      rec.current = r;
      started.current = performance.now();
      r.start(250);
      setRecording(true);
      setSeconds(0);
      timer.current = window.setInterval(() => {
        const t = (performance.now() - started.current) / 1000;
        setSeconds(t);
        if (t >= maxSeconds) r.state === 'recording' && r.stop();
      }, 200);
      // input level meter
      const ac = new AudioContext();
      ctx.current = ac;
      const an = ac.createAnalyser();
      an.fftSize = 512;
      ac.createMediaStreamSource(s).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        let peak = 0;
        for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
        setLevel(Math.min(1, peak / 90));
        raf.current = requestAnimationFrame(tick);
      };
      tick();
    } catch (e: any) {
      cleanup();
      setError(e?.name === 'NotAllowedError'
        ? 'Microphone permission was denied. Allow it in the browser address bar.'
        : `Could not start the microphone: ${e?.message ?? e}`);
    }
  }, [cleanup, maxSeconds]);

  const stop = useCallback((): Promise<Recording | null> => {
    setRecording(false);
    return new Promise((resolve) => {
      const r = rec.current;
      if (!r || r.state === 'inactive') return resolve(null);
      resolver.current = resolve;
      r.stop();
    });
  }, []);

  return { recording, seconds, level, error, start, stop };
}
