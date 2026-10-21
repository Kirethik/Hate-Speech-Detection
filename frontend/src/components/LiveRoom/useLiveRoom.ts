import { useCallback, useEffect, useRef, useState } from 'react';

export interface Member { id: string; name: string; muted: boolean; voice: 'enrolled' | 'learning' }
export interface Caption {
  key: string;
  from: string;
  name: string;
  text: string;
  originalText?: string;
  edited: boolean;
  method?: 'tts' | 'muted' | null;
  self: boolean;
  latencyMs?: number;
  at: number;
}
export interface JoinOptions { room: string; name: string; language: string; mode: 'rewrite' | 'counter'; voiceId?: string }
type Status = 'idle' | 'connecting' | 'live' | 'error';

// AudioWorklet: capture mic, downsample to 16 kHz mono PCM16, post 20 ms frames.
const WORKLET = `
class Cap extends AudioWorkletProcessor {
  constructor() { super(); this.r = sampleRate / 16000; this.ph = 0; this.s = 0; this.c = 0;
    this.out = new Int16Array(320); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.s += ch[i]; this.c++; this.ph += 1;
      if (this.ph >= this.r) {
        this.ph -= this.r;
        const v = Math.max(-1, Math.min(1, this.s / this.c));
        this.s = 0; this.c = 0;
        this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.n === 320) { this.port.postMessage(this.out.buffer.slice(0)); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor('civitas-capture', Cap);
`;

function wsUrl(room: string) {
  const base: string = (import.meta as any).env?.VITE_VOICE_API ?? '';
  if (base) return base.replace(/^http/, 'ws') + `/ws/rooms/${encodeURIComponent(room)}`;
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/rooms/${encodeURIComponent(room)}`;
}

export function useLiveRoom() {
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [me, setMe] = useState<string | null>(null);
  const [members, setMembers] = useState<Member[]>([]);
  const [states, setStates] = useState<Record<string, string>>({});
  const [playing, setPlaying] = useState<string | null>(null);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [muted, setMuted] = useState(false);
  const [level, setLevel] = useState(0);
  const [latency, setLatency] = useState<number | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const capCtx = useRef<AudioContext | null>(null);
  const playCtx = useRef<AudioContext | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const nextT = useRef(0);
  const mutedRef = useRef(false);
  const raf = useRef<number | undefined>(undefined);

  const teardown = useCallback(() => {
    if (raf.current) cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    capCtx.current?.close().catch(() => {});
    capCtx.current = null;
    playCtx.current?.close().catch(() => {});
    playCtx.current = null;
    const w = ws.current;
    ws.current = null;
    if (w && w.readyState <= 1) {
      try { w.send(JSON.stringify({ type: 'leave' })); } catch { /* closing */ }
      w.close();
    }
    setLevel(0);
  }, []);

  useEffect(() => teardown, [teardown]);

  const play = useCallback((pcm: ArrayBuffer, from: string, dur: number) => {
    const ctx = playCtx.current;
    if (!ctx || pcm.byteLength < 2) return;
    const i16 = new Int16Array(pcm);
    const buf = ctx.createBuffer(1, i16.length, 16000);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const t = Math.max(ctx.currentTime + 0.05, nextT.current);
    src.start(t);
    nextT.current = t + buf.duration;
    const startDelay = (t - ctx.currentTime) * 1000;
    window.setTimeout(() => setPlaying(from), startDelay);
    window.setTimeout(() => setPlaying((p) => (p === from ? null : p)), startDelay + dur * 1000);
  }, []);

  const join = useCallback(async (o: JoinOptions) => {
    setError(null);
    setCaptions([]);
    setStatus('connecting');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone needs https:// or localhost.');
      const s = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
      stream.current = s;
      playCtx.current = new AudioContext();
      const cap = new AudioContext();
      capCtx.current = cap;
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      await cap.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      const srcNode = cap.createMediaStreamSource(s);
      const node = new AudioWorkletNode(cap, 'civitas-capture');
      srcNode.connect(node);
      const an = cap.createAnalyser();
      an.fftSize = 512;
      srcNode.connect(an);
      const lv = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(lv);
        let pk = 0;
        for (const v of lv) pk = Math.max(pk, Math.abs(v - 128));
        setLevel(mutedRef.current ? 0 : Math.min(1, pk / 90));
        raf.current = requestAnimationFrame(tick);
      };
      tick();

      const w = new WebSocket(wsUrl(o.room));
      w.binaryType = 'arraybuffer';
      ws.current = w;
      node.port.onmessage = (e) => {
        if (!mutedRef.current && w.readyState === WebSocket.OPEN) w.send(e.data);
      };
      w.onopen = () => w.send(JSON.stringify({ type: 'join', name: o.name, language: o.language, mode: o.mode, voice_id: o.voiceId }));
      w.onmessage = (ev) => {
        if (typeof ev.data !== 'string') {
          const b = ev.data as ArrayBuffer;
          const n = new DataView(b).getUint32(0);
          const head = JSON.parse(new TextDecoder().decode(new Uint8Array(b, 4, n)));
          play(b.slice(4 + n), head.from, head.duration_s ?? 0);
          return;
        }
        const m = JSON.parse(ev.data);
        if (m.type === 'joined') { setMe(m.member_id); setMembers(m.members); setStatus('live'); }
        else if (m.type === 'presence') setMembers(m.members);
        else if (m.type === 'state') setStates((st) => ({ ...st, [m.member_id]: m.state }));
        else if (m.type === 'error') { setError(m.message); setStatus('error'); }
        else if (m.type === 'caption' || m.type === 'self_result') {
          const self = m.type === 'self_result';
          if (!self && m.latency_ms) setLatency(m.latency_ms);
          setCaptions((c) => [...c.slice(-199), {
            key: `${m.from}-${m.seq}`, from: m.from, name: m.name, text: m.text, originalText: m.original_text,
            edited: m.edited, method: m.method, self, latencyMs: m.latency_ms, at: Date.now(),
          }]);
        }
      };
      w.onclose = () => {
        setStatus((st) => (st === 'error' ? st : 'idle'));
        setMe(null);
        teardown();
      };
      w.onerror = () => { setError('Lost connection to the Civitas server.'); setStatus('error'); };
    } catch (e: any) {
      teardown();
      setStatus('error');
      setError(e?.name === 'NotAllowedError' ? 'Microphone permission was denied.' : (e?.message ?? String(e)));
    }
  }, [play, teardown]);

  const leave = useCallback(() => { teardown(); setStatus('idle'); setMe(null); setMembers([]); }, [teardown]);

  const toggleMute = useCallback(() => {
    const v = !mutedRef.current;
    mutedRef.current = v;
    setMuted(v);
    stream.current?.getAudioTracks().forEach((t) => { t.enabled = !v; });
    ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify({ type: 'mute', muted: v }));
  }, []);

  return { status, error, me, members, states, playing, captions, muted, level, latency, join, leave, toggleMute };
}
