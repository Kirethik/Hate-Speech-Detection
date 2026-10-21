# Civitas Voice: build guide (pretrained models, no training)

Goal: a user speaks (upload a recording, or live mic in a room). Civitas finds the hateful parts, writes a non-hateful replacement, speaks the replacement in the same person's voice, splices it into the audio, and other users hear only the cleaned version.

This guide replaces "train Model A / Model B" with downloaded models. Your existing repo pieces (FastAPI server, canonical JSON schema, React + Vite + TS frontend, SQLite history) stay. The stubs in `speech/` and `pipeline/` get real implementations.

---

## 0. TL;DR: the model stack

| Stage | Pick | Why | License |
|---|---|---|---|
| Voice activity detection | `silero-vad` (pip) | Tiny, CPU-fast, streaming-capable; needed for live mode and for chunking | MIT |
| ASR (en, hi, ur, auto language ID) | `faster-whisper` + `large-v3-turbo` | Word timestamps built in, int8 on GPU, auto language detection | MIT |
| ASR (ta, te, ml and other Indic) | `ai4bharat/indic-conformer-600m-multilingual` (gated) | Much better than Whisper on Dravidian languages | MIT |
| Word timestamps for IndicConformer | `ctc-forced-aligner` (MMS-300m aligner) | Whisper gives timestamps itself; IndicConformer doesn't | CC-BY-NC (see §3.6) |
| Hate detection, fast first pass | `textdetox/xlmr-large-toxicity-classifier-v2` + `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` | Pretrained classifiers; first covers en/hi/Hinglish, second covers code-mixed Tamil/Malayalam/Kannada, Hindi, Urdu, Bengali, Marathi | openrail++ / AFL-3.0 |
| Final verdict + spans + replacement text | Gemma 4 E4B (`ollama pull gemma4:e4b`) | Multilingual, Apache 2.0, small; does context-aware judgment AND writes the rewrite/counter-narrative in one JSON call | Apache 2.0 |
| Voice-cloned TTS (ta, te, ml, kn, bn, mr, gu, pa, or, as, hi) | `ai4bharat/IndicF5` (gated) | Zero-shot cloning from a short reference clip + its transcript, 11 Indian languages | MIT |
| Voice-cloned TTS (en, hi, + 21 others) | Chatterbox Multilingual (`pip install chatterbox-tts`) | Zero-shot cloning, good English quality, emotion control | MIT |
| Fallback TTS for anything else | `k2-fsa/OmniVoice` | 600+ languages, zero-shot, very fast | weights CC-BY-NC |
| Watermark on synthetic audio | `audioseal` (Meta) | Marks the synthesized segments as AI-generated | MIT |
| Live rooms (later, for scale) | LiveKit (self-hosted SFU + Python Agents) | WebRTC rooms; only needed past a WebSocket MVP | Apache 2.0 |

Hardware: production wants a single 24 GB GPU (L4 / A10G class). **Local development runs on an Intel Core Ultra + Arc laptop**: Whisper on the CPU (int8), classifiers on the Arc GPU through PyTorch XPU, Gemma through Ollama (Vulkan), TTS on XPU with CPU fallback. See `docs/VOICE_QUICKSTART.md`.

> **Implementation status (Oct 2026).** Phases 1–4 are built: `voice/` (pipeline), `server/voice_routes.py` (`/api/v1`), `tts_service/` (one TTS engine per process), the Voice Studio page, and the Windows setup/run scripts. One deviation from this guide: the Indic route **does not use the MMS forced aligner**. Sentences come from VAD chunks (we replace whole sentences anyway), which avoids a C++ build on Windows and the aligner's non-commercial license. Still to do: Docker/deployment (Phase 5), live rooms (Phase 6), AudioSeal watermarking, Demucs background preservation.

Every model above is an inference-only download. Nothing gets trained.

---

## 1. Two product decisions to settle first

### 1.1 What replaces the hateful sentence: "rewrite" or "counter-narrative"?

Your CLAUDE.md already separates these:

- **Rewrite**: same intent, hate removed. "These people are ruining our city" becomes "I'm worried about how our city is changing." Spoken in the speaker's voice, it still sounds like *them*. This is the natural default for in-place replacement.
- **Counter-narrative**: a statement that challenges the hateful idea. Dropped into the speaker's own voice, it makes them argue against themselves, which listeners will find odd unless it's phrased as a first-person reflection ("Blaming a whole community isn't fair; the real problem is...").

Build both. Expose `mode = rewrite | counter` as a per-upload and per-room setting with `rewrite` as the default. The LLM call in §4.4 returns both in one shot, so this costs nothing.

### 1.2 Consent and disclosure (needed for a real product)

You will be making a person's cloned voice say words they didn't say. For a real-world deployment:

1. Signup/room-join consent: "Civitas may replace parts of your speech with AI-generated audio in your voice."
2. Listeners see an "edited" marker on every replaced segment (the UI timeline in §7 does this naturally).
3. Watermark synthesized segments with AudioSeal and store an edit log (which seconds were replaced, original vs new text) per output.
4. Only clone the voice of the person who is speaking. Never accept a reference clip from somebody else.

This also protects the project in reviews and demos.

---

## 2. Architecture

```
                   ┌─────────────────────────── Browser (React) ───────────────────────────┐
                   │  Studio: upload/record → progress → A/B player + edit timeline        │
                   │  Live room: mic (AudioWorklet 16 kHz) → WS → hear others' cleaned audio│
                   └───────────────┬──────────────────────────────────┬────────────────────┘
                                   │ HTTPS /api, /media                │ WSS /ws/rooms/{id}
                              ┌────▼──────────────────────────────────▼────┐
                              │ Caddy (TLS, static React build, proxy)      │
                              └────┬───────────────────────────────────────┘
                                   │
          ┌────────────────────────▼─────────────────────────┐        ┌──────────────────────┐
          │ engine (FastAPI, 1 uvicorn worker, owns GPU)       │ HTTP   │ tts (FastAPI)         │
          │  VAD → ASR → align → detect → LLM → TTS → splice   ├───────►│ IndicF5 + Chatterbox  │
          │  job queue (asyncio, GPU lock, live has priority)  │        │ (+ OmniVoice optional)│
          │  SQLite/Postgres: jobs, edits, history, voices     │        └──────────────────────┘
          └───────┬───────────────────────────────┬──────────┘
                  │ OpenAI-compatible HTTP          │ volume
          ┌───────▼────────┐               ┌───────▼─────────────┐
          │ ollama (Gemma 4)│               │ /data (uploads,     │
          └────────────────┘               │ outputs, model cache)│
                                           └─────────────────────┘
```

Why this split:

- **TTS gets its own service.** `chatterbox-tts` and IndicF5 pin their own `torch`/`transformers` versions. Putting them in the engine's environment will cause dependency conflicts. Behind an HTTP `POST /synthesize` they can live in their own image, and you can split them further into `tts-indic` and `tts-multi` if their pins clash with each other.
- **The LLM runs in Ollama**, called through the OpenAI-compatible API. Switching later to vLLM or a hosted API only changes `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`.
- **The engine runs a single uvicorn worker**, so models load once. A GPU lock serializes inference. Live utterances jump the queue ahead of batch jobs.

The two modes share the pipeline:

| | Upload mode (Phase 1–3) | Live mode (Phase 4) |
|---|---|---|
| Input | 1 file, up to N minutes | 20 ms PCM frames over WebSocket |
| Unit of work | whole file → sentence segments | one utterance (VAD end-of-speech or 8 s max) |
| Voice reference | auto-picked clean 5–12 s from the same file, or the user's enrolled voice | enrolled voice (recorded at room join) |
| Output | cleaned WAV/MP3 + edit list | per-utterance audio broadcast to other room members |
| Latency | ~0.5–1× audio length on a 24 GB GPU (estimate; measure it) | utterance length + ~1–3 s |

**Live mode is never truly instant.** You can't know a sentence is hateful until you hear the end of it, so live mode works like a broadcast delay: each utterance plays for others shortly after the speaker finishes it. Say this plainly in your report and UI ("~2 s moderation delay").

---

## 3. Model details, alternatives, gotchas

### 3.1 ASR

- **Default:** `faster-whisper`, `WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")`, `word_timestamps=True`, `vad_filter=True`. It gives you language ID (`info.language`), segments and per-word start/end/probability.
- **Indic route:** if the detected language is ta/te/ml (or another Indic language besides hi/ur), re-transcribe with IndicConformer. It's gated on Hugging Face: log in, accept its terms, use an `HF_TOKEN`. Call it as `model(wav_16k_tensor, "ta", "ctc")` per VAD chunk, and align words afterwards (§3.6).
- **Code-mixed speech (Tanglish/Hinglish):** Whisper may output native script, Latin script or a mix. Don't fight it. The detectors and the LLM handle both scripts. Keep `task="transcribe"` (never `translate`).
- **Note:** AI4Bharat's model card now mentions a newer "Indic-Transcribe" model (25 languages). Try it if you have time; the interface is similar.

### 3.2 Detection: two layers

1. **Classifiers** (milliseconds per sentence) run on every sentence: max(textdetox, CNERG) gives a `p_toxic`.
2. **LLM judge** (Gemma 4) runs when `p_toxic ≥ 0.30`, or when the language isn't covered by the classifiers (Telugu, for example), or when the sentence is short and ambiguous. It sees the full transcript as context. That catches sarcasm, quoting ("he said X and that's wrong"), reclaimed slurs and implicit hate, which classifiers miss. It returns the final verdict, the hateful spans, the target group and both replacement texts.

Why not the LLM alone? Cost and latency. The classifiers filter out about 80–90% of normal sentences before the LLM is called. Why not the classifiers alone? They flag profanity as hate and miss implicit hate. Together they work well.

Later you can drop your own trained Model A (XLM-R multi-task) in as a third classifier behind the same `Detector.score()` interface. Nothing else changes.

### 3.3 LLM choice

- **Gemma 4 E4B** (`gemma4:e4b`, Apache 2.0, needs Ollama ≥ 0.22). About 5 GB of memory at 4-bit including KV headroom.
- **Dev on the 3050:** `gemma4:e2b`, or point `LLM_BASE_URL` at any hosted OpenAI-compatible API while developing.
- **If quality on Tamil/Telugu/Malayalam rewrites isn't good enough:** move up to `gemma4:12b` (needs the 24 GB GPU), or an Indic-specialised model (Sarvam family) served through vLLM.
- **Disable "thinking"/reasoning output** for latency if the model or server has it on by default.
- **Refusals:** instruction-tuned models sometimes refuse to *repeat* hate, but rewriting and detoxifying is normally accepted. The system prompt frames the task as moderation. If the model returns invalid JSON or refuses, retry once, then fall back (§4.5).

### 3.4 TTS routing

| Language | Engine | Reference input |
|---|---|---|
| ta, te, ml, kn, bn, mr, gu, pa, or, as | IndicF5 | ref wav + **exact transcript of the ref wav** |
| en, hi (+ ar, fr, de, es, ... Chatterbox's 23) | Chatterbox Multilingual | ref wav only, `language_id=` |
| ur | Chatterbox `hi` with Devanagari text (spoken Urdu and Hindi are mostly mutually intelligible). The LLM returns `tts_text` already in Devanagari. Or OmniVoice. | |
| anything else | OmniVoice (non-commercial weights) | ref wav + transcript |

Rules that matter for quality:

- Use 5–12 s of clean, single-speaker reference audio with no music. Longer isn't better for F5-style models, so trim to under 12 s.
- The reference language should match the output language. Otherwise the accent bleeds across. Chatterbox docs suggest `cfg_weight=0` when they differ.
- IndicF5 and OmniVoice need `ref_text` to match the reference audio exactly. Use the ASR transcript of the chosen reference span, or the enrollment prompt text (§6.3), which is known exactly.
- Both output 24 kHz. Resample to the original file's rate before splicing.

### 3.5 Voice consistency

The prosody of the replacement won't match the original sentence perfectly. To make it less noticeable:

- Replace whole sentences or clauses, never single words. A synthesized word in the middle of a natural sentence sounds jarring. A whole synthesized sentence between natural ones sounds fine.
- Match loudness (LUFS) to the original segment.
- Optional: separate vocals and background with Demucs (`htdemucs`), splice on the vocal stem, then remix with the original background. This keeps background noise/music continuous, so the replaced sentence doesn't sound "too clean". Turn it on per job (`keep_background=true`); it costs a few seconds per minute of audio.

### 3.6 Licenses (check before calling it "commercial")

Commercial-safe: faster-whisper/Whisper, IndicConformer, IndicF5, Chatterbox, Gemma 4, silero-vad, AudioSeal, CNERG (AFL-3.0), textdetox (openrail++, read its use restrictions).
Non-commercial: OmniVoice weights, the MMS-300m forced aligner (CC-BY-NC-4.0). Also XTTS-v2 and F5-TTS base weights, if you ever try them.

A college project or demo can use all of them. For a commercial launch, replace the MMS aligner. Options: use Whisper's word timestamps only, or derive timings from IndicConformer's own CTC frames.

---

## 4. The pipeline, step by step (upload mode)

File layout (new modules; nothing in `model.py`, `train.py` or `prepare_splits.py` changes):

```
config.py                    # → pydantic-settings, reads env vars (see §8.2)
speech/
  audio_io.py                # load/resample/save, ffmpeg decode of any upload format
  vad.py                     # silero (real)
  asr.py                     # WhisperAsr, IndicAsr, route_asr()
  align.py                   # ctc-forced-aligner wrapper (real)
  voice_ref.py               # pick reference span / load enrolled voice
  tts_client.py              # HTTP client to the tts service
  splice.py                  # snap, fit, loudness, crossfade, splice
  watermark.py               # audioseal
pipeline/
  detect.py                  # classifier layer
  llm.py                     # OpenAI-compatible client + prompt + schema
  audio_job.py               # orchestrates one upload end to end
  jobs.py                    # job store + asyncio worker + GPU lock + progress events
  live.py                    # room/session logic (Phase 4)
server/
  routes/jobs.py  routes/voices.py  routes/rooms.py  routes/health.py
tts_service/
  app.py                     # FastAPI: POST /synthesize, GET /health
  engines.py                 # IndicF5Engine, ChatterboxEngine, OmniVoiceEngine
docker/  Dockerfile.engine  Dockerfile.tts  Dockerfile.web  Caddyfile
docker-compose.yml  docker-compose.dev.yml  .env.example
scripts/download_models.py
```

### 4.1 Decode and normalise

```python
# speech/audio_io.py
import subprocess, numpy as np, soundfile as sf, librosa

def decode(path, sr=None):
    # ffmpeg handles webm/opus from browsers, m4a from phones, mp3, etc.
    out = path + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", out], check=True, capture_output=True)
    y, r = sf.read(out, dtype="float32")
    return (librosa.resample(y, orig_sr=r, target_sr=sr), sr) if sr and sr != r else (y, r)
```

Keep two copies in memory: `y16` (16 kHz, for VAD/ASR/align) and `y_out` (original rate, for splicing and output). Reject inputs over `MAX_AUDIO_SECONDS` (start with 300).

### 4.2 ASR with word timestamps

```python
# speech/asr.py
from dataclasses import dataclass, field
from faster_whisper import WhisperModel

@dataclass
class Word: text: str; start: float; end: float; prob: float
@dataclass
class Seg:  start: float; end: float; text: str; words: list = field(default_factory=list); conf: float = 1.0

class WhisperAsr:
    def __init__(self, cfg):
        self.m = WhisperModel(cfg.whisper_model, device=cfg.device, compute_type=cfg.whisper_compute)
    def run(self, wav_path, lang=None):
        segs, info = self.m.transcribe(wav_path, language=lang, task="transcribe",
                                       word_timestamps=True, vad_filter=True, beam_size=5)
        out = []
        for g in segs:
            ws = [Word(w.word.strip(), w.start, w.end, w.probability) for w in (g.words or [])]
            conf = sum(w.prob for w in ws) / len(ws) if ws else 0.0
            out.append(Seg(g.start, g.end, g.text.strip(), ws, conf))
        return info.language, out

INDIC = {"ta", "te", "ml", "kn", "bn", "mr", "gu", "pa", "or", "as"}

def route_asr(reg, wav_path, y16, lang_hint=None):
    lang, segs = reg.whisper.run(wav_path, lang_hint)
    if lang in INDIC and reg.indic_asr is not None:
        segs = reg.indic_asr.run(y16, lang)          # VAD chunks → IndicConformer → align (§3.6)
    return lang, split_sentences(segs)               # split on . ? ! । and pauses > 0.35 s
```

`IndicAsr.run` does this: silero VAD gives chunks (merged to at most 15 s each). For each chunk, `text = model(chunk_tensor, lang, "ctc")`, then `ctc-forced-aligner` (ISO-639-3 code: tam/tel/mal/...) gives word times, offset by the chunk start. The result is the same `Seg`/`Word` structure.

`split_sentences` matters. Whisper segments can be 10–20 s long and hold several sentences. You want replacement units of roughly 1–8 s.

### 4.3 Classifier pass

```python
# pipeline/detect.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Clf:
    def __init__(self, name, dev, toxic_idx=1):
        self.t = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval().half() if dev == "cuda" \
            else AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.dev, self.k = dev, toxic_idx
    @torch.inference_mode()
    def __call__(self, texts):
        b = self.t(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.dev)
        return self.m(**b).logits.float().softmax(-1)[:, self.k].tolist()

class Detector:
    def __init__(self, dev):
        self.cs = [Clf("textdetox/xlmr-large-toxicity-classifier-v2", dev),
                   Clf("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL", dev)]
    def score(self, texts):
        cols = [c(texts) for c in self.cs]
        return [max(v) for v in zip(*cols)]
```

Check `model.config.id2label` for each model once, and confirm the toxic index is 1 (both cards say index 1 = toxic/abusive).

### 4.4 LLM: verdict, spans, replacement (one JSON call)

```python
# pipeline/llm.py
import json
from openai import OpenAI

SCHEMA = {
  "type": "object",
  "properties": {
    "is_hate":   {"type": "boolean"},
    "severity":  {"type": "string", "enum": ["none", "offensive", "hate"]},
    "target":    {"type": "string", "enum": ["none", "religion", "caste", "gender", "sexual_orientation",
                                             "race_ethnicity", "nationality_region", "disability", "other"]},
    "spans":     {"type": "array", "items": {"type": "string"}},
    "rewrite":   {"type": "string"},
    "counter":   {"type": "string"},
    "tts_text":  {"type": "string"},
    "reason":    {"type": "string"}
  },
  "required": ["is_hate", "severity", "target", "spans", "rewrite", "counter", "tts_text", "reason"]
}

SYSTEM = """You are a content-moderation component in a speech platform.
You receive a full transcript for context and ONE target sentence.
Decide whether the target sentence is hate speech (attacks or dehumanises people for a protected
characteristic) or merely offensive (profanity/insult without a protected target), or neither.
Quoting or criticising hate is NOT hate. Use the context.
If severity is "offensive" or "hate", write:
 - rewrite: what the speaker could have said instead. Same language, same script, same meaning minus
   the hate/insult, first person, natural spoken style, within ±20% of the original word count.
 - counter: a first-person spoken sentence that challenges the hateful idea, same language and script,
   similar length.
 - tts_text: the text to synthesise for the requested mode, in the script the TTS needs
   (for Urdu, write it in Devanagari). For other languages, tts_text equals the chosen text.
 - spans: exact substrings of the target sentence that are hateful/offensive.
If severity is "none", rewrite/counter/tts_text are "" and spans is [].
Output only JSON matching the schema."""

class Llm:
    def __init__(self, cfg):
        self.c = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key or "ollama")
        self.model = cfg.llm_model
    def judge(self, context, sentence, lang, mode):
        user = f"language={lang}\nmode={mode}\n\nTRANSCRIPT:\n{context}\n\nTARGET SENTENCE:\n{sentence}"
        for _ in range(2):
            r = self.c.chat.completions.create(
                model=self.model, temperature=0.2,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": SCHEMA}})
            try:
                return json.loads(r.choices[0].message.content)
            except json.JSONDecodeError:
                continue
        return None
```

Map `target` onto your existing `label_maps.py` classes so the analytics pages keep working.

### 4.5 Safety gate on the replacement

Before synthesizing, run the replacement text back through `Detector.score()`. If `p_toxic ≥ 0.5`, call the LLM again with "previous attempt was still toxic". If it fails a second time, use a fallback:

- `fallback = "mute"`: replace the sentence with silence of equal length, plus a soft tone, and mark it `method="muted"`.
- or a fixed safe template in that language ("I'd rather say that differently.").

This keeps the existing contract from your README: "Model B outputs are never assumed safe."

### 4.6 Pick the voice reference

```python
# speech/voice_ref.py
def pick_ref(segs, flagged, lo=5.0, hi=12.0, gap=0.6):
    best, best_score = None, -1
    for i in range(len(segs)):
        if i in flagged: continue
        j = i
        while (j + 1 < len(segs) and j + 1 not in flagged
               and segs[j + 1].end - segs[i].start <= hi and segs[j + 1].start - segs[j].end < gap):
            j += 1
        dur = segs[j].end - segs[i].start
        if dur < lo: continue
        conf = sum(s.conf for s in segs[i:j + 1]) / (j - i + 1)
        if conf * dur > best_score:
            best_score, best = conf * dur, (segs[i].start, segs[j].end, " ".join(s.text for s in segs[i:j + 1]))
    return best   # (start, end, exact_text) or None
```

Fallback order: (1) the user's enrolled voice, if any; (2) the longest run of any speech, flagged or not, up to 12 s (only the timbre is copied, but angry prosody can carry over); (3) mute instead of replace.

### 4.7 Synthesize (via the tts service)

```python
# tts_service/engines.py  (runs in its own container/env)
import numpy as np, torch
from transformers import AutoModel

class IndicF5Engine:
    def __init__(self): self.m = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    def synth(self, text, ref_wav, ref_text, lang):
        a = self.m(text, ref_audio_path=ref_wav, ref_text=ref_text)
        a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
        return a, 24000

class ChatterboxEngine:
    def __init__(self, dev="cuda"):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.m = ChatterboxMultilingualTTS.from_pretrained(device=dev)
    def synth(self, text, ref_wav, ref_text, lang):
        w = self.m.generate(text, language_id=lang, audio_prompt_path=ref_wav)
        return w.squeeze(0).cpu().numpy(), self.m.sr

ROUTE = {**{l: "indicf5" for l in ["ta","te","ml","kn","bn","mr","gu","pa","or","as"]},
         "en": "chatterbox", "hi": "chatterbox", "ur": "chatterbox_hi"}
```

`POST /synthesize` takes `{text, lang, ref_id}` plus the reference wav (upload it once per job and cache it by hash) and returns a 24 kHz WAV. Check each library's current README for extra arguments (Chatterbox model version, `exaggeration`, `cfg_weight`).

### 4.8 Splice

```python
# speech/splice.py
import numpy as np, librosa, pyloudnorm as pyln

def snap(y, sr, t, win=0.08, hop_ms=5):
    """Move a cut point to the quietest 5 ms frame within ±80 ms (avoids clicks mid-phoneme)."""
    a, b = max(0, int((t - win) * sr)), min(len(y), int((t + win) * sr))
    h = max(1, int(sr * hop_ms / 1000)); seg = y[a:b]
    if len(seg) <= h: return t
    e = [float(np.sum(seg[i:i + h] ** 2)) for i in range(0, len(seg) - h, h)]
    return (a + int(np.argmin(e)) * h) / sr

def fit(w, slot):
    r = len(w) / max(1, slot)
    return librosa.effects.time_stretch(w, rate=r) if 0.8 <= r <= 1.25 else w   # else timeline shifts

def match_loud(w, ref, sr):
    m = pyln.Meter(sr)
    if len(ref) < 0.5 * sr or len(w) < 0.5 * sr: return w
    lw, lr = m.integrated_loudness(w), m.integrated_loudness(ref)
    return pyln.normalize.loudness(w, lw, lr) if np.isfinite(lw) and np.isfinite(lr) else w

def xjoin(parts, sr, ms=25):
    n, out = int(sr * ms / 1000), parts[0]
    for p in parts[1:]:
        k = min(n, len(out), len(p))
        if k == 0: out = np.concatenate([out, p]); continue
        t = np.linspace(0, np.pi / 2, k)
        out = np.concatenate([out[:-k], out[-k:] * np.cos(t) + p[:k] * np.sin(t), p[k:]])
    return out

def splice(y, sr, edits):
    """edits: sorted, non-overlapping dicts with start, end (s), wav (np at sr) + metadata."""
    parts, cur, shift, log = [], 0, 0.0, []
    for e in edits:
        s, t = snap(y, sr, e["start"]), snap(y, sr, e["end"])
        a, b = int(s * sr), int(t * sr)
        w = match_loud(fit(e["wav"], b - a), y[a:b], sr)
        parts += [y[cur:a], w]
        log.append({k: v for k, v in e.items() if k != "wav"} |
                   {"start": s, "end": t, "new_start": s + shift, "new_end": s + shift + len(w) / sr})
        shift += len(w) / sr - (t - s); cur = b
    parts.append(y[cur:])
    out = xjoin([p for p in parts if len(p)], sr)
    return np.clip(out, -1, 1), log
```

Run AudioSeal on each synthesized `w` before splicing (it works at 16 kHz: resample the copy, watermark it, resample back). Store the edit log with the job.

### 4.9 Orchestrator

```python
# pipeline/audio_job.py  (pseudocode-level; each step reports progress)
async def process(job, reg, cfg, emit):
    y_out, sr = decode(job.path); y16, _ = decode(job.path, 16000)
    emit("asr", 0.10);   lang, segs = route_asr(reg, job.wav16, y16, job.lang_hint)
    emit("detect", 0.35); p = reg.detector.score([s.text for s in segs])
    context = " ".join(s.text for s in segs)
    verdicts = {}
    for i, s in enumerate(segs):
        if p[i] >= cfg.llm_gate or lang not in cfg.clf_langs:
            v = reg.llm.judge(context, s.text, lang, job.mode)
            if v and v["severity"] != "none": verdicts[i] = v
    emit("voice", 0.55); ref = job.enrolled_ref or pick_ref(segs, set(verdicts))
    emit("tts", 0.65);   edits = []
    for i, v in verdicts.items():
        text = safe_text(reg, v, job.mode, lang)          # §4.5; None → mute
        wav = await reg.tts.synth(text, lang, ref) if text and ref else silence_like(segs[i], sr)
        edits.append({"start": segs[i].start, "end": segs[i].end, "wav": resample(wav, sr),
                      "original_text": segs[i].text, "replacement_text": text or "",
                      "severity": v["severity"], "target": v["target"], "spans": v["spans"],
                      "p_classifier": p[i], "method": "tts" if text else "muted"})
    emit("splice", 0.90); out, log = splice(y_out, sr, sorted(edits, key=lambda e: e["start"]))
    save(job, out, sr, log, segs, lang); emit("done", 1.0)
```

Extend your canonical schema with an `audio_edit` block so the history and analytics pages still work:

```json
"audio_edit": {
  "original_url": "/media/abc/original.wav",
  "cleaned_url": "/media/abc/cleaned.mp3",
  "duration_s": 61.2, "language": "ta", "mode": "rewrite", "voice_source": "auto|enrolled",
  "edits": [{"start": 12.31, "end": 15.02, "new_start": 12.31, "new_end": 14.88,
             "original_text": "...", "replacement_text": "...", "severity": "hate",
             "target": "religion", "spans": ["..."], "p_classifier": 0.91, "method": "tts", "watermarked": true}],
  "transcript": [{"start": 0.0, "end": 2.1, "text": "...", "flagged": false}]
}
```

---

## 5. Backend API (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/jobs` | multipart `file`, `mode=rewrite|counter`, `language=auto|xx`, `voice=auto|enrolled`, `keep_background=bool` → `{job_id}` |
| GET | `/api/v1/jobs/{id}` | status, stage, progress, error |
| GET | `/api/v1/jobs/{id}/events` | Server-Sent Events progress stream (the UI uses this) |
| GET | `/api/v1/jobs/{id}/result` | canonical result incl. `audio_edit` |
| GET | `/media/{job_id}/{file}` | original / cleaned audio (range requests on, for seeking) |
| POST | `/api/v1/voices/enroll` | 10–15 s recording of a shown prompt → `voice_id` |
| DELETE | `/api/v1/voices/{id}` | user can delete their voice reference |
| POST | `/api/v1/analyze/text` | keep your existing text endpoint (now backed by classifiers + LLM) |
| WS | `/ws/rooms/{room_id}` | live mode (§6) |
| GET | `/api/health`, `/api/ready` | liveness / models-loaded readiness (Docker healthchecks use these) |

Implementation notes:

- **Job worker:** an `asyncio.Queue` plus one background task, started in `lifespan`. GPU calls run in `run_in_executor` under an `asyncio.Lock`. Live utterances use a higher-priority queue. Job state lives in the DB, so a restart can mark jobs "failed: restarted" instead of losing them. When you need several GPU machines, move the queue to Redis (`arq`/`rq`) without changing the API.
- **Lazy loading + VRAM guard:** load each model on first use. On the 6 GB dev profile, keep at most one TTS engine resident and unload after use. Your `pipeline/registry.py` is the right place for this.
- **Fix in the current `server/app.py`:** `allow_origins=["*"]` with `allow_credentials=True` is rejected by browsers. Use `CORS_ORIGINS` from config. In production Caddy serves frontend and API on one origin, so CORS isn't needed at all.
- **Limits:** cap upload size and duration, rate-limit per IP (`slowapi`), delete uploads after N days (a background task).
- Remove the demo path in `pipeline/analyze.py::analyze_audio` that returns a fixed sentence, so a missing model fails loudly instead of quietly returning fake output.

---

## 6. Live mode (Phase 4)

### 6.1 Client → server

- Browser: `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}})` → an `AudioWorklet` that downsamples to 16 kHz mono Int16 and sends 20 ms frames (640 bytes) as binary WS messages.
- **Mic access only works on HTTPS or localhost.** In production Caddy handles TLS automatically.

### 6.2 Server per speaker

```python
from silero_vad import load_silero_vad, VADIterator
vad = VADIterator(load_silero_vad(), sampling_rate=16000, min_silence_duration_ms=500)
# feed 512-sample windows; on {'start': t} begin buffering, on {'end': t} or 8 s of buffer → utterance
```

For each utterance: ASR (Whisper, language fixed from the room setting to save time), then classifiers, then the LLM only if gated. Clean utterances are broadcast as the original audio. Flagged ones get TTS with the enrolled voice, are watermarked, and are then broadcast.

Broadcast message: a JSON header (`speaker`, `seq`, `text`, `edited`, `replacement_text`) followed by the binary audio, Opus or 16-bit PCM. The speaker never hears their own stream. Listeners play from a jitter buffer, scheduling `AudioBufferSourceNode`s back to back in sequence order.

### 6.3 Voice enrollment

When someone joins a room (and optionally in Studio), they read one fixed sentence in their language, shown on screen, for 10–15 s. That gives you a clean reference and an exact `ref_text`, which IndicF5 needs, and it doubles as an explicit consent step. Store it per user. Let them delete it.

### 6.4 When to move to LiveKit

The WebSocket design is fine for a demo and a few rooms. Move to LiveKit (self-hosted SFU, Python Agents framework) when you need many concurrent users, NAT traversal and proper WebRTC audio quality. In that design a "Civitas moderator" agent joins each room, subscribes to every raw track, and publishes one cleaned track per speaker. Clients subscribe only to the cleaned tracks. The pipeline code (`pipeline/live.py`) is reused unchanged inside the agent.

---

## 7. Frontend (React + Vite + TS, already in `frontend/`)

Libraries to add: `react-router-dom`, `@tanstack/react-query` (jobs/polling), `wavesurfer.js` (already present; use its Regions plugin), `lucide-react` (icons), `sonner` (toasts). Keep the existing CSS/design tokens, or move to Tailwind + shadcn/ui if you want faster component building. Pick one and don't mix them.

Pages:

1. **Landing:** what Civitas does, a short before/after demo clip, CTA "Try Studio".
2. **Studio** (core page):
   - Drop zone + "Record" button (MediaRecorder, webm/opus); mode toggle *Rewrite / Counter-narrative*; language select (Auto default); "Keep background" switch; voice source (Auto from recording / My enrolled voice).
   - Progress view driven by SSE: ASR → Detect → Voice → Synthesize → Splice, with per-stage times.
   - Result view: two stacked waveforms (Original / Cleaned) with synced playhead. Replaced regions are shaded on both, and clicking a region plays it. Below that, the transcript with flagged sentences struck through, the replacement shown inline, and a target/severity chip. Buttons: Download cleaned MP3, Download edit report (JSON), "Report a wrong edit" (posts to your existing `/api/feedback`).
3. **Live room:** create/join by code; enrollment step on first join; participant tiles with speaking indicator and an "edited" flash when a replacement plays; a captions panel showing the cleaned transcript; a visible "moderation delay ~2 s" badge.
4. **History:** past jobs with filters (language, edited/clean), from your SQLite history.
5. **Analytics:** reuse your current analytics view (edits per language and target, latency percentiles).
6. **Settings / Privacy:** delete my voice, delete my uploads, consent text.

Config: `VITE_API_BASE` defaults to `""` (same origin behind Caddy). For local dev, Vite proxies `/api`, `/ws` and `/media` to `http://localhost:8000` (`server.proxy` in `vite.config.js`, with `ws: true` for `/ws`).

---

## 8. Deployment

### 8.1 Images

`docker/Dockerfile.engine`
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.engine.txt .
RUN pip install --no-cache-dir -r requirements.engine.txt
COPY . .
ENV HF_HOME=/models PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD curl -fsS http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

`requirements.engine.txt` (pin exact versions once it works): `fastapi uvicorn[standard] python-multipart pydantic-settings sse-starlette faster-whisper transformers sentencepiece silero-vad soundfile librosa pyloudnorm openai httpx audioseal ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git`

`docker/Dockerfile.tts`: same base image; `pip install chatterbox-tts` plus `pip install git+https://github.com/ai4bharat/IndicF5.git`. **Build this one first.** If pip reports a conflict between the two, split it into `Dockerfile.tts-indic` and `Dockerfile.tts-multi`; the engine only knows a URL per language.

`docker/Dockerfile.web`
```dockerfile
FROM node:22-alpine AS build
WORKDIR /w
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM caddy:2-alpine
COPY docker/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /w/dist /srv
```

`docker/Caddyfile`
```
{$DOMAIN::80} {
    encode zstd gzip
    request_body /api/v1/jobs { max_size 50MB }
    @backend path /api/* /ws/* /media/*
    handle @backend { reverse_proxy engine:8000 }
    handle { root * /srv
             try_files {path} /index.html
             file_server }
}
```
Set `DOMAIN=civitas.example.com` and Caddy gets HTTPS certificates automatically. Leave it unset for plain `:80` locally.

### 8.2 `docker-compose.yml`

```yaml
x-gpu: &gpu
  deploy:
    resources:
      reservations:
        devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]

services:
  web:
    build: { context: ., dockerfile: docker/Dockerfile.web }
    ports: ["80:80", "443:443"]
    environment: [DOMAIN=${DOMAIN:-}]
    volumes: [caddy_data:/data]
    depends_on: [engine]
    restart: unless-stopped

  engine:
    build: { context: ., dockerfile: docker/Dockerfile.engine }
    <<: *gpu
    env_file: .env
    volumes: [models:/models, data:/data]
    depends_on: [tts, ollama]
    restart: unless-stopped

  tts:
    build: { context: ., dockerfile: docker/Dockerfile.tts }
    <<: *gpu
    env_file: .env
    volumes: [models:/models]
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest      # pin a version ≥ 0.22 once tested
    <<: *gpu
    volumes: [ollama:/root/.ollama]
    restart: unless-stopped

volumes: { models: {}, data: {}, ollama: {}, caddy_data: {} }
```

`.env.example`
```
HF_TOKEN=                       # needed for gated IndicF5 + IndicConformer (accept terms on HF first)
DEVICE=cuda
WHISPER_MODEL=large-v3-turbo
WHISPER_COMPUTE=int8_float16
ENABLE_INDIC_ASR=true
LLM_BASE_URL=http://ollama:11434/v1
LLM_MODEL=gemma4:e4b
LLM_API_KEY=
LLM_GATE=0.30
TTS_URL=http://tts:9000
MAX_AUDIO_SECONDS=300
DATA_DIR=/data
DB_URL=sqlite:////data/civitas.db
CORS_ORIGINS=http://localhost:5173
DOMAIN=
```

### 8.3 First deploy on a GPU VM

1. Get a Linux VM with an NVIDIA GPU (24 GB recommended), install the NVIDIA driver, Docker and the NVIDIA Container Toolkit. Check that `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works.
2. Point your domain's A record at the VM and open ports 80/443.
3. `git clone` the repo, `cp .env.example .env`, fill in `HF_TOKEN` and `DOMAIN`.
4. `docker compose up -d --build`
5. `docker compose exec ollama ollama pull gemma4:e4b`
6. `docker compose exec engine python scripts/download_models.py` warms the HF cache volume (Whisper, classifiers, IndicConformer, aligner); the same script in the tts container fetches IndicF5 and Chatterbox.
7. Open `https://your-domain`, upload a test clip, and watch `docker compose logs -f engine`.

`scripts/download_models.py` is just `huggingface_hub.snapshot_download(repo_id, token=os.environ["HF_TOKEN"])` over a list, plus `WhisperModel("large-v3-turbo")` once to cache the CTranslate2 weights. It's idempotent, so it's safe to re-run.

Split-hosting alternative: frontend on any static host (set `VITE_API_BASE=https://api.your-domain`), engine + tts + ollama on the GPU VM. Same images.

### 8.4 Dev profile for the RTX 3050 (6 GB)

Rough VRAM estimates (measure with `nvidia-smi`): Whisper turbo int8 ~1–1.5 GB, XLM-R-large fp16 ~1.1 GB, MuRIL ~0.5 GB, IndicConformer ~1.3 GB, aligner ~0.7 GB, Chatterbox ~2–3 GB, IndicF5 ~1.5–2 GB, Gemma 4 E4B q4 ~5 GB. That's far more than 6 GB at once, so in dev:

- Run the classifiers on CPU (`DETECTOR_DEVICE=cpu`); they're fast enough for a minute of audio.
- Run the LLM with `gemma4:e2b`, or point `LLM_BASE_URL` at a hosted OpenAI-compatible API.
- Keep only one TTS engine resident and unload it between jobs.
- Run outside Docker if Docker GPU support on Windows/WSL gets in the way: `uvicorn server.app:app`, `uvicorn tts_service.app:app --port 9000`, the Ollama desktop app, and `npm run dev`. `docker-compose.dev.yml` can do the same with bind mounts.

---

## 9. Execution plan (phases with "done when")

Hand these to your coding agent one at a time (your CLAUDE.md "plan first, tests per module" rules still apply).

**Phase 0: setup (½ day)**
Accept HF terms for `ai4bharat/IndicF5` and `ai4bharat/indic-conformer-600m-multilingual`, create `HF_TOKEN`, install Ollama and pull `gemma4:e2b` (dev). Record 6 test clips (en, hi, ta, te, ml, ur), 30–60 s each, with friends reading a mix of normal sentences and HateCheck-style test sentences, with their consent.
*Done when:* every model loads in a scratch notebook and produces output on one clip.

**Phase 1: text path on real models (1 day)**
`pipeline/detect.py`, `pipeline/llm.py`, and `/api/v1/analyze/text` rewired to classifiers + LLM.
*Done when:* 30 hand-written sentences across languages give sensible verdicts, rewrites pass the safety gate, and pytest covers the JSON parsing/fallback.

**Phase 2: audio analysis without TTS (1–2 days)**
`speech/audio_io.py`, `vad.py`, `asr.py` (Whisper + Indic route), `align.py`, sentence splitter, and job queue + SSE. The output marks flagged regions and mutes them.
*Done when:* uploading each test clip returns a transcript with correct timestamps (spot-check by listening), flagged sentences, and a muted output file.

**Phase 3: voice-cloned replacement (2–3 days)**
`tts_service/`, `voice_ref.py`, `splice.py`, watermark, and the `audio_edit` schema.
*Done when:* each test clip comes back with replacements in a recognisably similar voice, with no clicks at the cuts, and the edit log matches the audio.

**Phase 4: Studio UI (2–3 days)**
Upload/record, progress, A/B waveform with regions, transcript diff, downloads, feedback.
*Done when:* a non-technical friend can use it end to end without help.

**Phase 5: deployment (1 day)**
Dockerfiles, compose, Caddy, download script, `.env.example`, deploy to a GPU VM over HTTPS.
*Done when:* `docker compose up -d` on a fresh VM plus the two download commands gives a working site.

**Phase 6: live rooms (3–4 days)**
Enrollment, WS rooms, VAD streaming, priority queue, client jitter buffer.
*Done when:* two browsers in one room, one says a flagged sentence, the other hears the replaced version within ~3 s.

**Phase 7: polish/eval**
Demucs background option, analytics, latency dashboard, report numbers (§10).

---

## 10. How to evaluate (for your report)

- **Detection:** sentence-level precision/recall/F1 on your recorded clips plus HateCheck sentences read aloud. Report the ASR → detection drop compared with detection on the gold transcript.
- **ASR:** WER/CER per language on the test clips (`jiwer`).
- **Replacement safety:** % of replacements that pass the classifier gate on the first try; % muted.
- **Voice similarity:** cosine similarity of speaker embeddings (e.g. SpeechBrain ECAPA) between reference and replacement. Include an informal 1–5 MOS from 5 listeners.
- **Naturalness at cuts:** listener A/B test ("which sentence was edited?"). Close to 50% means the edits blend in.
- **Latency:** per-stage ms (already in your `timing_ms`), p50/p95, for upload and live.

---

## 11. Pitfalls checklist

- Mic doesn't work on the deployed site: you're on HTTP. Use HTTPS (Caddy + domain).
- Clicks at edit boundaries: missing `snap()`/crossfade, or a sample-rate mismatch (resample TTS output to the file's rate).
- Replacement sounds like a different person: the reference contains music/noise/two speakers, the reference language differs from the output language, or `ref_text` doesn't match the reference audio (IndicF5/OmniVoice).
- Wrong cut positions in Tamil/Malayalam: the Whisper path was used instead of IndicConformer + aligner, or the aligner got the wrong ISO-639-3 code.
- LLM returns prose instead of JSON: the server ignored `response_format`. Upgrade Ollama, keep the retry, and give it a fallback.
- OOM on the 3050: too many models resident. Use the dev profile (§8.4).
- `pip` conflict between chatterbox-tts and IndicF5: split the TTS containers (§8.1).
- Whisper hallucinating text in silence or music: keep `vad_filter=True` and drop segments with very low confidence.

---

## Sources

- IndicF5 model card: https://huggingface.co/ai4bharat/IndicF5
- IndicConformer 600M multilingual: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
- Chatterbox (Resemble AI) README: https://huggingface.co/ResembleAI/chatterbox/blob/main/README.md
- OmniVoice: https://huggingface.co/k2-fsa/OmniVoice
- textdetox XLM-R toxicity classifier v2: https://huggingface.co/textdetox/xlmr-large-toxicity-classifier-v2
- CNERG indic-abusive-allInOne-MuRIL: https://huggingface.co/Hate-speech-CNERG/indic-abusive-allInOne-MuRIL
- MMS-300m forced aligner / ctc-forced-aligner: https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner
- Gemma 4 family: https://www.llmreference.com/model-family/gemma-4
- Gemma 4 on Ollama: https://www.verdent.ai/guides/gemma-4-ollama
- AudioSeal: https://pypi.org/project/audioseal
