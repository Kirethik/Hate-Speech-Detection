# Civitas Voice: run it on your laptop (Windows, Intel Core Ultra + Arc)

Voice Studio takes a recording, finds hate speech, rewrites those sentences, re-speaks them in the
speaker's own voice, and splices them back in. Everything uses downloaded pretrained models; no training.

## What runs where on an Intel machine

| Part | Model | Runs on | Process |
|---|---|---|---|
| Speech-to-text (en, hi, ur, auto language ID) | faster-whisper `large-v3-turbo`, int8 | CPU (Core Ultra is fast at int8) | API |
| Speech-to-text (ta, te, ml, kn, bn, mr...) | AI4Bharat IndicConformer 600M (ONNX) | CPU | API |
| Hate classifiers | `textdetox/xlmr-large-toxicity-classifier-v2` + `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` | Arc GPU (PyTorch XPU), else CPU | API |
| Judge + rewrite + counter-narrative | Gemma 4 E4B via Ollama | Arc GPU via Vulkan (Ollama picks it up), else CPU | Ollama |
| Voice cloning (ta, te, ml, kn, bn, mr...) | IndicF5 | Arc (XPU) if it works, else CPU | TTS :9001 |
| Voice cloning (en, hi, ur, +20) | Chatterbox Multilingual | Arc (XPU) if it works, else CPU | TTS :9002 |

Each TTS engine runs as its own small server in its own Python environment, because their pinned
dependencies conflict. The API talks to them over HTTP.

RAM: about 10–14 GB at peak with everything loaded. 16 GB works; 32 GB is comfortable.

## One-time setup

Install these first:

1. **Intel Arc graphics driver** (latest, from intel.com). Needed for both PyTorch XPU and Ollama.
2. **Python 3.11** from python.org. Tick "py launcher".
3. **Node.js 20+** from nodejs.org.
4. **Git**.
5. **Ollama** from ollama.com (recent versions use the Arc GPU through Vulkan automatically).
6. A **Hugging Face account**. Open both pages below, click **Agree**, then create a *read* token under
   Settings → Access Tokens:
   - https://huggingface.co/ai4bharat/IndicF5
   - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

Then, in PowerShell from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

It creates `.venv`, `.venv-tts-chatterbox` and `.venv-tts-indicf5`, installs the frontend, asks for
your HF token (saved to `.env`), pulls the Gemma model and downloads the other models. Expect 15–30
minutes and roughly 15 GB of downloads the first time.

Options: `-Cpu` (skip the Intel GPU builds), `-SkipIndicF5` (English/Hindi only), `-SkipChatterbox`.

## Start

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1
```

Four windows open: IndicF5 (:9001), Chatterbox (:9002), API (:8000) and web (:5173).
Open **http://localhost:5173**. Voice Studio is the first tab. The pill at the top right turns green
when the LLM and at least one TTS engine are reachable.

To check everything from a terminal:

```powershell
.\.venv\Scripts\python.exe scripts\check_system.py
```

Fast wiring test with no voice models: `scripts\run_dev.ps1 -DummyTts` (replacements become a tone).

## What each tab does (all on real models)

| Tab | What it does |
|---|---|
| **Voice Studio** | Upload or record → hateful sentences are rewritten and re-voiced in the speaker's voice → compare original and cleaned audio, download. |
| **Live Room** | Several people talk in a room. Each sentence is checked; others hear hateful sentences re-voiced (a few seconds of delay). Use headphones. |
| **Analyzer** | Text, mic or file → verdict, target group, reason, highlighted words, rewrite and counter-narrative. |
| **Rewrite** | Any message → alternatives in 5 tones, each re-checked by the classifiers. |
| **Moderation / Analytics** | History of analyses, moderator feedback, counts by language. |
| **Evaluation** | Real accuracy / F1 / confusion matrix from `scripts\evaluate.py` on your labelled data. |
| **Settings** | What is loaded and running (classifiers, LLM, voice engines, device). |

`CIVITAS_ENGINE=pretrained` (the default) makes the original Analyzer/Rewrite endpoints use these
pretrained models. `CIVITAS_ENGINE=legacy` switches back to the Colab Model A/B code path.

## Try it

1. Record or upload 30–60 s of speech that includes one clearly hateful sentence among normal ones.
2. Tick the consent box and click **Clean this audio**.
3. Compare the *Original* and *Cleaned* waveforms. Red marks what was removed, teal what was re-voiced.

The first job is slow because the models load. Later jobs reuse them.

### Live Room

1. Open **Live Room**, type a name, tick the consent box, **Join room**.
2. Send the invite link (or room code) to a second person, or open it in a second browser window
   on the same computer. Both of you should use headphones.
3. Speak in sentences with short pauses. Others hear each sentence a few seconds later; hateful ones
   are replaced. Enroll your voice first (Voice Studio → Voice → Enroll) so replacements use your
   voice from the first sentence; otherwise Civitas learns it from your first ~3 s of clean speech.

To test with someone on another device on your Wi-Fi, they need `https://` for the microphone;
for now, testing in two windows on your own machine is the simplest.

### Evaluation (real numbers for your report)

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --split val --per-language 300            # classifiers, a few minutes
.\.venv\Scripts\python.exe scripts\evaluate.py --split val --per-language 300 --with-llm # + LLM judge, slower
```

Then open the **Evaluation** tab. Use `val` while tuning; run `--split test` once at the end for the
numbers you report (it reuses the threshold chosen on val).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

These use small fake models, so they run in seconds without downloads.

## How a recording is processed

```
upload → ffmpeg decode → language ID (Whisper)
       → Whisper (en/hi/ur) or VAD chunks + IndicConformer (Dravidian/Indic)
       → sentence splitting
       → classifiers on every sentence (fast)
       → Gemma judges only sentences with p ≥ 0.30 (or languages the classifiers don't cover)
            returns: severity, target, spans, rewrite, counter-narrative, tts_text
       → safety gate: the replacement text must score < 0.5, else one retry, else mute
       → voice reference: 4–12 s of the speaker's clean speech (or the enrolled voice)
       → IndicF5 / Chatterbox synthesise the replacement in that voice
       → splice: cut at the quietest point, time-stretch to fit (±25%), match loudness, crossfade
       → original.mp3 + cleaned.mp3 + edit list (shown in the UI and downloadable as JSON)
```

Code map: `voice/` (pipeline), `server/voice_routes.py` (REST + SSE under `/api/v1`),
`tts_service/` (TTS servers), `frontend/src/components/VoiceStudio/` (UI).

## Tuning (all in `.env`)

| Setting | Default | When to change |
|---|---|---|
| `CIVITAS_WHISPER_MODEL` | `large-v3-turbo` | `small` for 3–4× faster ASR at lower accuracy |
| `CIVITAS_LLM_MODEL` | `gemma4:e4b` | `gemma4:e2b` if Ollama is slow or memory is tight |
| `CIVITAS_LLM_GATE` | `0.30` | lower for more recall (more LLM calls), higher for speed |
| `CIVITAS_ALWAYS_LLM` | `false` | `true` sends every sentence to the LLM (best recall for implicit hate) |
| `CIVITAS_ENABLE_INDIC_ASR` | `true` | `false` to use Whisper for every language |
| `CIVITAS_DETECTOR_DEVICE` | `auto` | `cpu` if the XPU build misbehaves |
| `CIVITAS_TTS_DEVICE` | `auto` | `cpu` if a TTS engine errors on the Arc GPU |
| `CIVITAS_LLM_BASE_URL` | Ollama | any OpenAI-compatible server (LM Studio, llama.cpp server, hosted) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Pill says "Voice setup needed" | Hover it to see what's missing. Start Ollama (`ollama serve`) or the TTS windows. |
| Every flagged sentence is **muted**, not re-voiced | No TTS engine for that language is running, or the TTS window shows an error. Check `scripts\check_system.py`. |
| `FAIL ai4bharat/...: gated model` | Click **Agree** on the model page and put the token in `.env` as `HF_TOKEN=`. |
| `Intel XPU available: False` | Update the Arc driver and reboot. Everything still works on CPU, just slower. |
| A TTS engine crashes on XPU | Set `CIVITAS_TTS_DEVICE=cpu` in `.env` and restart that window. |
| Ollama is very slow | Check `ollama ps`. If it says 100% CPU, update Ollama/driver or use `gemma4:e2b`. |
| Microphone does nothing | The browser blocked it. Allow mic access in the address bar (localhost is allowed). |
