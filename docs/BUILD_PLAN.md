# Civitas AI — Build Plan for Claude Code (terminal)

Speech + text hate speech detection with explanations and alternate-speech recommendations.

---

## How to use this document

1. Put `CLAUDE.md` in your repo root. Claude Code reads it automatically every session, so the rules and output schema don't need repeating.
2. Put this file at `docs/BUILD_PLAN.md`.
3. Work **one phase at a time**. For each phase:
   - Start `claude` in the repo root.
   - Press **Shift+Tab** until you're in **plan mode**, paste the phase prompt, read the plan, push back if needed, then approve.
   - When Claude says the phase is done: run the tests yourself, do the "Your checkpoint" items, commit.
   - Type `/clear` before starting the next phase so old context doesn't pollute it. The phase prompts are written to be self-contained.
4. Anything marked **YOU** is a manual step. Claude Code can't click in Colab, download gated datasets, or record your voice.

### Target architecture

```
 Mic / audio file / text
          │
   [VAD: silero-vad]  (audio only)
          │
   [ASR: faster-whisper  |  IndicConformer for ta/te/ml]  → words + timestamps
          │
   [Language ID + script normalization / transliteration]
          │
   [Model A: XLM-R 4-head]  ──► uncertain? ──► [NLI mDeBERTa] + [dehumanization lexicon]
          │
   is hate / offensive?
          │ yes
   [Model B: mT0 generator]  → Rewrite + Respond candidates
          │
   [Safety gate: re-score every candidate with Model A, drop unsafe]
          │
   FastAPI (REST + WebSocket)  →  React UI  (+ optional TTS playback)
```

### Proposed repo layout (Claude should adapt to what already exists, not force this)

```
civitas/
  model_a/        existing code: model.py dataset.py train.py infer.py nli_stage.py
                  dehumanization.py identity_augment.py converters/ build_dataset.py prepare_splits.py
  model_b/        generator data prep, inference, safety gate
  speech/         vad.py asr.py script_norm.py align.py
  pipeline/       analyze.py  (one function: text or audio in → canonical JSON out)
  server/         FastAPI app
  web/            React + Vite + TS
  notebooks/      Colab notebooks (01..05)
  eval/           robustness, functional tests, end-to-end speech eval
  data/raw/       gitignored, you download into here
  artifacts/      gitignored, checkpoints pulled from Drive
```

---

## Phase 0 — Repo audit (no code changes)

**Prompt:**

```
Read CLAUDE.md and docs/BUILD_PLAN.md. Do NOT edit any code in this phase.

Audit the repository and write docs/AUDIT.md containing:
1. The actual file tree and what each module does (1–2 lines each).
2. How data flows: converters → build_dataset.py → prepare_splits.py → train.py → infer.py.
   Include the exact CLI commands currently used, or say they don't exist.
3. Every stub, TODO, placeholder, or NotImplemented you find, with file:line.
4. Mismatches between the code and CLAUDE.md, specifically:
   - Is Malayalam ("ml") in the language list and does any converter emit it?
   - Which sources actually produce target / severity / rationale labels, and how they map
     into TARGET_CLASSES and SEVERITY_CLASSES today.
   - Which of HASOC / IEEE DataPort converters exist.
5. Current dependencies and versions (requirements / pyproject), and anything missing.
6. Row counts per source × language × label, IF processed data exists locally.
   If it doesn't, write the command I should run to produce it.
7. A short risk list: things likely to break when we add speech input.

End with a numbered list of questions you need me to answer before Phase 1.
```

**Your checkpoint:** read `AUDIT.md`, answer its questions in the chat, commit it.

---

## Phase 1 — Fix the foundations

**Prompt:**

```
Phase 1 of docs/BUILD_PLAN.md. Use docs/AUDIT.md as context. Plan first, then implement.

Goals:
A. Language codes: add "ml" (Malayalam) end to end. Add a "script" field to the unified
   schema: "native" | "latin" | "mixed", detected automatically from Unicode ranges.
   Keep backward compatibility with existing processed files (default by detection).

B. Class maps: create model_a/label_maps.py with FINAL TARGET_CLASSES and SEVERITY_CLASSES
   and a per-source mapping table (source label → unified label, or -1).
   Also write docs/LABEL_MAPPING.md as a readable table.
   Do not guess ambiguous mappings — list them for me to decide.

C. Implement normalize_code_mixed() in dataset.py:
   - leetspeak reversal (e.g. 0→o, 1→i/l, 3→e, 4→a, 5→s, @→a, $→s) ONLY inside alphabetic tokens
   - collapse spaced-out words ("h a t e" → "hate") and repeated chars (>2 → 2)
   - strip zero-width chars, normalize Unicode (NFKC), unify quotes
   - keep emojis (they carry signal)
   Must return a char-offset map so rationale spans on the normalized text can be mapped
   back to the ORIGINAL text. Write tests for the offset map.

D. Script augmentation for speech robustness (important):
   ASR outputs native script (Devanagari/Tamil/Telugu/Malayalam/Urdu Nastaliq), but much of
   our training data is romanized or code-mixed. Add an optional training-time augmentation
   in a new module model_a/script_augment.py:
   - native → latin transliteration and latin → native, with probability p (config)
   - use the `indic-transliteration` package for rule-based conversion;
     leave a clean interface so AI4Bharat IndicXlit can be plugged in later
   - Urdu script ↔ Roman-Urdu: implement a simple rule-based mapper and mark it as
     approximate in the docstring
   - rationale spans must be remapped (or set to "[]" with rationale label masked if
     remapping is impossible — the masked loss already handles missing labels)

E. Tests for all of the above. Run them.

Don't touch compute_multitask_loss or split logic.
```

**Your checkpoint:** decide the ambiguous label mappings it lists. Spot-check 20 transliterated samples per language by eye — bad transliteration silently poisons training.

---

## Phase 2 — New dataset converters

### YOU: download the data first

For each dataset below: find the official source (paper page / GitHub / Hugging Face / shared-task site), read the license, download into `data/raw/<name>/`. Some need a request form or shared-task registration. Fill the `SOURCES.md` table Claude creates with the real URL and license.

### Recommended datasets

**Detection — fill your weakest gaps first (Telugu, Malayalam, Hindi native script)**

| Dataset | Languages | Why you need it |
|---|---|---|
| HASOC 2019 / 2020 / 2021 | en, hi (+mr) | Already in your plan; converter missing. 2021 has rationale spans. |
| DravidianLangTech shared tasks — offensive / hate / abusive tracks (incl. the Telugu hate speech track and the Tamil/Malayalam abusive-comment tracks) | ta, ml, te | Telugu and Malayalam look underrepresented. These are the main public sources. |
| CONSTRAINT 2021 Hindi Hostility | hi (Devanagari) | Native-script Hindi — matches what ASR produces. Has fine-grained hostile classes that can feed target/severity. |
| IEEE DataPort (Razi & Ejaz) | per your docs | Already planned; implement the converter. |
| HateCheck (English) and Multilingual HateCheck (includes Hindi) | en, hi | **Test-only.** Functional test cases (negation, counter-speech, reclaimed slurs, quoting). Never train on them. |

**Speech — to evaluate (and optionally train) on real audio**

| Dataset | What it is |
|---|---|
| ADIMA (ShareChat, ICASSP 2022) | Abusive audio clips across ~10 Indic languages. The most on-target speech dataset for you. Check access terms. |
| MuTox (Meta) | Multilingual audio toxicity annotations; check its language list for your languages. |
| DeToxy | English speech toxicity benchmark. |
| Kathbath / IndicVoices (AI4Bharat) | Not hate data — use them to measure ASR word-error-rate per language so you know how much ASR errors cost you. |

**Alternate speech — for Model B**

| Dataset | Use |
|---|---|
| CONAN, Multi-CONAN, Multitarget-CONAN, DIALOCONAN | Expert-written hate → counter-narrative pairs (English + a few European languages). Core of "Respond". |
| Gab & Reddit counter-speech benchmark (Qian et al., 2019) | Hate posts with crowd-written interventions. |
| ParaDetox + the multilingual TextDetox 2024 data (includes Hindi) | Toxic → non-toxic parallel rewrites. Core of "Rewrite". |
| Your own silver data | Indic counter-speech data is scarce. Machine-translate CONAN/ParaDetox pairs with IndicTrans2 in Colab, then hand-check a sample per language (Phase 4). |

Verify every dataset's license allows your use (course project vs. public demo vs. commercial differ).

**Prompt:**

```
Phase 2 of docs/BUILD_PLAN.md.

1. Create data/SOURCES.md: a table with columns name | languages | local path | official URL |
   license | labels available | status. Leave URL and license as "TODO(user)" — do not invent them.

2. Write converters in model_a/converters/ for every dataset folder that exists under data/raw/.
   For each one:
   - First inspect the actual files (columns, delimiters, encodings, label values) and print
     a summary. Do not assume a schema.
   - Map to the unified schema using model_a/label_maps.py. Unknown labels → -1, not a guess.
   - Set language and script fields correctly; code-mixed text gets the base language code.
   - Test-only sets (HateCheck, Multilingual HateCheck) go to data/processed/functional_tests/
     and must never enter the training pool — add an assert in build_dataset.py.

3. Add configurable per-source caps (like SBIC/DynaHate at 15k) and per-language
   upsampling weights in a config file, so English implicit-hate sets don't swamp Indic data.

4. Re-run build_dataset.py and prepare_splits.py. Write docs/DATA_REPORT.md with row counts by
   source × language × hate_label, and counts of non-missing target/severity/rationale labels
   per language. Flag any language with < 3,000 training rows or any target class with < 200.

5. Package the processed splits for Colab: one command that produces
   artifacts/civitas_data_v2.zip containing train/val/test parquet files + label_maps.py
   + a manifest.json (row counts, git commit hash, creation date).
```

**Your checkpoint:** read `DATA_REPORT.md`. If Telugu or Malayalam are still thin, find more data before training — no model trick fixes missing data.

---

## Phase 3 — Retrain Model A in Google Colab

**Prompt:**

```
Phase 3 of docs/BUILD_PLAN.md. Write Colab notebooks; you cannot run them.

Create notebooks/02_train_model_a.ipynb. Requirements:
- First markdown cell: a numbered "WHAT YOU DO" checklist for me (runtime type, uploads, etc.).
- Cell: check GPU with nvidia-smi; print whether it is T4/L4/A100 and set defaults accordingly
  (batch size, grad accumulation, fp16 vs bf16; T4 has no bf16).
- Cell: mount Google Drive; project folder /content/drive/MyDrive/civitas/.
- Cell: install pinned requirements; clone my repo from GitHub (ask me for the URL as a form field)
  OR unzip an uploaded code zip — support both.
- Cell: unzip civitas_data_v2.zip from Drive; verify manifest row counts.
- Cell: training config as a visible dict (lr, epochs, max_len=128, loss weights per head,
  focal loss gamma, script_augment p, identity_augment modes).
- Training must:
  * checkpoint to Drive every N steps AND keep the best-on-val checkpoint
  * auto-resume from the latest Drive checkpoint if the session restarted
  * log per-epoch val metrics: hate macro-F1, per-language hate F1, severity macro-F1,
    target macro-F1 (on rows that have target labels), rationale token-F1
  * write metrics to Drive as JSON + a CSV history
- Final cell (separate, clearly labelled "RUN ONCE AT THE VERY END"): evaluate best checkpoint on
  the test split and write test_report.json. Warn in markdown that re-running this after
  further tuning invalidates the test set.
- Also run the functional test suites (HateCheck / Multilingual HateCheck) and report
  per-functionality accuracy.

Create notebooks/04_export.ipynb:
- Export best Model A checkpoint to ONNX, then dynamic int8 quantization.
- Compare PyTorch vs ONNX-int8 outputs on 200 val rows (max abs prob diff, label agreement).
- Save to Drive: model_a_fp32.onnx, model_a_int8.onnx, tokenizer files, label_maps.py, config.

Also write docs/COLAB_GUIDE.md: step-by-step for me, including what to do when Colab disconnects,
how long each run should roughly take on a T4, and exactly which files to download back into
artifacts/ and which numbers to paste back to you.
```

### YOU: running it in Colab

1. Upload `civitas_data_v2.zip` to `MyDrive/civitas/`.
2. Open the notebook in Colab → Runtime → Change runtime type → GPU (T4 on free tier; L4/A100 if you have Pro).
3. Run all cells. If it disconnects, reconnect and run all again — it should resume from the Drive checkpoint.
4. Don't run the "RUN ONCE" test cell until you're done tuning.
5. Download the export files into `artifacts/model_a/`.
6. Paste `metrics.json` (val) back into Claude Code with: *"Here are the val metrics. Diagnose weak spots and propose at most 3 changes."*

---

## Phase 4 — Model B: alternate speech (Rewrite + Respond)

**Prompt:**

```
Phase 4 of docs/BUILD_PLAN.md. Plan first.

Model B generates two kinds of alternate speech:
- REWRITE: keep the speaker's underlying point (if any legitimate one exists) but remove hate,
  slurs, dehumanization and generalizations about groups. Same language/script as input.
- RESPOND: a short (1–3 sentences), calm, non-preachy counter-narrative that corrects the
  generalization with facts or empathy. Never insults the speaker. Same language as input.

1. Data prep (model_b/prepare_data.py):
   - Convert CONAN family, Qian et al. Gab/Reddit, ParaDetox + TextDetox data (whatever exists in
     data/raw/) into one format: {task: "rewrite"|"respond", source_text, target_text, language}.
   - Input prompt format for the model: "<task> <lang> <target_group>: <text>"
     (target_group from Model A labels; "unknown" if missing).
   - Write notebooks/03a_translate_silver.ipynb: translate English pairs into hi, ta, te, ml, and
     Roman-Urdu (via Urdu then transliteration) with IndicTrans2 in Colab. Save to Drive.
     Mark these rows source="silver_mt".
   - Write a small CLI tool model_b/review_silver.py that shows me random silver pairs one at a
     time and lets me mark keep / fix / drop, saving decisions. I will review ~100 per language.

2. Training notebook notebooks/03_train_model_b.ipynb:
   - Base model: bigscience/mt0-base (instruction-tuned mT5). Offer mt0-small as a fallback.
   - IMPORTANT: mT5-family models are known to produce NaN losses in fp16. On T4 train in fp32
     with Adafactor and gradient accumulation; use bf16 only on A100/L4.
   - Same Drive checkpointing + auto-resume pattern as notebook 02.
   - Eval: BLEU/chrF against references, plus a SAFETY metric: % of generated outputs that the
     exported Model A scores as hate (target: near 0) and % that are exact copies of the input.
   - Export to Drive.

3. Inference with a safety gate (model_b/generate.py):
   - generate k=4 candidates per task (sampling), re-score each with Model A,
     drop any with hate_prob above a configurable threshold or high overlap with the
     rationale spans of the input, return the best 1–2.
   - If nothing passes, return a safe templated fallback per language and set a flag
     "fallback": true. Never return an unchecked generation.
   - Tests with a mocked Model A.
```

**Your checkpoint:** review the silver pairs (the CLI tool). Then read 30 real generations per language yourself. Generation quality in Tamil/Telugu/Malayalam will be the weakest part of the system — say so honestly in any report.

---

## Phase 5 — Unified inference pipeline (text first)

**Prompt:**

```
Phase 5 of docs/BUILD_PLAN.md.

Create pipeline/analyze.py with one public function:
  analyze_text(text, lang_hint=None) -> dict matching the canonical JSON in CLAUDE.md.

Steps inside:
1. language + script detection (fastText lid.176 or a lightweight script-range heuristic;
   lang_hint wins if given)
2. normalize_code_mixed with offset map
3. Model A via ONNX Runtime (CUDA provider if available, else CPU int8)
4. uncertainty band → NLI stage + dehumanization lexicon (reuse existing modules; band from config)
5. decision_path recorded
6. if hate or offensive → Model B with safety gate
7. rationale spans mapped back to ORIGINAL text offsets
8. timing_ms for every stage

Also:
- A ModelRegistry that lazy-loads each model once and reports VRAM use; must fit on a 6 GB
  RTX 3050 with Model A + NLI + Model B + ASR loaded, or fall back to CPU for NLI/Model B.
  Measure and write numbers to docs/PERF.md.
- CLI: python -m pipeline.analyze "some text" --pretty
- Tests with small fixture texts per language, including the Bronzites example and a
  non-hate text that mentions a religion neutrally (to catch false positives on identity terms).
```

---

## Phase 6 — Speech pipeline

**Prompt:**

```
Phase 6 of docs/BUILD_PLAN.md. Plan first; explain trade-offs before choosing.

Build speech/ :
1. vad.py — silero-vad; split audio into speech segments, drop silence.
2. asr.py — pluggable ASR backends behind one interface returning
   {text, language, words: [{word, start, end, prob}], avg_confidence}:
   a. faster-whisper (default; model size from config; int8_float16 on GPU to fit 6 GB)
   b. an AI4Bharat Indic ASR backend (IndicConformer or IndicWhisper) for ta/te/ml/hi —
      write the adapter and document setup; if it can't run locally, make it optional.
   Language: auto-detect, but allow the user to lock the language from the UI.
3. script_norm.py — after ASR, produce BOTH the native-script transcript and a romanized
   variant; run Model A on the variant that matches how that language dominates our training
   data (decide from DATA_REPORT.md; make it configurable), or on both and take the max
   hate_prob (config flag). Record which was used.
4. align.py — map rationale char spans → words → audio_start/audio_end using ASR word
   timestamps. Handle subword/char offset mismatch carefully; test it.
5. Low ASR confidence (config threshold): still analyze, but add "low_asr_confidence": true
   so the UI can warn that the transcript may be wrong.
6. pipeline/analyze.py gets analyze_audio(path_or_bytes, lang_hint=None) returning the same
   canonical JSON with input.mode="audio".
7. Streaming: a StreamSession class that accepts 16 kHz mono PCM chunks, runs VAD, and emits
   an analysis result per completed utterance (not per chunk).
8. Optional TTS (speech/tts.py): speak a chosen suggestion. Start with a browser-side fallback
   (Web Speech API in the UI) and leave a server-side interface for an Indic TTS model later.

Tests: generate short test WAVs with a TTS or include 3 tiny fixture clips; test VAD splitting,
alignment math, and the end-to-end call with ASR mocked.
```

**Your checkpoint:** record 5 clips per language on your phone (neutral, offensive, and implicit hate you script yourself — use invented group names like "Kelvarians" so you're not voicing real slurs). Run `analyze_audio` on them and read the transcripts first. If ASR is garbling a language, fix ASR before blaming Model A.

---

## Phase 7 — Backend API

**Prompt:**

```
Phase 7 of docs/BUILD_PLAN.md.

FastAPI app in server/:
- POST /api/analyze/text        {text, lang_hint?}               → canonical JSON
- POST /api/analyze/audio       multipart file (wav/mp3/m4a/webm) → canonical JSON (+ word list)
- WS   /ws/stream               client sends 16 kHz PCM chunks; server sends
                                {type:"partial_transcript"} and {type:"utterance_result", ...}
- POST /api/suggest             {text, task:"rewrite"|"respond", lang} → more suggestions
- POST /api/feedback            {analysis_id, correct_label?, bad_suggestion?, note}
                                → stored in SQLite (feedback table) for future retraining
- GET  /api/history             last N analyses (text, verdict, time) from SQLite
- GET  /api/health              models loaded, device, VRAM, versions

Requirements: pydantic models mirroring the canonical JSON; models loaded once at startup;
file size + duration limits; audio converted with ffmpeg; CORS for the Vite dev server;
structured logging without storing raw audio unless a config flag is on (privacy default OFF).
Tests with httpx TestClient and mocked pipeline. Add a Makefile or scripts: `make dev-api`.
```

---

## Phase 8 — Web UI

**Prompt (paste the whole block, including the design brief):**

```
Phase 8 of docs/BUILD_PLAN.md. Build web/ with React + Vite + TypeScript. Plan the design
first (tokens + ASCII wireframe), show me, then build.

WHO USES IT: content moderators, teachers, and community managers reviewing speech from
voice notes, livestream clips, or typed comments. They need to understand WHY something was
flagged and have a better thing to say, quickly. The tone is a calm, trustworthy review tool,
not a flashy AI demo and not an alarm system.

LAYOUT (desktop), one screen, three columns; stacks vertically on mobile:
  LEFT — Input
    - Tabs: Speak (live mic) | Upload audio | Type text
    - Speak: a large record button with a live input-level meter; elapsed time; stop.
    - Upload: drag-and-drop, accepted formats listed plainly, file duration shown.
    - Language: "Auto-detect" default, or lock to one of the 6 languages.
  CENTER — Transcript
    - Waveform of the audio (wavesurfer.js) with flagged regions shaded; clicking a flagged
      word seeks the audio there and plays it.
    - Transcript text with rationale words underlined + tinted by rationale score;
      hover shows the score.
    - Toggle "Show native script / Show romanized".
    - If low ASR confidence: an inline note "The transcript may be inaccurate — check it
      before acting", and let the user edit the transcript and re-analyze.
    - Sensitive-content blur ON by default for flagged utterances; "Reveal" button.
  RIGHT — Verdict
    - Verdict in plain words: "Hate speech", "Offensive", or "No issue found".
    - Severity as a 3-step scale (normal → offensive → hate) with the current step marked
      by position AND text, not color alone.
    - Target group, with confidence.
    - "Why": top rationale phrases; and if the second stage ran, a line like
      "Second check: describes a group as a disease (confidence 0.74)".
    - Collapsible "Model details": probabilities, decision_path, timings.
  BOTTOM (full width) — Alternate speech
    - Two groups: "Say it differently" (Rewrite) and "Reply with" (Respond).
    - Each suggestion: text, Copy, Play aloud (Web Speech API), "Not helpful" feedback.
    - "More suggestions" button calls /api/suggest.
  Also: a History drawer (last analyses) and a "This was wrong" feedback button on the verdict.
  Live mic mode: each finished utterance appends a row to a running session timeline.

DESIGN SYSTEM:
  - Typefaces: Atkinson Hyperlegible for UI text; the Noto Sans family for Indic scripts
    (Noto Sans Devanagari, Tamil, Telugu, Malayalam, and Noto Nastaliq Urdu) via Google Fonts,
    so every language renders correctly. Test each script visibly.
  - Palette (starting point, refine it): ink #1F2A44, surface #F6F7F9, panel #FFFFFF,
    muted text #5B6475, suggestion accent (calm teal) #2F7A83, offensive amber #B7791F,
    hate brick #9B2C2C. Severity must also be carried by text/position, never color alone.
    Provide a dark theme via CSS variables and prefers-color-scheme.
  - Avoid generic AI-dashboard patterns: no gradient washes, no identical card grid with
    the same shadow, no all-caps eyebrow labels, no emoji decoration. Hierarchy comes from
    type size, weight, and spacing.
  - One deliberate motion moment: when an analysis lands, flagged words tint in sequence
    along the transcript. Respect prefers-reduced-motion.
  - Copy: sentence case, plain verbs. Buttons say what they do ("Analyze", "Copy reply",
    "Play aloud"). Errors say what happened and how to fix it (e.g. "Microphone blocked.
    Allow microphone access in your browser's site settings, then try again.").
  - Accessibility: keyboard reachable everything, visible focus, ARIA live region announcing
    the verdict, WCAG AA contrast in both themes.

TECH: Vite + React + TS, CSS modules or Tailwind (your call, justify it), wavesurfer.js,
AudioWorklet to capture mic → 16 kHz PCM → WebSocket. A typed API client generated from
(or mirroring) the server's pydantic models. Loading and empty states for every panel.
A mock mode (VITE_MOCK=1) using fixture JSON so the UI works without the backend.

When built, run it in mock mode, take screenshots if you can, critique against this brief,
and fix the top 3 issues before telling me it's done.
```

**Your checkpoint:** use it for 15 minutes as if you were a moderator. Write down every moment you hesitated or were confused, and paste the list back as the next prompt.

---

## Phase 9 — Evaluation & robustness

**Prompt:**

```
Phase 9 of docs/BUILD_PLAN.md.

Build eval/:
1. perturb.py — style perturbations: leetspeak, spacing, char repetition, emoji insertion,
   romanize/nativize script, identity-term nonce swap, adding polite framing
   ("With respect, ..."), negation, quoting ("He said '...' and that's wrong").
2. flip_rate.py — for each perturbation, % of val predictions whose hate label flips.
   Report per language. Negation and quoting SHOULD change the label; the rest should not —
   report them separately.
3. functional.py — run HateCheck / Multilingual HateCheck test suites, per-functionality accuracy.
4. speech_e2e.py — run analyze_audio on any available speech sets (ADIMA/MuTox/DeToxy) and on
   my recorded clips; report ASR WER where references exist, and hate F1 on transcripts vs on
   gold text, so we see how much ASR costs us per language.
5. calibration.py — reliability diagram for hate_prob; pick the NLI uncertainty band from val
   data instead of hand-set numbers.
6. One command: `make eval` → docs/EVAL_REPORT.md with tables and short plain-language findings.
```

---

## Phase 10 — Package and demo

**Prompt:**

```
Phase 10 of docs/BUILD_PLAN.md.

1. Dockerfiles for server (CUDA and CPU variants) and web; docker-compose.yml that mounts
   artifacts/ read-only. `docker compose up` must work on a fresh machine given artifacts/.
2. README.md: what it does, architecture diagram (mermaid), setup for local + Docker, where
   training happens (Colab notebooks, in order), known limitations per language, and an
   ethics section (false positives on identity mentions, dialect bias, human review required,
   no raw audio stored by default).
3. scripts/demo.sh: starts everything and opens the UI with 6 example inputs (one per
   language) preloaded in the History drawer.
4. A model card for Model A and Model B in docs/ (data, metrics per language, limitations).
5. Final full test run; fix failures; tag release v1.0.
```

---

## Things that will bite you (read before starting)

- **Script mismatch is the #1 speech risk.** ASR gives native script; your training data is heavily romanized/code-mixed. Phase 1D and Phase 6.3 exist for this. Test it early with your own recordings.
- **Your `hate_label` is really "abusive/offensive."** Keep the severity head as the source of truth for "hate" in the UI, not the binary head alone.
- **Telugu and Malayalam data are thin.** Check the Phase 2 report before spending Colab hours.
- **Model B can repeat the hate.** The safety gate isn't optional.
- **Colab free tier disconnects.** Everything must checkpoint to Drive and resume.
- **Human review stays in the loop.** The UI supports decisions; it doesn't make them.
