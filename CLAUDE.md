# CLAUDE.md — Civitas AI

Read this file at the start of every session. It is the standing brief for this repo.

## What we are building

Civitas AI: a multilingual hate speech detector that works on **speech and text**, explains its decision, and recommends **alternate speech**:

1. **Rewrite** — a non-hateful version of what the speaker said (for the speaker / a moderation nudge).
2. **Respond** — a calm counter-narrative reply (for a bystander / moderator).

Languages: English, Hindi, Roman-Urdu, Tamil, Telugu, Malayalam (plus code-mixed variants).

## Components

- **Model A** (exists): `xlm-roberta-base` shared encoder + 4 heads (hate binary, target group 8-class, severity 3-class, token rationale). Masked multi-task loss. Second stages: dehumanization lexicon (`dehumanization.py`) and NLI (`mDeBERTa-v3-base-mnli-xnli`) that runs only inside the uncertainty band.
- **Model B** (to build): seq2seq generator (mT0/mT5 family) for Rewrite and Respond, gated by Model A so it never emits hate.
- **Speech pipeline** (to build): VAD → ASR with word timestamps → language/script normalization → Model A → Model B → optional TTS.
- **Server**: FastAPI (REST + WebSocket for live mic).
- **Web UI**: React + Vite + TypeScript.

## Hardware reality

- Local: RTX 3050 **6 GB**. Local is for inference, tests, and small debug runs only.
- **All real training happens in Google Colab.** You (Claude Code) cannot run Colab. You write the notebooks; the user runs them and reports results back. Every notebook must be runnable top-to-bottom, save checkpoints to Google Drive, and be resumable after a disconnect.

## Rules for how you work here

1. **Plan before editing.** For every phase, first propose a plan (files to touch, why, risks) and wait for approval.
2. **Do not silently change** `model.py`, `compute_multitask_loss`, `prepare_splits.py` leakage/dedup logic, or class maps. Propose the change, explain the impact, wait.
3. **Never leak eval data.** The `test` split is used once for final reporting. Checkpoint selection uses `val` only.
4. **Datasets:** never invent dataset URLs, column names, or licenses. If you don't know a dataset's exact format, write the converter against a documented expected schema and tell the user what to verify. Raw data lives in `data/raw/<source>/` and is gitignored.
5. **Every new module gets tests** (`pytest`). Run them before saying a phase is done.
6. **Commit at the end of each phase** with a clear message, after tests pass.
7. When a step needs the user (download data, run Colab, record audio, paste metrics), stop and say exactly what to do, as a numbered checklist, and what to paste back.
8. Explanations: plain language and pseudocode, not dense math.
9. Keep GPU memory in mind: fp16/int8 inference, lazy-load models, one model instance per process.

## Canonical analysis output (all layers must produce/consume this)

```json
{
  "input": {"mode": "text|audio", "text": "...", "language": "ta", "script": "latin|native", "asr_confidence": 0.91},
  "detection": {
    "hate_prob": 0.87,
    "is_hate": true,
    "severity": {"label": "hate", "probs": {"normal": 0.05, "offensive_profanity": 0.08, "hate": 0.87}},
    "target": {"label": "religion", "probs": {}},
    "rationale": [{"start_char": 4, "end_char": 17, "text": "...", "score": 0.8, "audio_start": 1.2, "audio_end": 1.9}],
    "second_stage": {"nli_ran": true, "nli_score": 0.74, "dehumanization_hits": ["vermin"]},
    "decision_path": "base|base+nli|base+lexicon"
  },
  "suggestions": {
    "rewrite": [{"text": "...", "language": "ta", "safety_check_hate_prob": 0.03}],
    "respond": [{"text": "...", "language": "ta", "safety_check_hate_prob": 0.02}]
  },
  "timing_ms": {"asr": 0, "model_a": 0, "nli": 0, "model_b": 0}
}
```

## Known gaps (from the context doc)

- `normalize_code_mixed` is a stub.
- Malayalam is claimed as supported but `ml` is missing from the unified schema's language list — confirm and fix.
- `TARGET_CLASSES` / `SEVERITY_CLASSES` are placeholders; need a per-source mapping table.
- HASOC and IEEE DataPort converters are not implemented.
- No perturbation / robustness evaluation loop yet.
