---
title: Civitas AI
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
pinned: false
short_description: Multilingual hate speech detection and rewriting for speech
---

# Civitas AI

Multilingual hate-speech detection for **speech and text** (English, Hindi, Roman-Urdu, Tamil,
Telugu, Malayalam). It explains each decision and suggests alternate speech: a **rewrite** of what
the speaker said and a calm **counter-narrative** reply. In Voice Studio, hateful sentences are
re-voiced in the speaker's own (cloned) voice, while everything else stays as recorded.

**Pipeline:** faster-whisper ASR → toxicity classifiers (XLM-R large, MuRIL) → LLM judge/rewriter →
voice-cloning TTS (Chatterbox for en/hi, IndicF5 for Indian languages).

> Runs on a free 2-core CPU Space, so it is slow: text checks take tens of seconds and a re-voiced
> sentence can take a few minutes. After a restart, models need several minutes to load.

Voice references you upload are used only to synthesise the rewritten sentences and are not kept
across restarts.
