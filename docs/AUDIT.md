# Pre-Phase-1 Repo Audit

## 1. File Tree & Module Descriptions
- `api.py`: FastAPI server for REST and WebSocket interfaces.
- `build_dataset.py`: Concatenates outputs from dataset converters into combined CSVs.
- `calibrate_nli.py`: Fits the NLI stage's decision threshold on the validation set.
- `dataset.py`: Unified dataset loader for Model A, handles tokenisation and augmentations.
- `dehumanization.py`: Secondary stage for detecting dehumanisation metaphors using a lexicon.
- `demo_ui.py`: Gradio web interface for the model.
- `eval_hatecheck.py`: Functional evaluation script for HateCheck.
- `identity_augment.py`: Implements identity-term data augmentation to mitigate unseen-group failures.
- `infer.py`: Runs inference using the trained Model A.
- `model.py`: Defines Model A architecture (multilingual encoder + multi-task heads).
- `nli_stage.py`: Uses an NLI model as a second stage for zero-shot group generalisation.
- `prepare_splits.py`: De-duplicates and cleans the concatenated dataset splits, fixing leakage and creating train/val/test sets.
- `probe_model.py`: Behavioural probe for analysing Model A outputs.
- `train.py`: Main training loop for Model A.
- `train_original.py`: Backup/original version of the training loop.
- `converters/dravidiancodemix.py`: Converter for Tamil/Kannada/Malayalam code-mixed dataset.
- `converters/dynahate.py`: Converter for DynaHate.
- `converters/hatexplain.py`: Converter for HateXplain.
- `converters/implicit_hate.py`: Converter for Implicit Hate Corpus.
- `converters/macd.py`: Converter for MACD (ShareChatAI) dataset.
- `converters/ruhsold.py`: Converter for Roman Urdu Hate Speech dataset.
- `converters/sbic.py`: Converter for Social Bias Inference Corpus.
- `converters/toxigen.py`: Converter for ToxiGen dataset.
- `converters/_hf.py`: Shared utilities for Hugging Face-based converters.

## 2. Data Flow
**Flow:** converters → build_dataset.py → prepare_splits.py → train.py → infer.py
**Exact CLI Commands to prepare data:**
```bash
python build_dataset.py
python prepare_splits.py
```

## 3. Stubs, TODOs, Placeholders, NotImplemented
- `calibrate_nli.py:4`: "The placeholder 0.70 in nli_stage.py was wrong: ..."
- `dataset.py:152`: "Placeholder de-obfuscation / normalization step" in `normalize_code_mixed`
- `demo_ui.py:84`: `placeholder="Enter text to moderate..."`
- `identity_augment.py:152`: "Like \_\_call\_\_ but never passes through -..." (TestPassthrough class reference)
- `nli_stage.py` has a hardcoded placeholder 0.70 threshold.
- `converters/dravidiancodemix.py` does not currently emit `ml` (Malayalam).

## 4. CLAUDE.md Mismatches
- **Is `ml` emitted by any converter?** No, no converter currently emits Malayalam (`ml`). `dravidiancodemix.py` focuses on Tamil and Kannada, and doesn't process `ml`.
- **Which sources produce target/severity/rationale labels and what values?**
  - **Target**: 
    - `hatexplain.py` maps annotators' target lists to TARGET_CLASSES.
    - `sbic.py` maps `targetCategory` to TARGET_CLASSES.
    - `toxigen.py` maps `target_group` to TARGET_CLASSES.
  - **Severity**:
    - `hatexplain.py` maps normal/offensive/hatespeech to normal/offensive_profanity/hate.
    - `ruhsold.py` maps 5 fine-grained labels to normal/offensive_profanity/hate.
    - `sbic.py` produces normal/offensive_profanity/hate based on offensive+group-targeted status.
    - `dynahate.py` and `toxigen.py` map binary hate to normal/hate.
    - `implicit_hate.py` sets severity to hate for hateful text.
  - **Rationale**:
    - `hatexplain.py` extracts token offsets (spans) for rationales.
- **Which HASOC/IEEE DataPort converters exist?** None exist in the `converters/` directory.

## 5. Dependencies Audit
Imports found in Python files but missing in `requirements.txt`:
- `fastapi`
- `gradio`
- `numpy`
- `peft`
- `pydantic`
- `pytest`
- `requests`
- `uvicorn` (needed for FastAPI)
- `httpx` (needed for FastAPI tests/clients)
- `onnxruntime`
- `aiofiles`
- `python-multipart`
- `indic-transliteration` (will be used in Phase 1)

## 6. Train.csv Status
`data/train.csv` does not exist. 
Command to produce it:
```bash
python build_dataset.py
python prepare_splits.py
```

## 7. Risk List for Adding Speech Input
1. Model was trained entirely on text; speech introduces ASR artifacts (misspellings, wrong word boundaries, lack of punctuation).
2. Code-mixing in speech (e.g., Hinglish) may not map perfectly to the Romanized text the model was trained on, or ASR may transcribe code-mixed speech entirely into native scripts (Devanagari, Tamil) which the model has rarely seen.
3. Audio length limits and streaming latency for a REST/WebSocket API could cause timeouts or memory spikes.

## 8. Questions for User
1. Since we have no HASOC or IEEE DataPort converters, should we implement them now, or proceed with the existing sources?
2. Which ASR system (Whisper, Google Cloud, Azure) will be used for speech input, so we can mock its typical artifacts?
3. What is the intended deployment target and memory limit for the final system?
