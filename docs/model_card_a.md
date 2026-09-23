# Model Card: Civitas Model A (Detection)

## Model Details
- **Architecture**: `xlm-roberta-base` encoder + 4 task-specific classification heads (Hate, Target, Severity, Rationale).
- **Languages**: English (en), Hindi (hi), Tamil (ta), Telugu (te), Malayalam (ml), Roman Urdu (ur_roman).
- **Input**: Code-mixed or monolingual text.
- **Output**: JSON payload with probabilities for hate, severity class, target group, and rationale character spans.

## Intended Use
- Moderation pre-filtering in dual-node moderation architectures.
- Triggering secondary checks (lexicon lookup, NLI verification) for uncertain predictions.

## Limitations & Risks
- **Dialect Bias**: Performance on highly localized slang (e.g. specific to isolated districts) is not guaranteed.
- **Identity Term False Positives**: While the model is trained with balanced identity-mention datasets, it may still over-flag sentences asserting marginalized identities.
- **Transliteration Noise**: Performance degrades on heavily phonetically ambiguous transliterated text.

## Training Data
A custom blend of 12 datasets including HateXplain, MACD, DravidianCodeMix, and others. See `docs/DATA_REPORT.md` for the exact breakdown.

## Metrics (Validation)
(Populated after Phase 3 Colab training)
- Hate Macro-F1: TBD
- Severity Macro-F1: TBD
- Target Group F1: TBD
