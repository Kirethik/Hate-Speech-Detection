# Model Card: Civitas Model B (Generation)

## Model Details
- **Architecture**: `bigscience/mt0-small` fine-tuned with QLoRA/Adafactor.
- **Task**: Seq2Seq generation (Conditional text generation).
- **Languages**: en, hi, ta, te, ml, ur_roman.

## Intended Use
- **Rewrite**: Generates a safe, de-toxified version of a flagged utterance.
- **Respond**: Generates an empathetic counter-narrative addressing the speaker.

## Safety Mechanisms
- **Gating**: Model B outputs are re-scored by Model A. Any output with `hate_prob > 0.5` is discarded.
- **Fallback**: If all generated candidates fail the safety check or have extreme overlap with the input, a pre-written safe template is returned.

## Limitations & Risks
- **Hallucination**: The model may occasionally invent facts in counter-narratives.
- **Language Fidelity**: Generation quality in Telugu and Malayalam may be lower than English due to limited training data in those languages.

## Training Data
ParaDetox, TextDetox 2024, CONAN, Multi-CONAN, Qian et al. Gab/Reddit (with machine translations via IndicTrans2 for Indic coverage).
