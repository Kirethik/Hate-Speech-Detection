# Model B — Checkpoints (LoRA Adapters)

This directory stores the QLoRA adapter weights for Model B (`google/mt5-small`
fine-tuned for counter-narrative generation).

## Contents after training

- `adapter_model.safetensors` — LoRA adapter weights (~5–20 MB)
- `adapter_config.json` — PEFT configuration
- `tokenizer_config.json` + `spiece.model` — tokenizer files
- `training_metrics.json` — per-epoch loss curve and eval metrics
- `test_metrics_gen.json` — final BERTScore + Distinct-2 on held-out test set

## Reproducing

```bash
python model_b_generation/build_dataset_gen.py
python model_b_generation/prepare_splits_gen.py
python model_b_generation/train_gen.py
python model_b_generation/evaluate_gen.py
```

The base model (`google/mt5-small`) is downloaded from HuggingFace on first use
and cached locally. Only the LoRA adapter weights are saved here — the full
model is reconstructed at inference time by loading the adapter on top of the
frozen base weights.
