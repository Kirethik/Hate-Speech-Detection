# Civitas AI — Model Artifact Integration Guide

This guide explains how to install trained model artifacts into Civitas AI and transition from **Demo Mode** to **Real Model Inference Mode**.

## Artifact Locations

The application loads models from designated folders in `model_artifacts/`:

```text
Civitas-AI/
└── model_artifacts/
    ├── hate_detector/
    │   ├── best_model.pt (or model.safetensors)
    │   ├── config.json
    │   ├── tokenizer.json
    │   └── model_metadata.json
    │
    └── alternative_generator/
        ├── adapter_config.json (LoRA) or model.safetensors (Full)
        ├── adapter_model.safetensors
        ├── tokenizer.json
        └── model_metadata.json
```

---

## Installation Steps

### Option A: Using the Automated Installer
Place `hate_detector_artifact.zip` and/or `alternative_generator_artifact.zip` in your project folder (or `Downloads` folder) and run:

```bash
# Extract and verify models
python scripts/install_models.py --check
```

### Option B: Manual Installation
1. Extract `hate_detector_artifact.zip` into `model_artifacts/hate_detector/`.
2. Extract `alternative_generator_artifact.zip` into `model_artifacts/alternative_generator/`.

---

## Verifying Model Status

Check the status via CLI or API:

```bash
# Check loaded models and device
python -c "from src.models.model_loader import get_system_model_status; import json; print(json.dumps({k: v.to_dict() for k, v in get_system_model_status().items()}, indent=2))"
```

Or via API:
```bash
curl http://127.0.0.1:8000/api/model-status
```

Once installed, the status badge in the web UI will transition from `Demo Mode` to `Real Model Active`.
