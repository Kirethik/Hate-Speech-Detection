# Civitas AI — Test Suite

This directory contains automated unit, integration, and API tests.

## Running Tests

Run all unit tests:
```bash
python -m pytest tests/ -q
```

Run specific test modules:
```bash
# Model interfaces and contracts
python -m pytest tests/test_model_interfaces.py

# FastAPI server endpoints
python -m pytest tests/test_api.py

# Dataset schemas and preprocessing
python -m pytest tests/test_preprocessing.py

# Text normalization and code-mixed scripts
python -m pytest tests/test_normalize.py

# Label mappings
python -m pytest tests/test_label_maps.py
```
