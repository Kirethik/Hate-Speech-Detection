# Civitas AI — Google Colab Training Guide

This guide details how to execute independent training runs for **Model A (Hate Speech Detector)** and **Model B (Constructive Alternative Generator)** on Google Colab with GPU acceleration.

## Overview

| Model | Notebook File | Base Architecture | Dataset Input | Output Artifact |
|---|---|---|---|---|
| **Model A** | `training/01_train_hate_detector.ipynb` | `FacebookAI/xlm-roberta-base` | `datasets/processed/detector/{train,validation,test}.csv` | `hate_detector_artifact.zip` |
| **Model B** | `training/02_train_alternative_generator.ipynb` | `google/mt5-small` / `mt0-base` | `datasets/processed/generator/{train,validation,test}.csv` | `alternative_generator_artifact.zip` |

---

## Instructions for Training Model A (Detector)

1. **Open Google Colab**: Navigate to [colab.research.google.com](https://colab.research.google.com).
2. **Upload Notebook**: Select **File → Upload notebook** and choose `training/01_train_hate_detector.ipynb`.
3. **Select GPU**: Navigate to **Runtime → Change runtime type → T4 GPU** (or A100/L4 if Colab Pro).
4. **Mount Drive & Upload Data**:
   - Run the cells sequentially.
   - When prompted, allow Google Drive access (checkpoints will be saved automatically to `My Drive/Civitas_AI/hate_detector`).
   - Upload `datasets/processed/detector/train.csv`, `validation.csv`, and `test.csv` (or place them in `My Drive/Civitas_AI/data/`).
5. **Run Training**: Click **Runtime → Run all**.
6. **Download Artifact**:
   - When training completes and the reload verification passes, download `hate_detector_artifact.zip` (located in `/content/` or in `My Drive/Civitas_AI/hate_detector/`).
   - Copy this ZIP into your local `model_artifacts/hate_detector/` directory or run `python scripts/install_models.py`.

---

## Instructions for Training Model B (Generator)

1. **Open Google Colab**: Navigate to [colab.research.google.com](https://colab.research.google.com).
2. **Upload Notebook**: Select **File → Upload notebook** and choose `training/02_train_alternative_generator.ipynb`.
3. **Select GPU**: Navigate to **Runtime → Change runtime type → T4 GPU**.
4. **Upload Data**:
   - Provide `datasets/processed/generator/train.csv`, `validation.csv`, and `test.csv`.
5. **Run Training**: Click **Runtime → Run all**.
6. **Download Artifact**:
   - Download `alternative_generator_artifact.zip`.
   - Copy this ZIP into your local `model_artifacts/alternative_generator/` directory or run `python scripts/install_models.py`.

---

## Reconnection & Recovery

Both notebooks save checkpoints at each epoch to Google Drive. If your Colab session times out or disconnects:
1. Reconnect and mount Google Drive.
2. The notebook will automatically detect existing checkpoints in Drive and resume without restarting from step zero.
