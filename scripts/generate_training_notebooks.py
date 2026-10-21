"""
Civitas AI — Generates reproducible Google Colab training notebooks:
1. training/01_train_hate_detector.ipynb
2. training/02_train_alternative_generator.ipynb
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }


def md_cell(source):
    lines = source.strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines]
    }


def code_cell(source):
    lines = source.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines]
    }


def build_detector_notebook():
    cells = [
        md_cell("""# Civitas AI — Model A: Hate Speech Detector Training Pipeline
## Fine-tuning XLM-RoBERTa for Multilingual Hate Speech Classification

This notebook is fully self-contained and executable in Google Colab. It trains, evaluates, exports, and verifies the **Model A Hate Speech Detector**.

### Workflow:
1. **Environment Setup & Hardware Check**: Inspect PyTorch, CUDA, and GPU.
2. **Hyperparameter Configuration**: Central configuration cell.
3. **Dataset Acquisition**: Upload CSVs directly or mount Google Drive.
4. **Tokenization & Preprocessing**: XLM-RoBERTa tokenizer with attention masks.
5. **Model Initialization**: `AutoModelForSequenceClassification` with explicit label mappings.
6. **Training**: HuggingFace `Trainer` with validation macro-F1 checkpointing and Google Drive recovery.
7. **Comprehensive Evaluation**: Confusion matrix, classification report, per-class F1, and language-wise analysis.
8. **Export**: Full model, tokenizer, config, and metadata saved and zipped as `hate_detector_artifact.zip`.
9. **Reload Verification**: Reload artifact from disk and verify inference consistency."""),

        md_cell("### 1. Environment & Hardware Diagnostics"),
        code_cell("""# Install required packages
!pip install -q transformers datasets accelerate scikit-learn pandas numpy matplotlib seaborn safetensors

import sys, os, json, zipfile, shutil, random
import torch
import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 50)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU. For fast training, enable GPU under Runtime -> Change runtime type.")
print("=" * 50)"""),

        md_cell("### 2. Configuration & Hyperparameters"),
        code_cell("""# Central Training Configuration
BASE_MODEL = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 256
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2

WARMUP_RATIO = 0.1
RANDOM_SEED = 42

OUTPUT_DIR = "/content/runs/hate_detector"
ARTIFACT_DIR = "/content/model_artifacts/hate_detector"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Civitas_AI/hate_detector"

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("Configuration loaded. Base model:", BASE_MODEL)"""),

        md_cell("### 3. Google Drive Mount & Dataset Loading\nYou can either mount Google Drive (recommended for checkpoint backup) or upload files directly."),
        code_cell("""# Mount Google Drive for persistent checkpoints
USE_DRIVE = True #@param {type:"boolean"}

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
        print("Google Drive mounted. Checkpoints will be backed up to:", DRIVE_BACKUP_DIR)
    except Exception as e:
        print("Could not mount drive or running locally:", e)

# Locate or upload dataset splits
DATA_DIR = "/content/data"
os.makedirs(DATA_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train.csv")
val_path = os.path.join(DATA_DIR, "validation.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

# If files don't exist, check Drive or prompt upload
if not os.path.exists(train_path):
    drive_data = "/content/drive/MyDrive/Civitas_AI/data"
    if os.path.exists(os.path.join(drive_data, "train.csv")):
        shutil.copy(os.path.join(drive_data, "train.csv"), train_path)
        shutil.copy(os.path.join(drive_data, "validation.csv"), val_path)
        shutil.copy(os.path.join(drive_data, "test.csv"), test_path)
        print("Copied dataset files from Google Drive!")
    else:
        print("Upload your train.csv, validation.csv, and test.csv:")
        from google.colab import files
        uploaded = files.upload()
        for fn in uploaded.keys():
            shutil.move(fn, os.path.join(DATA_DIR, fn))

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

print(f"Loaded splits: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")
print("Train Columns:", df_train.columns.tolist())
print("Label Distribution (Train):\\n", df_train['label'].value_counts(normalize=True))"""),

        md_cell("### 4. Tokenization & Dataset Formatting"),
        code_cell("""from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# Format splits as HuggingFace datasets
train_ds = Dataset.from_pandas(df_train[["text", "label"]])
val_ds = Dataset.from_pandas(df_val[["text", "label"]])
test_ds = Dataset.from_pandas(df_test[["text", "label"]])

train_encoded = train_ds.map(preprocess_function, batched=True)
val_encoded = val_ds.map(preprocess_function, batched=True)
test_encoded = test_ds.map(preprocess_function, batched=True)

print("Tokenization complete!")"""),

        md_cell("### 5. Model Initialization & Evaluation Metrics"),
        code_cell("""from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

id2label = {0: "non_hateful", 1: "hateful"}
label2id = {"non_hateful": 0, "hateful": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": p,
        "macro_recall": r,
        "weighted_f1": w_f1
    }

print("Model initialized for binary sequence classification.")"""),

        md_cell("### 6. Training with HuggingFace Trainer"),
        code_cell("""training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    seed=RANDOM_SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting training...")
train_result = trainer.train()
print("Training completed!")"""),

        md_cell("### 7. Evaluation on Held-Out Test Set\nEvaluate the best checkpoint once on the held-out test split."),
        code_cell("""print("Evaluating best checkpoint on held-out test set...")
test_results = trainer.predict(test_encoded)
preds = np.argmax(test_results.predictions, axis=1)
labels = test_results.label_ids

report_dict = classification_report(labels, preds, target_names=["non_hateful", "hateful"], output_dict=True)
print("Classification Report:\\n", classification_report(labels, preds, target_names=["non_hateful", "hateful"]))

cm = confusion_matrix(labels, preds)
print("Confusion Matrix:\\n", cm)

# Save test metrics
os.makedirs(ARTIFACT_DIR, exist_ok=True)
with open(os.path.join(ARTIFACT_DIR, "test_report.json"), "w") as f:
    json.dump(report_dict, f, indent=2)"""),

        md_cell("### 8. Export Model Artifacts\nSaves complete model, tokenizer, and metadata, and creates `hate_detector_artifact.zip`."),
        code_cell("""# Save using HuggingFace standard save_pretrained
trainer.save_model(ARTIFACT_DIR)
tokenizer.save_pretrained(ARTIFACT_DIR)

metadata = {
    "model_name": "Civitas AI Model A — Hate Speech Detector",
    "base_model": BASE_MODEL,
    "architecture": "XLM-RoBERTa",
    "timestamp": datetime.utcnow().isoformat(),
    "max_length": MAX_LENGTH,
    "id2label": id2label,
    "label2id": label2id,
    "test_metrics": report_dict
}

with open(os.path.join(ARTIFACT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Create ZIP archive
zip_path = "/content/hate_detector_artifact.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(ARTIFACT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ARTIFACT_DIR)
            z.write(full_path, arcname=rel_path)

print(f"Artifact created: {zip_path} ({os.path.getsize(zip_path) / 1e6:.2f} MB)")

# Copy to Google Drive if available
if USE_DRIVE and os.path.exists(DRIVE_BACKUP_DIR):
    shutil.copy(zip_path, os.path.join(DRIVE_BACKUP_DIR, "hate_detector_artifact.zip"))
    print("Backed up artifact to Google Drive!")"""),

        md_cell("### 9. Reload Verification (Smoke Test)\nLoad the exported model and tokenizer from disk in an isolated step to verify portability."),
        code_cell("""from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print("Verifying exported artifact from disk...")
loaded_tok = AutoTokenizer.from_pretrained(ARTIFACT_DIR)
loaded_model = AutoModelForSequenceClassification.from_pretrained(ARTIFACT_DIR)
loaded_model.eval()

test_phrases = [
    "Thank you for your valuable feedback, I appreciate it.",
    "You people are a disgusting plague on our society.",
    "I completely disagree with your policy argument."
]

for phrase in test_phrases:
    inputs = loaded_tok(phrase, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
        pred_id = int(torch.argmax(outputs.logits, dim=-1))
        pred_label = loaded_model.config.id2label[pred_id]
    print(f"Input: {phrase!r}")
    print(f"  -> Prediction: {pred_label} (non_hateful: {probs[0]:.3f}, hateful: {probs[1]:.3f})\\n")

print("Reload verification SUCCESS! Download hate_detector_artifact.zip and install into Civitas AI.")""")
    ]
    return make_notebook(cells)


def build_generator_notebook():
    cells = [
        md_cell("""# Civitas AI — Model B: Constructive Alternative Generator Training Pipeline
## Fine-tuning mT5 / mT0 for Conditional Multilingual Text Rewriting with Tone Control

This notebook is fully self-contained and executable in Google Colab. It trains, evaluates, exports, and verifies the **Model B Constructive Alternative Generator**.

### Workflow:
1. **Environment Setup & Hardware Check**: Inspect PyTorch, CUDA, and GPU.
2. **Hyperparameter Configuration**: Configure base model, seq2seq lengths, and tone tags.
3. **Dataset Loading**: Load paired constructive alternatives (`input_text`, `target_text`, `tone`, `language`).
4. **Input Conditioning Formatting**: Format inputs as `rewrite_tone={tone}: {input_text}`.
5. **Model Initialization**: `AutoModelForSeq2SeqLM` (e.g. `google/mt5-small` or `bigscience/mt0-base`).
6. **Training**: `Seq2SeqTrainer` with target padding token masking (`-100`).
7. **Evaluation**: SacreBLEU, ROUGE-L, and qualitative comparisons.
8. **Export**: Full model, tokenizer, generation config, and metadata zipped as `alternative_generator_artifact.zip`.
9. **Reload Verification**: Reload from disk and verify generation across tones: `calm`, `factual`, `firm`, `empathetic`, `professional`."""),

        md_cell("### 1. Environment & Hardware Diagnostics"),
        code_cell("""!pip install -q transformers datasets accelerate sentencepiece evaluate sacrebleu rouge-score pandas numpy safetensors

import sys, os, json, zipfile, shutil, random
import torch
import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 50)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU. Recommend T4 GPU or higher for seq2seq training.")
print("=" * 50)"""),

        md_cell("### 2. Configuration & Hyperparameters"),
        code_cell("""# Base Model Selection: google/mt5-small or bigscience/mt0-base
BASE_MODEL = "google/mt5-small"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

NUM_EPOCHS = 3
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

RANDOM_SEED = 42

OUTPUT_DIR = "/content/runs/alternative_generator"
ARTIFACT_DIR = "/content/model_artifacts/alternative_generator"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Civitas_AI/alternative_generator"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("Configuration loaded. Base generator model:", BASE_MODEL)"""),

        md_cell("### 3. Google Drive Mount & Dataset Loading"),
        code_cell("""USE_DRIVE = True #@param {type:"boolean"}

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
        print("Google Drive mounted. Backup directory:", DRIVE_BACKUP_DIR)
    except Exception as e:
        print("Drive mount note:", e)

DATA_DIR = "/content/data_gen"
os.makedirs(DATA_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train.csv")
val_path = os.path.join(DATA_DIR, "validation.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(train_path):
    drive_data = "/content/drive/MyDrive/Civitas_AI/generator_data"
    if os.path.exists(os.path.join(drive_data, "train.csv")):
        shutil.copy(os.path.join(drive_data, "train.csv"), train_path)
        shutil.copy(os.path.join(drive_data, "validation.csv"), val_path)
        shutil.copy(os.path.join(drive_data, "test.csv"), test_path)
        print("Copied generator data from Google Drive!")
    else:
        print("Upload your generator train.csv, validation.csv, and test.csv:")
        from google.colab import files
        uploaded = files.upload()
        for fn in uploaded.keys():
            shutil.move(fn, os.path.join(DATA_DIR, fn))

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

print(f"Loaded pairs: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")
print(df_train.head(2)[["input_text", "target_text"]])"""),

        md_cell("### 4. Conditioning Formatting & Tokenization"),
        code_cell("""from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def format_prompt(row):
    tone = row.get("tone", "calm")
    input_text = str(row["input_text"]).strip()
    return f"rewrite_tone={tone}: {input_text}"

def preprocess_seq2seq(batch):
    inputs = [format_prompt({"tone": t, "input_text": i}) for t, i in zip(batch.get("tone", ["calm"]*len(batch["input_text"])), batch["input_text"])]
    targets = [str(t).strip() for t in batch["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    
    # Mask padding tokens for loss computation
    labels_with_ignore = []
    for label_seq in labels["input_ids"]:
        labels_with_ignore.append([-100 if token == tokenizer.pad_token_id else token for token in label_seq])
    
    model_inputs["labels"] = labels_with_ignore
    return model_inputs

train_ds = Dataset.from_pandas(df_train).map(preprocess_seq2seq, batched=True)
val_ds = Dataset.from_pandas(df_val).map(preprocess_seq2seq, batched=True)
test_ds = Dataset.from_pandas(df_test).map(preprocess_seq2seq, batched=True)

print("Seq2Seq preprocessing complete!")"""),

        md_cell("### 5. Model Initialization & Seq2Seq Metrics"),
        code_cell("""from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate

model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bleu_score = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "bleu": round(bleu_score["score"], 2),
        "rouge1": round(rouge_score["rouge1"] * 100, 2),
        "rougeL": round(rouge_score["rougeL"] * 100, 2)
    }

print("Generator initialized. Metrics configured.")"""),

        md_cell("### 6. Training with Seq2SeqTrainer"),
        code_cell("""training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    seed=RANDOM_SEED
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

print("Starting generator training...")
trainer.train()
print("Generator training completed!")"""),

        md_cell("### 7. Evaluation on Held-Out Test Set"),
        code_cell("""print("Evaluating on test set...")
test_metrics = trainer.evaluate(test_ds)
print("Test Metrics:", test_metrics)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
with open(os.path.join(ARTIFACT_DIR, "evaluation_report.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)"""),

        md_cell("### 8. Export Model Artifacts\nSaves complete model, tokenizer, and generation config, and creates `alternative_generator_artifact.zip`."),
        code_cell("""trainer.save_model(ARTIFACT_DIR)
tokenizer.save_pretrained(ARTIFACT_DIR)

metadata = {
    "model_name": "Civitas AI Model B — Constructive Alternative Generator",
    "base_model": BASE_MODEL,
    "architecture": "mT5 / Seq2Seq",
    "timestamp": datetime.utcnow().isoformat(),
    "max_input_length": MAX_INPUT_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "supported_tones": ["calm", "factual", "firm", "empathetic", "professional"],
    "prompt_prefix": "rewrite_tone={tone}: {input_text}",
    "test_metrics": test_metrics
}

with open(os.path.join(ARTIFACT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

zip_path = "/content/alternative_generator_artifact.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(ARTIFACT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ARTIFACT_DIR)
            z.write(full_path, arcname=rel_path)

print(f"Artifact created: {zip_path} ({os.path.getsize(zip_path) / 1e6:.2f} MB)")

if USE_DRIVE and os.path.exists(DRIVE_BACKUP_DIR):
    shutil.copy(zip_path, os.path.join(DRIVE_BACKUP_DIR, "alternative_generator_artifact.zip"))
    print("Backed up artifact to Google Drive!")"""),

        md_cell("### 9. Reload Verification & Tone Testing\nReload generator from disk and test generation across tones."),
        code_cell("""from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

print("Verifying exported generator from disk...")
loaded_tok = AutoTokenizer.from_pretrained(ARTIFACT_DIR)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(ARTIFACT_DIR)
loaded_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

test_sentence = "You people are complete idiots and ruining everything."
tones = ["calm", "factual", "firm", "empathetic", "professional"]

print(f"Original Text: {test_sentence!r}\\n")
for tone in tones:
    prompt = f"rewrite_tone={tone}: {test_sentence}"
    inputs = loaded_tok(prompt, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
    with torch.no_grad():
        out = loaded_model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=3,
            early_stopping=True
        )
    output_text = loaded_tok.decode(out[0], skip_special_tokens=True)
    print(f"[{tone.upper()} REWRITE]: {output_text}")

print("\\nReload verification SUCCESS! Download alternative_generator_artifact.zip and install into Civitas AI.")""")
    ]
    return make_notebook(cells)


def main():
    det_nb = build_detector_notebook()
    det_path = TRAINING_DIR / "01_train_hate_detector.ipynb"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(det_nb, f, indent=2)
    print(f"Generated: {det_path} ({len(det_nb['cells'])} cells)")

    gen_nb = build_generator_notebook()
    gen_path = TRAINING_DIR / "02_train_alternative_generator.ipynb"
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_nb, f, indent=2)
    print(f"Generated: {gen_path} ({len(gen_nb['cells'])} cells)")


if __name__ == "__main__":
    main()
