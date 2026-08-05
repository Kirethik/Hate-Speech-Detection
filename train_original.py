"""
Training entry point for Model A (Civitas AI multilingual detector).

Usage:
    python train.py \
        --train_csv data/train.csv \
        --val_csv data/val.csv \
        --output_dir checkpoints/model_a \
        --epochs 5 --batch_size 16 --lr 2e-5

Expected CSV schema: see dataset.py docstring.
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import XLMRobertaTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from model import CivitasDetector, FocalLoss, compute_multitask_loss
from dataset import CivitasDetectorDataset, TARGET_CLASSES, SEVERITY_CLASSES


def compute_class_weights(labels, num_classes, device):
    labels = np.array(labels)
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full_weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present, weights):
        full_weights[cls] = w
    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def evaluate(model, dataloader, device):
    model.eval()
    all_hate_preds, all_hate_labels = [], []
    per_lang = defaultdict(lambda: {"preds": [], "labels": []})

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs["hate_logits"].argmax(dim=-1).cpu().tolist()
            labels = batch["hate_label"].tolist()
            langs = batch["language"]

            all_hate_preds.extend(preds)
            all_hate_labels.extend(labels)
            for p, l, lang in zip(preds, labels, langs):
                per_lang[lang]["preds"].append(p)
                per_lang[lang]["labels"].append(l)

    macro_f1 = f1_score(all_hate_labels, all_hate_preds, average="macro")

    per_lang_f1 = {}
    for lang, d in per_lang.items():
        if len(set(d["labels"])) < 2:
            per_lang_f1[lang] = None  # not enough class variety to score meaningfully
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            d["labels"], d["preds"], average="macro", zero_division=0
        )
        per_lang_f1[lang] = float(f1)

    return {"macro_f1": macro_f1, "per_language_f1": per_lang_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--output_dir", default="checkpoints/model_a")
    parser.add_argument("--encoder_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.encoder_name)

    train_dataset = CivitasDetectorDataset(args.train_csv, tokenizer, args.max_length)
    val_dataset = CivitasDetectorDataset(args.val_csv, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CivitasDetector(
        encoder_name=args.encoder_name,
        num_target_classes=len(TARGET_CLASSES),
        num_severity_classes=len(SEVERITY_CLASSES),
    ).to(device)

    # Class-weighted focal loss for hate + severity (imbalance per slide 11)
    hate_labels = train_dataset.df["hate_label"].tolist()
    hate_weights = compute_class_weights(hate_labels, 2, device)
    hate_loss_fn = FocalLoss(gamma=2.0, weight=hate_weights)

    severity_rows = train_dataset.df[train_dataset.df["severity_label"] >= 0]
    if len(severity_rows) > 0:
        severity_weights = compute_class_weights(
            severity_rows["severity_label"].tolist(), len(SEVERITY_CLASSES), device
        )
    else:
        severity_weights = None
    severity_loss_fn = FocalLoss(gamma=2.0, weight=severity_weights)

    target_loss_fn = nn.CrossEntropyLoss()
    rationale_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch_gpu = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            outputs = model(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
            )

            loss, loss_dict = compute_multitask_loss(
                outputs, batch_gpu,
                hate_loss_fn, target_loss_fn, severity_loss_fn, rationale_loss_fn,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f} {loss_dict}")

        avg_loss = running_loss / len(train_loader)
        metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] avg_train_loss={avg_loss:.4f} "
              f"val_macro_f1={metrics['macro_f1']:.4f} "
              f"per_language={metrics['per_language_f1']}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> new best model saved to {save_path}")

    print(f"Training complete. Best val macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
