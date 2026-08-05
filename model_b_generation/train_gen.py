"""
Training loop for Model B — QLoRA fine-tuning of mT5-small for
counter-narrative generation.

Training spec (Section 5 of the blueprint):
  - QLoRA: 4-bit NF4 base, LoRA r=8 adapters trained
  - Optimizer: AdamW, lr=1e-4, cosine schedule with 10% warmup
  - Batch size: 8, gradient accumulation steps: 2 (effective 16)
  - Max input/output length: 128/64 tokens
  - Checkpoint: LoRA adapters only (not full base model) → checkpoints_gen/

Usage:
    python -m model_b_generation.train_gen [--epochs 10] [--lr 1e-4]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Allow running as `python train_gen.py` from the package directory
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from dataset_gen import CivitasGenDataset
from model_gen import load_model_for_training

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune mt5-small with QLoRA")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train_gen.csv and val_gen.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints_gen", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    return parser.parse_args()

def train():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer = load_model_for_training(lora_r=args.lora_r)
    if not torch.cuda.is_available():
        model.to(device)
        
    train_csv = os.path.join(args.data_dir, "train_gen.csv")
    val_csv = os.path.join(args.data_dir, "val_gen.csv")
    
    train_dataset = CivitasGenDataset(train_csv, tokenizer)
    val_dataset = CivitasGenDataset(val_csv, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    metrics = {"epochs": []}
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Try using modern torch.amp
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, batch in enumerate(train_pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
            total_train_loss += loss.item() * args.gradient_accumulation_steps
            train_pbar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                if scaler:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        metrics["epochs"].append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            print(f"New best model! Saving to {args.output_dir}...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val_loss
    
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Training complete. Best epoch: {best_epoch} with val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
