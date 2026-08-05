"""
Training entry point for Model A (Civitas AI multilingual detector).

Usage:
    python train.py                                  # sensible defaults
    python train.py --batch_size 32 --epochs 4       # override

What this does differently from a plain fine-tuning loop, and why:

  * SPLIT HYGIENE. Refuses to run unless data/{train,val,test}.csv exist and are
    disjoint (run prepare_splits.py first). `val` selects the checkpoint, `test`
    is touched exactly once at the end. Selecting and reporting on the same set
    is how you get a good number instead of a good model.
  * DYNAMIC PADDING. Texts here average 29 tokens; padding every row to 128
    burns ~78% of the GPU on padding. Sequences are pre-tokenised once and
    padded per-batch to the longest member instead.
  * BF16 AUTOCAST. The target GPU is Ampere (sm_86) which has bf16 tensor cores.
    bf16 needs no GradScaler and has fp32's exponent range, so it will not
    silently produce inf losses the way fp16 can.
  * MID-EPOCH EVAL + EARLY STOP. An epoch here is ~6k optimiser steps; evaluating
    only at epoch boundaries gives you a handful of chances to catch the best
    checkpoint and none to catch divergence.
  * HONEST METRICS. Macro-F1 is reported per-language, per-source and per-class,
    not just pooled. Pooled macro-F1 is carried by MACD (58% of eval) and by the
    fact that language correlates with the label, so a model can score well by
    learning "Malayalam -> probably fine" and still be a bad detector.
  * TUNED THRESHOLD. The decision threshold is fitted on val and then applied
    unchanged to test, rather than assuming 0.5 is optimal.
"""

import argparse
import contextlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizerFast, get_linear_schedule_with_warmup

from dataset import SEVERITY_CLASSES, TARGET_CLASSES, _parse_spans, normalize_code_mixed
from identity_augment import expand_with_identity_augmentation
from model import CivitasDetector, FocalLoss, compute_multitask_loss

log = logging.getLogger("train")


def setup_logging(log_path: Path):
    """Log to stdout AND to train.log, so `tail -f train.log` actually works."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def amp_context(amp_dtype):
    """autocast when AMP is on, a no-op otherwise (keeps CPU runs working)."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class PreTokenizedDataset(Dataset):
    """
    Tokenises the whole frame once up front and keeps variable-length int32
    arrays, so the training loop never pays for tokenisation and batches can be
    padded to their own longest member instead of to max_length.

    Rationale spans are char offsets into the ORIGINAL text, but the model sees
    normalise()d text. Normalisation here is whitespace-only, so for the one
    source that has rationales (HateXplain, whose text is already single-spaced)
    it is a no-op. Rather than assume that, any row whose text actually changes
    under normalisation has its rationale supervision dropped — misaligned token
    labels are worse than absent ones.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length

        raw_texts = self.df["text"].astype(str).tolist()
        texts = [normalize_code_mixed(t) for t in raw_texts]

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        self.input_ids = [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]

        self.hate = self.df["hate_label"].astype(int).to_numpy()
        self.target = self.df["target_label"].fillna(-1).astype(int).to_numpy()
        self.severity = self.df["severity_label"].fillna(-1).astype(int).to_numpy()
        self.language = self.df["language"].astype(str).tolist()
        self.source = self.df.get("source", pd.Series(["unknown"] * len(self.df))).astype(str).tolist()

        self.rationale = []
        self.has_rationale = np.zeros(len(self.df), dtype=bool)
        spans_col = self.df["rationale_spans"].astype(str).tolist() if "rationale_spans" in self.df else ["[]"] * len(self.df)

        n_dropped_misaligned = 0
        for i, raw_spans in enumerate(spans_col):
            n_tok = len(self.input_ids[i])
            spans = _parse_spans(raw_spans)
            annotated = len(spans) > 0
            if annotated and texts[i] != raw_texts[i].strip():
                # normalisation moved characters -> char spans no longer align
                n_dropped_misaligned += 1
                annotated = False
            if not annotated:
                self.rationale.append(np.full(n_tok, -100, dtype=np.int64))
                continue

            labels = np.full(n_tok, -100, dtype=np.int64)
            offsets = enc["offset_mapping"][i]
            for t, (start, end) in enumerate(offsets):
                if start == end:      # special token
                    continue
                labels[t] = 0
                for s_start, s_end in spans:
                    if start < s_end and end > s_start:
                        labels[t] = 1
                        break
            self.rationale.append(labels)
            self.has_rationale[i] = True

        if n_dropped_misaligned:
            log.info(f"  dropped rationale supervision on {n_dropped_misaligned} rows "
                     f"(normalisation shifted char offsets)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "rationale_labels": self.rationale[i],
            "hate_label": int(self.hate[i]),
            "target_label": int(max(self.target[i], 0)),
            "target_mask": int(self.target[i] >= 0),
            "severity_label": int(max(self.severity[i], 0)),
            "severity_mask": int(self.severity[i] >= 0),
            "rationale_mask": int(self.has_rationale[i]),
            "language": self.language[i],
            "source": self.source[i],
        }


def make_collate(pad_token_id: int):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        n = len(batch)
        input_ids = np.full((n, maxlen), pad_token_id, dtype=np.int64)
        attention = np.zeros((n, maxlen), dtype=np.int64)
        rationale = np.full((n, maxlen), -100, dtype=np.int64)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = b["input_ids"]
            attention[i, :L] = 1
            rationale[i, :L] = b["rationale_labels"]
        out = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "rationale_labels": torch.from_numpy(rationale),
        }
        for key in ("hate_label", "target_label", "target_mask",
                    "severity_label", "severity_mask", "rationale_mask"):
            out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
        out["language"] = [b["language"] for b in batch]
        out["source"] = [b["source"] for b in batch]
        return out
    return collate


def compute_class_weights(labels, num_classes, device):
    labels = np.asarray(labels)
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present, weights):
        full[cls] = w
    return torch.tensor(full, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    """One forward pass over a loader, returning raw predictions for all heads."""
    model.eval()
    out = defaultdict(list)
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with amp_context(amp_dtype):
            o = model(input_ids=input_ids, attention_mask=attention_mask)

        out["hate_prob"].append(torch.softmax(o["hate_logits"].float(), -1)[:, 1].cpu().numpy())
        out["hate_true"].append(batch["hate_label"].numpy())
        out["target_pred"].append(o["target_logits"].float().argmax(-1).cpu().numpy())
        out["target_true"].append(batch["target_label"].numpy())
        out["target_mask"].append(batch["target_mask"].numpy())
        out["sev_pred"].append(o["severity_logits"].float().argmax(-1).cpu().numpy())
        out["sev_true"].append(batch["severity_label"].numpy())
        out["sev_mask"].append(batch["severity_mask"].numpy())

        r_pred = o["rationale_logits"].float().argmax(-1).cpu().numpy()
        r_true = batch["rationale_labels"].numpy()
        keep = r_true != -100
        out["rat_pred"].append(r_pred[keep])
        out["rat_true"].append(r_true[keep])

        out["language"].extend(batch["language"])
        out["source"].extend(batch["source"])

    merged = {}
    for k, v in out.items():
        merged[k] = np.array(v) if k in ("language", "source") else np.concatenate(v)
    return merged


def tune_threshold(probs, labels):
    """Pick the hate-head decision threshold that maximises macro-F1 on val."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def score(pred: dict, threshold: float) -> dict:
    """Full metric breakdown. Slice-level numbers are the point, not the pooled one."""
    hate_pred = (pred["hate_prob"] >= threshold).astype(int)
    hate_true = pred["hate_true"]

    metrics = {
        "threshold": threshold,
        "hate_macro_f1": float(f1_score(hate_true, hate_pred, average="macro")),
        "hate_accuracy": float((hate_pred == hate_true).mean()),
    }
    try:
        metrics["hate_roc_auc"] = float(roc_auc_score(hate_true, pred["hate_prob"]))
    except ValueError:
        metrics["hate_roc_auc"] = None

    p, r, f, sup = precision_recall_fscore_support(
        hate_true, hate_pred, labels=[0, 1], zero_division=0
    )
    metrics["hate_per_class"] = {
        name: {"precision": float(p[i]), "recall": float(r[i]),
               "f1": float(f[i]), "support": int(sup[i])}
        for i, name in enumerate(["not_abusive", "abusive"])
    }

    for slice_name in ("language", "source"):
        breakdown = {}
        for raw_key in sorted(set(pred[slice_name])):
            key = str(raw_key)             # numpy str_ serialises badly in JSON/logs
            m = pred[slice_name] == raw_key
            if m.sum() == 0 or len(set(hate_true[m])) < 2:
                breakdown[key] = None      # single-class slice: F1 not meaningful
                continue
            breakdown[key] = {
                "macro_f1": float(f1_score(hate_true[m], hate_pred[m], average="macro")),
                "n": int(m.sum()),
            }
        metrics[f"hate_by_{slice_name}"] = breakdown

    # Auxiliary heads, scored only where they were actually supervised.
    sev_m = pred["sev_mask"].astype(bool)
    if sev_m.sum():
        metrics["severity_macro_f1"] = float(
            f1_score(pred["sev_true"][sev_m], pred["sev_pred"][sev_m], average="macro")
        )
        metrics["severity_n"] = int(sev_m.sum())

    tgt_m = pred["target_mask"].astype(bool)
    if tgt_m.sum():
        t_true, t_pred = pred["target_true"][tgt_m], pred["target_pred"][tgt_m]
        metrics["target_macro_f1"] = float(f1_score(t_true, t_pred, average="macro", zero_division=0))
        _, _, tf, tsup = precision_recall_fscore_support(
            t_true, t_pred, labels=list(range(len(TARGET_CLASSES))), zero_division=0
        )
        # Per-class matters here: 'disability' has ~26 training examples, so a
        # respectable macro average would be hiding a head that cannot predict it.
        metrics["target_per_class"] = {
            TARGET_CLASSES[i]: {"f1": float(tf[i]), "support": int(tsup[i])}
            for i in range(len(TARGET_CLASSES))
        }
        metrics["target_n"] = int(tgt_m.sum())

    if len(pred["rat_true"]):
        metrics["rationale_token_f1"] = float(
            f1_score(pred["rat_true"], pred["rat_pred"], average="macro", zero_division=0)
        )
        metrics["rationale_n_tokens"] = int(len(pred["rat_true"]))
    return metrics


def log_metrics(tag: str, m: dict):
    log.info(f"  [{tag}] hate macro-F1={m['hate_macro_f1']:.4f} "
             f"acc={m['hate_accuracy']:.4f} auc={m.get('hate_roc_auc') or float('nan'):.4f} "
             f"@thr={m['threshold']:.2f}")
    langs = {k: (v["macro_f1"] if v else None) for k, v in m["hate_by_language"].items()}
    srcs = {k: (v["macro_f1"] if v else None) for k, v in m["hate_by_source"].items()}
    log.info(f"  [{tag}] by language: { {k: (round(v,3) if v else None) for k,v in langs.items()} }")
    log.info(f"  [{tag}] by source:   { {k: (round(v,3) if v else None) for k,v in srcs.items()} }")
    aux = []
    for key, label in [("severity_macro_f1", "severity"), ("target_macro_f1", "target"),
                       ("rationale_token_f1", "rationale")]:
        if key in m:
            aux.append(f"{label}={m[key]:.4f}")
    if aux:
        log.info(f"  [{tag}] aux heads: " + "  ".join(aux))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--output_dir", default="checkpoints/model_a")
    parser.add_argument("--encoder_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--evals_per_epoch", type=int, default=3)
    parser.add_argument("--patience", type=int, default=4, help="evals without improvement before stopping")
    # hate is only mildly imbalanced (57/43) so focal down-weighting mostly adds
    # noise and hurts calibration; severity genuinely is imbalanced, so it keeps gamma=2.
    parser.add_argument("--hate_gamma", type=float, default=0.0)
    parser.add_argument("--severity_gamma", type=float, default=2.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--identity_aug", type=int, default=1,
                        help="augmented copies per identity-bearing train row "
                             "(0 disables; see identity_augment.py)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path("train.log"))
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log.info(f"device={device} amp={amp_dtype} args={vars(args)}")

    # --- guard: refuse to train on splits that overlap -----------------------
    frames = {}
    for name, path in [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]:
        if not Path(path).exists():
            log.error(f"missing {path} — run `python prepare_splits.py` first")
            return 1
        frames[name] = pd.read_csv(path)
    import re as _re
    keyed = {n: set(d["text"].astype(str).str.strip().str.casefold()
                    .str.replace(_re.compile(r"[^\w]+", _re.UNICODE), "", regex=True))
             for n, d in frames.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keyed[a] & keyed[b]
        if overlap:
            log.error(f"{len(overlap)} texts shared between {a} and {b} — "
                      f"re-run prepare_splits.py; refusing to train on leaked splits")
            return 1
    log.info(f"splits verified disjoint: "
             f"train={len(frames['train'])} val={len(frames['val'])} test={len(frames['test'])}")

    # Identity augmentation runs AFTER the disjointness check (so the check sees
    # real data) and only on train. Rewriting a group noun can in principle turn
    # a train row into a copy of a held-out one, so augmented rows are re-checked
    # against the same loose key and dropped on collision — augmentation must
    # never reintroduce the leakage prepare_splits.py just removed.
    if args.identity_aug > 0:
        before = len(frames["train"])
        augmented = expand_with_identity_augmentation(
            frames["train"], n_variants=args.identity_aug, seed=args.seed
        )
        eval_keys = keyed["val"] | keyed["test"]
        aug_keys = (augmented["text"].astype(str).str.strip().str.casefold()
                    .str.replace(_re.compile(r"[^\w]+", _re.UNICODE), "", regex=True))
        collide = aug_keys.isin(eval_keys)
        if collide.any():
            log.info(f"  dropped {int(collide.sum())} augmented rows colliding with val/test")
            augmented = augmented[~collide]
        frames["train"] = augmented.reset_index(drop=True)
        log.info(f"identity augmentation ({args.identity_aug} variant/row): "
                 f"{before} -> {len(frames['train'])} train rows "
                 f"(+{len(frames['train']) - before})")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.encoder_name)
    log.info("pre-tokenising...")
    t0 = time.time()
    datasets = {n: PreTokenizedDataset(d, tokenizer, args.max_length) for n, d in frames.items()}
    log.info(f"pre-tokenised in {time.time() - t0:.1f}s")

    collate = make_collate(tokenizer.pad_token_id)
    loader_kw = dict(collate_fn=collate, num_workers=args.num_workers,
                     pin_memory=(device.type == "cuda"),
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size * 2, shuffle=False, **loader_kw)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    model = CivitasDetector(
        encoder_name=args.encoder_name,
        num_target_classes=len(TARGET_CLASSES),
        num_severity_classes=len(SEVERITY_CLASSES),
    ).to(device)

    hate_weights = compute_class_weights(datasets["train"].hate, 2, device)
    hate_loss_fn = FocalLoss(gamma=args.hate_gamma, weight=hate_weights)
    sev = datasets["train"].severity
    sev_weights = compute_class_weights(sev[sev >= 0], len(SEVERITY_CLASSES), device) if (sev >= 0).any() else None
    severity_loss_fn = FocalLoss(gamma=args.severity_gamma, weight=sev_weights)
    target_loss_fn = nn.CrossEntropyLoss()
    rationale_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    log.info(f"class weights: hate={hate_weights.tolist()} severity={sev_weights.tolist() if sev_weights is not None else None}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    eval_every = max(1, len(train_loader) // args.evals_per_epoch)
    log.info(f"{len(train_loader)} batches/epoch, {total_steps} optimiser steps total, "
             f"eval every {eval_every} batches")

    best_f1, bad_evals, global_step = -1.0, 0, 0
    ckpt_path = out_dir / "best_model.pt"
    stop = False
    t_start = time.time()

    for epoch in range(args.epochs):
        if stop:
            break
        model.train()
        running, seen = 0.0, 0
        t_window = time.time()

        for step, batch in enumerate(train_loader):
            gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
            with amp_context(amp_dtype):
                outputs = model(input_ids=gpu["input_ids"], attention_mask=gpu["attention_mask"])
                loss, _ = compute_multitask_loss(
                    outputs, gpu, hate_loss_fn, target_loss_fn, severity_loss_fn, rationale_loss_fn
                )
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item(); seen += 1
            if step % 100 == 0:
                # rate over the window since the last log — using cumulative time
                # against an epoch-local step counter under-reports badly after
                # epoch 0 (it read 8 ex/s in epoch 3 for a run doing ~82).
                now = time.time()
                rate = seen * args.batch_size / (now - t_window + 1e-9)
                log.info(f"epoch {epoch} step {step}/{len(train_loader)} "
                         f"loss={running/seen:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                         f"{rate:.0f} ex/s")
                running, seen, t_window = 0.0, 0, now

            if (step + 1) % eval_every == 0 or (step + 1) == len(train_loader):
                pred = predict(model, val_loader, device, amp_dtype)
                thr, _ = tune_threshold(pred["hate_prob"], pred["hate_true"])
                metrics = score(pred, thr)
                metrics.update(epoch=epoch, step=step, global_step=global_step)
                log.info(f"--- eval @ epoch {epoch} step {step + 1} ---")
                log_metrics("val", metrics)

                if metrics["hate_macro_f1"] > best_f1:
                    best_f1, bad_evals = metrics["hate_macro_f1"], 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "threshold": thr,
                        "args": vars(args),
                        "val_metrics": metrics,
                        "target_classes": TARGET_CLASSES,
                        "severity_classes": SEVERITY_CLASSES,
                    }, ckpt_path)
                    (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
                    log.info(f"  -> new best (val macro-F1 {best_f1:.4f}), saved {ckpt_path}")
                else:
                    bad_evals += 1
                    log.info(f"  no improvement ({bad_evals}/{args.patience}), best={best_f1:.4f}")
                    if bad_evals >= args.patience:
                        log.info("early stopping")
                        stop = True
                        break
                model.train()

    # --- final: score the selected checkpoint on test, exactly once ----------
    log.info(f"training finished in {(time.time() - t_start)/60:.1f} min; best val macro-F1 {best_f1:.4f}")
    if not ckpt_path.exists():
        log.error("no checkpoint was saved")
        return 1

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"evaluating best checkpoint on TEST (threshold {ckpt['threshold']:.2f} fitted on val)")
    test_pred = predict(model, test_loader, device, amp_dtype)
    test_metrics = score(test_pred, ckpt["threshold"])
    log_metrics("test", test_metrics)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    log.info(f"wrote {out_dir / 'test_metrics.json'}")
    log.info(f"HEADLINE  val macro-F1={best_f1:.4f}  |  test macro-F1={test_metrics['hate_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
