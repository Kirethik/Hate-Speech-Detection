"""
Training entry point for Model A (Civitas AI multilingual detector).

Usage:
    python train.py --data_dir data --output_dir checkpoints/model_a
    python train.py ... --resume                 # continue from output_dir/last.pt
    python train.py --eval_only --split test --checkpoint checkpoints/model_a/best_model.pt

What this does, and why:

  * SPLIT HYGIENE. Refuses to run unless {train,val,test} exist and are
    disjoint (run prepare_splits.py first). `val` selects the checkpoint and
    the decision threshold. `test` is NEVER touched by a training run: it is
    scored only by the separate `--eval_only --split test` mode, once, at the
    very end of the project.
  * RESUMABLE. `last.pt` (model + optimizer + scheduler + scaler + position +
    RNG) is written every --save_every steps and at every eval, atomically, so
    a Colab disconnect loses minutes, not hours. Batch order is seeded per
    epoch, so a resumed run sees exactly the batches it would have seen.
  * DYNAMIC PADDING. Texts are pre-tokenised once and padded per batch to the
    longest member rather than to max_length.
  * MIXED PRECISION. bf16 on GPUs that have it natively (Ampere+: A100, L4,
    RTX 30xx); fp16 + GradScaler on T4/V100 (which only emulate bf16).
  * AUGMENTATION (train only, as extra copies, collision-checked against
    val/test): identity-term swaps, native<->latin transliteration (ASR emits
    native script; most data is romanized) and typo perturbations.
  * LANGUAGE WEIGHTS. data_config.yaml's per_language_weights become sampling
    weights, so thin languages (te, ml) are seen more often per epoch.
  * HONEST METRICS. Macro-F1 per language, per source and per script, plus
    per-class target F1, not just pooled numbers.
"""

import argparse
import contextlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from transformers import XLMRobertaTokenizerFast, get_linear_schedule_with_warmup

from dataset import _parse_spans
from identity_augment import expand_with_identity_augmentation
from label_maps import SEVERITY_CLASSES, TARGET_CLASSES
from model import CivitasDetector, FocalLoss, compute_multitask_loss
from script_augment import expand_with_script_augmentation
from spelling_augment import expand_with_spelling_augmentation
from text_norm import detect_script, normalize_code_mixed, to_original_span

log = logging.getLogger("train")

# Bumped whenever normalize_code_mixed changes behaviour, so a checkpoint
# records which normalisation it was trained with (inference must match).
NORMALIZATION_VERSION = 2


def setup_logging(log_path: Path, append: bool = False):
    """Log to stdout AND to a file, so `tail -f` works."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def pick_amp(device: torch.device, disable: bool):
    """(autocast dtype or None, use GradScaler)."""
    if disable or device.type != "cuda":
        return None, False
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:          # Ampere or newer: native bf16
        return torch.bfloat16, False
    return torch.float16, True


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


def atomic_save(obj, path: Path):
    """Write to a temp file then rename, so a disconnect never leaves a torn checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def loose_key(series: pd.Series) -> pd.Series:
    """Same loose key prepare_splits.py dedups on."""
    return series.astype(str).str.strip().str.casefold().str.replace(_NON_WORD, "", regex=True)


def load_split(data_dir: Path, name: str) -> pd.DataFrame | None:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = data_dir / f"{name}{ext}"
        if path.exists():
            df = reader(path)
            if "script" not in df.columns:
                df["script"] = df["text"].astype(str).map(detect_script)
            for col, default in (("target_label", -1), ("severity_label", -1),
                                 ("rationale_spans", "[]"), ("source", "unknown")):
                if col not in df.columns:
                    df[col] = default
            return df
    return None


class PreTokenizedDataset(Dataset):
    """
    Normalises and tokenises the whole frame once up front and keeps
    variable-length int32 arrays, so batches can be padded to their own
    longest member instead of to max_length.

    Rationale spans are char offsets into the ORIGINAL text; the model sees
    normalised text. Each token's normalised offsets are mapped back through
    normalize_code_mixed's offset map before checking overlap, so rationale
    supervision survives normalisation instead of being dropped.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        raw_texts = self.df["text"].astype(str).tolist()
        normed = [normalize_code_mixed(t) for t in raw_texts]
        texts = [t for t, _ in normed]

        enc = tokenizer(texts, truncation=True, max_length=max_length,
                        padding=False, return_offsets_mapping=True)
        self.input_ids = [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]

        self.hate = self.df["hate_label"].astype(int).to_numpy()
        self.target = self.df["target_label"].fillna(-1).astype(int).to_numpy()
        self.severity = self.df["severity_label"].fillna(-1).astype(int).to_numpy()
        self.language = self.df["language"].astype(str).tolist()
        self.source = self.df["source"].astype(str).tolist()
        self.script = self.df["script"].astype(str).tolist()

        self.rationale = []
        self.has_rationale = np.zeros(len(self.df), dtype=bool)
        for i, raw_spans in enumerate(self.df["rationale_spans"].astype(str).tolist()):
            n_tok = len(self.input_ids[i])
            spans = _parse_spans(raw_spans)
            labels = np.full(n_tok, -100, dtype=np.int64)
            if spans:
                offset_map = normed[i][1]
                for t, (start, end) in enumerate(enc["offset_mapping"][i]):
                    if start == end:      # special token
                        continue
                    o_start, o_end = to_original_span(start, end, offset_map, raw_texts[i])
                    labels[t] = int(any(o_start < s_end and o_end > s_start
                                        for s_start, s_end in spans))
                self.has_rationale[i] = True
            self.rationale.append(labels)

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
            "script": self.script[i],
        }


class Collate:
    """A class rather than a closure so DataLoader workers can pickle it on Windows."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        n = len(batch)
        input_ids = np.full((n, maxlen), self.pad_token_id, dtype=np.int64)
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
        for key in ("language", "source", "script"):
            out[key] = [b[key] for b in batch]
        return out


def compute_class_weights(labels, num_classes, device):
    labels = np.asarray(labels)
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present, weights):
        full[cls] = w
    return torch.tensor(full, dtype=torch.float32, device=device)


def language_sample_weights(languages: list[str], config_path: str | None) -> np.ndarray | None:
    """Per-row sampling weights from data_config.yaml per_language_weights (or None)."""
    if not config_path or not Path(config_path).exists():
        return None
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    weights = cfg.get("per_language_weights") or {}
    if not weights or all(float(w) == 1.0 for w in weights.values()):
        return None
    return np.array([float(weights.get(lang, 1.0)) for lang in languages], dtype=np.float64)


def make_train_loader(ds, batch_size, epoch, seed, sample_weights, collate, loader_kw):
    """Seeded per epoch, so a resumed run replays the exact same batch order."""
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights), num_samples=len(ds),
                                        replacement=True, generator=g)
    else:
        sampler = RandomSampler(ds, generator=g)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True,
                      collate_fn=collate, **loader_kw)


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

        for key in ("language", "source", "script"):
            out[key].extend(batch[key])

    return {k: (np.array(v) if k in ("language", "source", "script") else np.concatenate(v))
            for k, v in out.items()}


def tune_threshold(probs, labels):
    """Pick the hate-head decision threshold that maximises macro-F1 on val."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return round(best_t, 2), best_f1


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

    p, r, f, sup = precision_recall_fscore_support(hate_true, hate_pred, labels=[0, 1], zero_division=0)
    metrics["hate_per_class"] = {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
        for i, name in enumerate(["not_abusive", "abusive"])
    }

    for slice_name in ("language", "source", "script"):
        breakdown = {}
        for raw_key in sorted(set(pred[slice_name])):
            key = str(raw_key)
            m = pred[slice_name] == raw_key
            if m.sum() == 0 or len(set(hate_true[m])) < 2:
                breakdown[key] = None      # single-class slice: F1 not meaningful
                continue
            breakdown[key] = {"macro_f1": float(f1_score(hate_true[m], hate_pred[m], average="macro")),
                              "n": int(m.sum())}
        metrics[f"hate_by_{slice_name}"] = breakdown

    sev_m = pred["sev_mask"].astype(bool)
    if sev_m.sum():
        metrics["severity_macro_f1"] = float(
            f1_score(pred["sev_true"][sev_m], pred["sev_pred"][sev_m], average="macro"))
        metrics["severity_n"] = int(sev_m.sum())

    tgt_m = pred["target_mask"].astype(bool)
    if tgt_m.sum():
        t_true, t_pred = pred["target_true"][tgt_m], pred["target_pred"][tgt_m]
        metrics["target_macro_f1"] = float(f1_score(t_true, t_pred, average="macro", zero_division=0))
        _, _, tf, tsup = precision_recall_fscore_support(
            t_true, t_pred, labels=list(range(len(TARGET_CLASSES))), zero_division=0)
        metrics["target_per_class"] = {
            TARGET_CLASSES[i]: {"f1": float(tf[i]), "support": int(tsup[i])}
            for i in range(len(TARGET_CLASSES))
        }
        metrics["target_n"] = int(tgt_m.sum())

    if len(pred["rat_true"]):
        metrics["rationale_token_f1"] = float(
            f1_score(pred["rat_true"], pred["rat_pred"], average="macro", zero_division=0))
        metrics["rationale_n_tokens"] = int(len(pred["rat_true"]))
    return metrics


def log_metrics(tag: str, m: dict):
    log.info(f"  [{tag}] hate macro-F1={m['hate_macro_f1']:.4f} "
             f"acc={m['hate_accuracy']:.4f} auc={m.get('hate_roc_auc') or float('nan'):.4f} "
             f"@thr={m['threshold']:.2f}")
    for slice_name in ("language", "source", "script"):
        vals = {k: (round(v["macro_f1"], 3) if v else None) for k, v in m[f"hate_by_{slice_name}"].items()}
        log.info(f"  [{tag}] by {slice_name}: {vals}")
    aux = [f"{label}={m[key]:.4f}" for key, label in
           [("severity_macro_f1", "severity"), ("target_macro_f1", "target"),
            ("rationale_token_f1", "rationale")] if key in m]
    if aux:
        log.info(f"  [{tag}] aux heads: " + "  ".join(aux))


def append_history(out_dir: Path, m: dict):
    """One flat CSV row per eval, for plotting learning curves."""
    row = {k: m.get(k) for k in ("epoch", "step", "global_step", "threshold", "hate_macro_f1",
                                 "hate_accuracy", "hate_roc_auc", "severity_macro_f1",
                                 "target_macro_f1", "rationale_token_f1")}
    for lang, v in m["hate_by_language"].items():
        row[f"f1_{lang}"] = v["macro_f1"] if v else None
    path = out_dir / "metrics_history.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# --------------------------------------------------------------------------- #
# Model loading shared by --eval_only and inference
# --------------------------------------------------------------------------- #
def build_model(encoder_name: str) -> CivitasDetector:
    return CivitasDetector(encoder_name=encoder_name,
                           num_target_classes=len(TARGET_CLASSES),
                           num_severity_classes=len(SEVERITY_CLASSES))


def eval_only(args) -> int:
    """Score a saved checkpoint on one split with its val-fitted threshold."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / f"eval_{args.split}.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = pick_amp(device, args.no_amp)

    ckpt_path = Path(args.checkpoint or out_dir / "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder = ckpt.get("args", {}).get("encoder_name", args.encoder_name)
    model = build_model(encoder)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    df = load_split(Path(args.data_dir), args.split)
    if df is None:
        log.error(f"no {args.split} split in {args.data_dir}")
        return 1
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(encoder)
    ds = PreTokenizedDataset(df, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                        collate_fn=Collate(tokenizer.pad_token_id))
    if args.split == "test":
        log.info("=" * 70)
        log.info("SCORING THE TEST SPLIT. Do this once, after all tuning is finished.")
        log.info("Any change made after looking at these numbers invalidates them.")
        log.info("=" * 70)
    metrics = score(predict(model, loader, device, amp_dtype), ckpt["threshold"])
    metrics["checkpoint"] = str(ckpt_path)
    log_metrics(args.split, metrics)
    report = out_dir / f"{args.split}_report.json"
    report.write_text(json.dumps(metrics, indent=2))
    log.info(f"wrote {report}")
    return 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="holds {train,val,test}.{parquet,csv}")
    p.add_argument("--output_dir", default="checkpoints/model_a")
    p.add_argument("--encoder_name", default="xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--evals_per_epoch", type=int, default=3)
    p.add_argument("--patience", type=int, default=4, help="evals without improvement before stopping")
    p.add_argument("--save_every", type=int, default=500, help="optimizer steps between last.pt saves")
    p.add_argument("--resume", action="store_true", help="continue from output_dir/last.pt if present")
    p.add_argument("--max_train_rows", type=int, default=0, help="debug: subsample train (0 = all)")
    # loss
    # hate is only mildly imbalanced, so focal down-weighting mostly adds noise and
    # hurts calibration; severity genuinely is imbalanced, so it keeps gamma=2.
    p.add_argument("--hate_gamma", type=float, default=0.0)
    p.add_argument("--severity_gamma", type=float, default=2.0)
    p.add_argument("--w_hate", type=float, default=1.0)
    p.add_argument("--w_target", type=float, default=0.5)
    p.add_argument("--w_severity", type=float, default=0.5)
    p.add_argument("--w_rationale", type=float, default=0.5)
    p.add_argument("--no_amp", action="store_true")
    # augmentation (train only)
    p.add_argument("--identity_aug", type=int, default=1,
                   help="augmented copies per identity-bearing train row (0 disables)")
    p.add_argument("--script_augment_p", type=float, default=0.3,
                   help="fraction of non-English train rows to add transliterated copies of")
    p.add_argument("--spelling_augment_p", type=float, default=0.15,
                   help="fraction of train rows to add typo-perturbed copies of")
    p.add_argument("--lang_weights", default="data_config.yaml",
                   help="yaml with per_language_weights for sampling ('' disables)")
    # evaluation-only mode
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--checkpoint", default=None)
    return p.parse_args(argv)


def check_disjoint(frames: dict) -> dict | None:
    keyed = {n: set(loose_key(d["text"])) for n, d in frames.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keyed[a] & keyed[b]
        if overlap:
            log.error(f"{len(overlap)} texts shared between {a} and {b} — "
                      f"re-run prepare_splits.py; refusing to train on leaked splits")
            return None
    return keyed


def augment_train(train: pd.DataFrame, eval_keys: set, args) -> pd.DataFrame:
    """Add augmented copies, then drop any copy that collides with val/test."""
    before = len(train)
    if args.identity_aug > 0:
        train = expand_with_identity_augmentation(train, n_variants=args.identity_aug, seed=args.seed)
    base = train
    if args.script_augment_p > 0:
        train = expand_with_script_augmentation(train, frac=args.script_augment_p, seed=args.seed)
    if args.spelling_augment_p > 0:
        # typo copies are drawn from the pre-transliteration rows only
        extra = expand_with_spelling_augmentation(base, frac=args.spelling_augment_p, seed=args.seed)
        train = pd.concat([train, extra.iloc[len(base):]], ignore_index=True)
    collide = loose_key(train["text"]).isin(eval_keys)
    collide[:before] = False  # originals were already verified disjoint
    if collide.any():
        log.info(f"  dropped {int(collide.sum())} augmented rows colliding with val/test")
        train = train[~collide]
    train = train.reset_index(drop=True)
    added = train["source"].astype(str).str.extract(r"_(aug|translit|typo)$")[0].value_counts().to_dict()
    log.info(f"augmentation: {before} -> {len(train)} train rows  {added}")
    return train


def main(argv=None):
    args = parse_args(argv)
    if args.eval_only:
        return eval_only(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best_model.pt"
    resuming = args.resume and last_path.exists()
    setup_logging(out_dir / "train.log", append=resuming)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = pick_amp(device, args.no_amp)
    num_workers = 0 if os.name == "nt" else args.num_workers
    log.info(f"device={device} amp={amp_dtype} scaler={use_scaler} resume={resuming}")
    log.info(f"args={vars(args)}")

    # --- data + split hygiene ------------------------------------------------
    frames = {}
    for name in ("train", "val", "test"):
        df = load_split(Path(args.data_dir), name)
        if df is None:
            log.error(f"missing {name}.parquet/.csv in {args.data_dir} — run prepare_splits.py first")
            return 1
        frames[name] = df
    keyed = check_disjoint(frames)
    if keyed is None:
        return 1
    log.info(f"splits verified disjoint: " + " ".join(f"{n}={len(d)}" for n, d in frames.items()))
    del frames["test"]  # training never touches test

    train = frames["train"]
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=args.seed)
    train = augment_train(train, keyed["val"] | keyed["test"], args)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.encoder_name)
    log.info("pre-tokenising...")
    t0 = time.time()
    train_ds = PreTokenizedDataset(train, tokenizer, args.max_length)
    val_ds = PreTokenizedDataset(frames["val"], tokenizer, args.max_length)
    log.info(f"pre-tokenised in {time.time() - t0:.1f}s; "
             f"{int(train_ds.has_rationale.sum())} train rows carry rationale supervision")

    collate = Collate(tokenizer.pad_token_id)
    loader_kw = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"),
                     persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collate, **loader_kw)
    sample_weights = language_sample_weights(train_ds.language, args.lang_weights)
    if sample_weights is not None:
        per_lang = dict(zip(train_ds.language, sample_weights.tolist()))
        log.info(f"language sampling weights: {per_lang}")

    # --- model / losses / optimiser ---------------------------------------
    model = build_model(args.encoder_name).to(device)
    hate_weights = compute_class_weights(train_ds.hate, 2, device)
    hate_loss_fn = FocalLoss(gamma=args.hate_gamma, weight=hate_weights)
    sev = train_ds.severity
    sev_weights = compute_class_weights(sev[sev >= 0], len(SEVERITY_CLASSES), device) if (sev >= 0).any() else None
    severity_loss_fn = FocalLoss(gamma=args.severity_gamma, weight=sev_weights)
    target_loss_fn = nn.CrossEntropyLoss()
    rationale_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_weights = {"hate": args.w_hate, "target": args.w_target,
                    "severity": args.w_severity, "rationale": args.w_rationale}
    log.info(f"class weights: hate={hate_weights.tolist()} "
             f"severity={sev_weights.tolist() if sev_weights is not None else None} "
             f"loss weights={loss_weights}")

    batches_per_epoch = len(train_ds) // args.batch_size
    steps_per_epoch = batches_per_epoch // args.grad_accum
    total_steps = max(1, steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    eval_every = max(1, batches_per_epoch // args.evals_per_epoch)
    log.info(f"{batches_per_epoch} batches/epoch, {total_steps} optimiser steps total, "
             f"eval every {eval_every} batches")

    state = {"epoch": 0, "batch": 0, "global_step": 0, "best_score": -1.0, "bad_evals": 0}
    if resuming:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        log.info(f"resumed from {last_path}: epoch {state['epoch']} batch {state['batch']} "
                 f"(global step {state['global_step']}, best {state['best_score']:.4f})")

    def save_last():
        atomic_save({
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "state": dict(state),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
            "args": vars(args),
        }, last_path)

    def evaluate_and_maybe_save() -> bool:
        """Returns True when early stopping triggers."""
        pred = predict(model, val_loader, device, amp_dtype)
        thr, _ = tune_threshold(pred["hate_prob"], pred["hate_true"])
        metrics = score(pred, thr)
        metrics.update(epoch=state["epoch"], step=state["batch"], global_step=state["global_step"])
        log.info(f"--- eval @ epoch {state['epoch']} batch {state['batch']} ---")
        log_metrics("val", metrics)
        append_history(out_dir, metrics)
        stop = False
        if metrics["hate_macro_f1"] > state["best_score"]:
            state["best_score"], state["bad_evals"] = metrics["hate_macro_f1"], 0
            atomic_save({
                "model_state_dict": model.state_dict(),
                "threshold": thr,
                "args": vars(args),
                "val_metrics": metrics,
                "target_classes": TARGET_CLASSES,
                "severity_classes": SEVERITY_CLASSES,
                "normalization_version": NORMALIZATION_VERSION,
            }, best_path)
            (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
            log.info(f"  -> new best (val macro-F1 {state['best_score']:.4f}), saved {best_path}")
        else:
            state["bad_evals"] += 1
            log.info(f"  no improvement ({state['bad_evals']}/{args.patience}), "
                     f"best={state['best_score']:.4f}")
            stop = state["bad_evals"] >= args.patience
        model.train()
        return stop

    # --- loop ----------------------------------------------------------------
    t_start = time.time()
    stop = False
    for epoch in range(state["epoch"], args.epochs):
        state["epoch"] = epoch
        loader = make_train_loader(train_ds, args.batch_size, epoch, args.seed,
                                   sample_weights, collate, loader_kw)
        skip = state["batch"]
        model.train()
        running, seen, t_window = 0.0, 0, time.time()
        for step, batch in enumerate(loader):
            if step < skip:
                continue  # already trained on before the disconnect
            gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
            with amp_context(amp_dtype):
                outputs = model(input_ids=gpu["input_ids"], attention_mask=gpu["attention_mask"])
                loss, _ = compute_multitask_loss(outputs, gpu, hate_loss_fn, target_loss_fn,
                                                 severity_loss_fn, rationale_loss_fn, loss_weights)
            scaler.scale(loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["global_step"] += 1
                if state["global_step"] % args.save_every == 0:
                    state["batch"] = step + 1
                    save_last()

            state["batch"] = step + 1
            running += loss.item(); seen += 1
            if step % 100 == 0:
                now = time.time()
                rate = seen * args.batch_size / (now - t_window + 1e-9)
                log.info(f"epoch {epoch} step {step}/{batches_per_epoch} loss={running / seen:.4f} "
                         f"lr={scheduler.get_last_lr()[0]:.2e} {rate:.0f} ex/s")
                running, seen, t_window = 0.0, 0, now

            if (step + 1) % eval_every == 0 or (step + 1) == batches_per_epoch:
                stop = evaluate_and_maybe_save()
                save_last()
                if stop:
                    log.info("early stopping")
                    break
        if stop:
            break
        state["epoch"], state["batch"] = epoch + 1, 0
        save_last()

    log.info(f"training finished in {(time.time() - t_start) / 60:.1f} min; "
             f"best val macro-F1 {state['best_score']:.4f}")
    if not best_path.exists():
        log.error("no checkpoint was saved")
        return 1
    log.info(f"best checkpoint: {best_path}. The test split was NOT scored. When all tuning "
             f"is done, run once:  python train.py --eval_only --split test "
             f"--data_dir {args.data_dir} --output_dir {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
