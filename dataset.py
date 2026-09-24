"""
Unified dataset loader for Model A.

Every source you listed (MACD, DravidianCodeMix, HASOC 2019-21, RUHSOLD,
HateXplain, Bohra et al.) has a different raw schema. Rather than writing a
bespoke training loop per dataset, convert each one to this single CSV schema
first (one small conversion script per source — see `converters/` stub below),
then everything downstream (tokenization, batching, multi-task loss) is
identical regardless of which dataset a row came from.

UNIFIED SCHEMA (one row per example):
  text              str   raw text, original script/Romanization preserved
  language          str   one of config.SUPPORTED_LANGUAGES (en, hi, ur, ur_roman, ta, te, ml, kn)
  hate_label        int   0 = not hate, 1 = hate            (ALWAYS required)
  target_label      int   class index into TARGET_CLASSES, or -1 if unknown
  severity_label    int   class index into SEVERITY_CLASSES, or -1 if unknown
  rationale_spans   str   JSON list of [start_char, end_char] spans, or "[]"
                          if this example has no rationale annotation at all
                          (as opposed to an annotated-but-empty span list)
  source            str   dataset name, for per-source / per-language eval breakdowns
  script            str   'native', 'latin', or 'mixed'

Missing target/severity/rationale labels are handled by masking (see model.py),
not by skipping the example — this is what lets weakly-labeled sources
(e.g. MACD: hate label only) train jointly with richly-labeled ones
(e.g. HateXplain: hate + target + rationale).
"""

import json
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset

from text_norm import detect_script, normalize_code_mixed, to_original_span  # noqa: F401  (re-exported)

# "political" targets come from SBIC's "social" targetCategory (see label_maps.py).
from label_maps import TARGET_CLASSES, SEVERITY_CLASSES


class CivitasDetectorDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128, augmenter=None):
        """
        augmenter: optional IdentityAugmenter (see identity_augment.py). Pass it
        for the TRAINING split only — augmenting val/test would defeat the point,
        which is to measure generalisation to groups the model never trained on.
        """
        self.df = pd.read_csv(csv_path)
        self.augmenter = augmenter
        required = {"text", "language", "hate_label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV at {csv_path} is missing required columns: {missing}")

        # Optional columns default to "unknown" (-1) if not present at all
        for col, default in [("target_label", -1), ("severity_label", -1), ("rationale_spans", "[]")]:
            if col not in self.df.columns:
                self.df[col] = default

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        hate_label = int(row["hate_label"])
        target_label = int(row["target_label"])
        severity_label = int(row["severity_label"])

        spans = _parse_spans(row["rationale_spans"])
        has_rationale = len(spans) > 0 or str(row["rationale_spans"]) not in ("[]", "", "nan")

        raw_text = str(row["text"])
        # Rationale spans are CHARACTER offsets into the original string, so any
        # augmentation that changes the text length silently invalidates them —
        # the rationale head would then be trained against misaligned tokens.
        # Rationale-annotated rows (HateXplain, ~9k of 229k) therefore opt out of
        # augmentation rather than lose their span supervision.
        if self.augmenter is not None and not has_rationale:
            raw_text, target_label = self.augmenter(raw_text, target_label)
            
        script = str(row.get("script", detect_script(raw_text))) if "script" in self.df.columns else detect_script(raw_text)

        norm_text, offset_map = normalize_code_mixed(raw_text)

        encoding = self.tokenizer(
            norm_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)

        # Build token-level rationale labels from char spans, -100 = ignore
        rationale_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        if has_rationale:
            # Default every real (non-special) token to "not rationale" (0),
            # then flip tokens overlapping an annotated span to "rationale" (1)
            for i, (start, end) in enumerate(offsets.tolist()):
                if start == end:  # special tokens ([CLS], [SEP], [PAD]) have (0,0)
                    continue
                rationale_labels[i] = 0
            for span_start, span_end in spans:
                for i, (start, end) in enumerate(offsets.tolist()):
                    if start == end:
                        continue
                    
                    orig_start, orig_end = to_original_span(start, end, offset_map, raw_text)

                    if orig_start < span_end and orig_end > span_start:
                        rationale_labels[i] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hate_label": torch.tensor(hate_label, dtype=torch.long),
            "target_label": torch.tensor(max(target_label, 0), dtype=torch.long),
            "target_mask": torch.tensor(1 if target_label >= 0 else 0, dtype=torch.long),
            "severity_label": torch.tensor(max(severity_label, 0), dtype=torch.long),
            "severity_mask": torch.tensor(1 if severity_label >= 0 else 0, dtype=torch.long),
            "rationale_labels": rationale_labels,
            "rationale_mask": torch.tensor(1 if has_rationale else 0, dtype=torch.long),
            "language": row["language"],
            "source": row.get("source", "unknown"),
            "script": script,
        }


def _parse_spans(raw):
    if raw is None or (isinstance(raw, float)):  # NaN
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []
