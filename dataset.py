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
  language          str   one of: en, hi, ur_roman, ta, te (ISO-ish codes you chose)
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
import unicodedata
import re

_DEVANAGARI = range(0x0900, 0x0980)
_TAMIL = range(0x0B80, 0x0C00)
_TELUGU = range(0x0C00, 0x0C80)
_MALAYALAM = range(0x0D00, 0x0D80)
_ARABIC_URDU = range(0x0600, 0x0700)

_INDIC_RANGES = [_DEVANAGARI, _TAMIL, _TELUGU, _MALAYALAM, _ARABIC_URDU]


def detect_script(text: str) -> str:
    """
    Returns 'native' if text is predominantly Indic/Arabic script,
    'latin' if predominantly Latin, 'mixed' if both are significant.
    """
    native_count = sum(1 for c in text if any(ord(c) in r for r in _INDIC_RANGES))
    latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 0x0250)
    total = native_count + latin_count
    if total == 0:
        return 'latin'
    native_ratio = native_count / total
    if native_ratio > 0.8:
        return 'native'
    elif native_ratio < 0.2:
        return 'latin'
    return 'mixed'

# NOTE: "political" is no longer a dead class — it is resolved by implicit-hate data.
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
                    
                    orig_start = offset_map[start] if start < len(offset_map) else len(raw_text)
                    orig_end = offset_map[end - 1] + 1 if end > 0 and (end - 1) < len(offset_map) else len(raw_text)

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


def normalize_code_mixed(text: str) -> tuple[str, list[int]]:
    """
    Normalizes text for hate speech detection.
    Returns: (normalized_text, offset_map)
    where offset_map[i] = position in original text corresponding to position i in normalized.
    """
    offset_map = list(range(len(text)))

    # 1. Unicode NFKC normalization + strip zero-width chars
    t1 = []
    m1 = []
    for i, c in enumerate(text):
        if c in ('\u200b', '\u200c', '\u200d', '\ufeff'):
            continue
        norm_c = unicodedata.normalize('NFKC', c)
        for nc in norm_c:
            t1.append(nc)
            m1.append(offset_map[i])

    text1 = "".join(t1)

    # 2. Unify quotes
    quote_map = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'"}
    t2 = [quote_map.get(c, c) for c in text1]
    m2 = m1
    text2 = "".join(t2)

    def apply_regex(t, offsets, pattern, replacer):
        new_t = []
        new_m = []
        last_end = 0
        for match in re.finditer(pattern, t):
            new_t.extend(t[last_end:match.start()])
            new_m.extend(offsets[last_end:match.start()])
            rs, ro = replacer(match, offsets)
            new_t.extend(rs)
            new_m.extend(ro)
            last_end = match.end()
        new_t.extend(t[last_end:])
        new_m.extend(offsets[last_end:])
        return "".join(new_t), new_m

    # 3. Collapse deliberately spaced single chars
    def unspace(match, offsets):
        ms = match.group(0)
        mo = offsets[match.start():match.end()]
        rs, ro = [], []
        for c, o in zip(ms, mo):
            if c != ' ':
                rs.append(c)
                ro.append(o)
        return rs, ro
    text3, m3 = apply_regex(text2, m2, r'(?i)(?:[a-z]\s+){2,}[a-z]', unspace)

    # 4. Collapse repeated chars >2 consecutive identical
    def unrepeat(match, offsets):
        ms = match.group(0)
        mo = offsets[match.start():match.end()]
        return ms[:2], mo[:2]
    text4, m4 = apply_regex(text3, m3, r'(.)\1{2,}', unrepeat)

    # 5. Leetspeak reversal ONLY inside alphabetic tokens (never in numbers/URLs)
    leet_map = {'0':'o', '1':'i', '3':'e', '4':'a', '5':'s', '6':'g', '7':'t', '@':'a', '$':'s'}
    def deleet(match, offsets):
        ms = match.group(0)
        mo = offsets[match.start():match.end()]
        if not ms.startswith('http') and re.search(r'[a-zA-Z]', ms):
            rs = [leet_map.get(c, c) for c in ms]
        else:
            rs = list(ms)
        return rs, mo
    text5, m5 = apply_regex(text4, m4, r'[a-zA-Z0-9@$]+', deleet)

    # 6. Keep emojis (never strip them) - implicitly handled because we don't strip them
    return text5, m5
