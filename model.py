"""
Model A — Civitas AI Multilingual Detector
Shared XLM-RoBERTa encoder + four task heads:
  1. Hate (binary)               -> sequence classification
  2. Target group (multi-class)  -> sequence classification
  3. Severity (ordinal)          -> sequence classification (treated as multi-class
                                     with an ordinal-aware loss option)
  4. Token rationale              -> token classification (BIO-less binary tagging:
                                     each token is "rationale" or "not")

Design notes:
- One shared encoder keeps the model small enough to fine-tune on a single GPU
  and forces the heads to share representations (matches the "shared encoder"
  box in your architecture slide).
- Heads are independent linear layers on top of pooled / token-level hidden
  states, so you can disable any head (e.g. if a batch has no rationale
  labels) without touching the others.
- Losses are combined with configurable weights so you can tune how much the
  rationale head (sparse labels: only HateXplain + HASOC21 have it) drags on
  training vs. the hate/target/severity heads (dense labels: every dataset
  has them).
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig


class FocalLoss(nn.Module):
    """
    Standard multi-class focal loss. Use this instead of plain cross-entropy
    for the hate and severity heads, since both are class-imbalanced
    (per your slide 11: "non-abusive text dominates and severe abuse is rare").
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class CivitasDetector(nn.Module):
    def __init__(
        self,
        encoder_name: str = "xlm-roberta-base",
        num_target_classes: int = 8,     # e.g. none, religion, gender, caste, ethnicity, disability, political, other
        num_severity_classes: int = 3,   # e.g. normal/offensive/hate or profanity/offensive/hate
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Pooled (CLS-based) heads
        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 2),  # binary: hate / not-hate
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_target_classes),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_severity_classes),
        )

        # Token-level head (rationale): binary per-token classification
        self.rationale_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 2),  # 0 = not rationale, 1 = rationale
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state          # (B, T, H) — token-level
        pooled_output = sequence_output[:, 0, :]              # CLS token — sequence-level
        pooled_output = self.dropout(pooled_output)

        hate_logits = self.hate_head(pooled_output)
        target_logits = self.target_head(pooled_output)
        severity_logits = self.severity_head(pooled_output)
        rationale_logits = self.rationale_head(self.dropout(sequence_output))

        return {
            "hate_logits": hate_logits,
            "target_logits": target_logits,
            "severity_logits": severity_logits,
            "rationale_logits": rationale_logits,
        }


def compute_multitask_loss(
    outputs: dict,
    batch: dict,
    hate_loss_fn,
    target_loss_fn,
    severity_loss_fn,
    rationale_loss_fn,
    loss_weights: dict | None = None,
):
    """
    Combines all four head losses. Any label that's missing for a given
    example (mask == 0) is excluded from that head's loss — this is what lets
    you mix datasets that only have hate labels (e.g. MACD) with datasets that
    also have target/severity/rationale (e.g. HASOC21, HateXplain) in the same
    training run.
    """
    weights = loss_weights or {"hate": 1.0, "target": 0.5, "severity": 0.5, "rationale": 0.5}

    total_loss = 0.0
    loss_dict = {}

    # Hate — always present
    hate_loss = hate_loss_fn(outputs["hate_logits"], batch["hate_label"])
    total_loss += weights["hate"] * hate_loss
    loss_dict["hate_loss"] = hate_loss.item()

    # Target — masked
    if batch["target_mask"].any():
        idx = batch["target_mask"].bool()
        target_loss = target_loss_fn(outputs["target_logits"][idx], batch["target_label"][idx])
        total_loss += weights["target"] * target_loss
        loss_dict["target_loss"] = target_loss.item()

    # Severity — masked
    if batch["severity_mask"].any():
        idx = batch["severity_mask"].bool()
        severity_loss = severity_loss_fn(outputs["severity_logits"][idx], batch["severity_label"][idx])
        total_loss += weights["severity"] * severity_loss
        loss_dict["severity_loss"] = severity_loss.item()

    # Rationale — masked, token-level (only where rationale_mask == 1 per example)
    if batch["rationale_mask"].any():
        idx = batch["rationale_mask"].bool()
        r_logits = outputs["rationale_logits"][idx]           # (n, T, 2)
        r_labels = batch["rationale_labels"][idx]              # (n, T) with -100 for ignored positions
        rationale_loss = rationale_loss_fn(
            r_logits.reshape(-1, 2), r_labels.reshape(-1)
        )
        total_loss += weights["rationale"] * rationale_loss
        loss_dict["rationale_loss"] = rationale_loss.item()

    loss_dict["total_loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    return total_loss, loss_dict
