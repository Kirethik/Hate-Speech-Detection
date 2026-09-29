"""
Model B = bigscience/mt0-base + a LoRA adapter (no bitsandbytes / 4-bit).

mT5-family weights overflow in fp16, so training runs in fp32 on T4 and
bf16 on Ampere+ (L4/A100). Inference on the RTX 3050 uses bf16 when the GPU
supports it, else fp32 (mt0-base is ~580M params -> ~2.3 GB fp32, ~1.2 GB bf16).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_BASE = "bigscience/mt0-base"
# attention + feed-forward projections of the T5/mT5 blocks
LORA_TARGETS = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]


def load_for_training(base: str = DEFAULT_BASE, lora_r: int = 16, lora_alpha: int = 32,
                      lora_dropout: float = 0.05, adapter_dir: str | None = None):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    if adapter_dir and (Path(adapter_dir) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=LORA_TARGETS, task_type=TaskType.SEQ_2_SEQ_LM))
    return model, tok


@dataclass
class ModelB:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    base: str
    adapter_dir: str | None


def inference_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_for_inference(adapter_dir: str | None, base: str = DEFAULT_BASE,
                       device: str | torch.device = "auto", merge: bool = True) -> ModelB:
    """
    Base + adapter, merged into plain weights for faster generation.
    With no adapter the bare mt0-base is returned (zero-shot; poor quality,
    only good for smoke tests) and adapter_dir is None.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    has_adapter = bool(adapter_dir) and (Path(adapter_dir) / "adapter_config.json").exists()
    tok_src = adapter_dir if has_adapter and (Path(adapter_dir) / "tokenizer_config.json").exists() else base
    tok = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    if has_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge:
            model = model.merge_and_unload()
    model.to(device=device, dtype=inference_dtype(device)).eval()
    return ModelB(model, tok, device, base, adapter_dir if has_adapter else None)
