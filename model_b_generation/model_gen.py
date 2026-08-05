"""
Model B — QLoRA-wrapped mT5-small loader.

Two entry points:
  - load_model_for_training(): loads the base model in 4-bit NF4
    (frozen) and applies LoRA adapters for parameter-efficient fine-tuning.
  - load_model_for_inference(): loads the base model plus a saved LoRA
    adapter checkpoint for generation.

QLoRA config (Section 5 of the blueprint):
  - Base: google/mt5-small (~300M params) in 4-bit NormalFloat
  - LoRA: r=8, alpha=32, target_modules=['q', 'v'], dropout=0.05
  - Only the adapter weights are trained; the base is frozen
"""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

def load_model_for_training(model_name="google/mt5-small", lora_r=8, resume_checkpoint=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device_map = "auto" if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=['q', 'v'],
            lora_dropout=0.05,
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, lora_config)
        
    model.print_trainable_parameters()
    
    return model, tokenizer

def load_model_for_inference(checkpoint_dir, model_name="google/mt5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device_map = "auto" if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    return model, tokenizer
