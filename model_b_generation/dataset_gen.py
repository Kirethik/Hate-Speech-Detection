"""
Unified dataset loader for Model B — counter-narrative generation.

Reads CSVs produced by build_dataset_gen.py with the unified schema:
  hate_text, language, style, response_text, source

Tokenizes each row into a seq2seq training example with the task prefix:
  "generate [factual|empathetic] counter-narrative in [English|Hindi|Tamil]: {hate_text}"

Target output: the raw response string, no prefix. Pad tokens in labels
are set to -100 for standard T5 cross-entropy loss masking.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

LANGUAGE_MAP = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

class CivitasGenDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_input_length: int = 128, max_output_length: int = 64):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        self.data = pd.read_csv(csv_path)
        # Ensure necessary columns are present
        required_cols = ['hate_text', 'language', 'style', 'response_text']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.data = self.data.dropna(subset=required_cols).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        hate_text = str(row['hate_text'])
        lang_code = str(row['language']).lower()
        style = str(row['style']).lower()
        response_text = str(row['response_text'])
        
        lang = LANGUAGE_MAP.get(lang_code, "English")
        
        # Task prefix exactly: generate [factual|empathetic] counter-narrative in [English|Hindi|Tamil]: {hate_text}
        input_text = f"generate {style} counter-narrative in {lang}: {hate_text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            response_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()
        
        # Set pad tokens in labels to -100 for loss masking
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
