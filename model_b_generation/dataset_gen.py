"""
Seq2seq dataset for Model B. Rows come from prepare_splits_gen.py
(PAIR_COLUMNS); prompts come from prompts.build_prompt, the same helper
inference uses.

Padding is dynamic (per batch, in `collate`), not max_length, so short
chat-style sentences don't waste compute on T4.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from model_b_generation.prompts import build_prompt


def read_pairs(path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    need = {"task", "source_text", "target_text", "language"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)}")
    if "target" not in df.columns:
        df["target"] = "unknown"
    return df.dropna(subset=list(need)).reset_index(drop=True)


class GenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_input_length: int = 128,
                 max_output_length: int = 64):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_in = max_input_length
        self.max_out = max_output_length

    def __len__(self):
        return len(self.df)

    def prompt(self, i: int) -> str:
        r = self.df.iloc[i]
        return build_prompt(r["task"], r["language"], r["source_text"], r["target"])

    def __getitem__(self, i):
        enc = self.tok(self.prompt(i), max_length=self.max_in, truncation=True)
        lab = self.tok(text_target=str(self.df.iloc[i]["target_text"]),
                       max_length=self.max_out, truncation=True)
        return {"input_ids": enc["input_ids"], "labels": lab["input_ids"]}


def make_collate(pad_id: int):
    def collate(batch):
        n_in = max(len(b["input_ids"]) for b in batch)
        n_out = max(len(b["labels"]) for b in batch)
        ids = torch.full((len(batch), n_in), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n_in), dtype=torch.long)
        labels = torch.full((len(batch), n_out), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[i, :len(b["input_ids"])] = 1
            labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return collate


def sampling_weights(df: pd.DataFrame, power: float = 0.5) -> torch.Tensor:
    """
    Per-row weights so small (task, language) buckets are seen more often:
    bucket probability ∝ size**power (power=1 -> natural mix, 0 -> uniform).
    """
    keys = df["task"].astype(str) + "|" + df["language"].astype(str)
    sizes = keys.map(keys.value_counts())
    w = sizes.astype(float) ** power / sizes.astype(float)
    return torch.tensor((w / w.sum()).to_numpy(), dtype=torch.double)
