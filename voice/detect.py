"""
Fast first-pass toxicity classifiers (pretrained, no training).

score(texts) -> max toxic probability across the configured classifiers.
Runs on XPU (Intel Arc) / CUDA if PyTorch sees one, else CPU.
"""

from __future__ import annotations

import logging

log = logging.getLogger("civitas.detect")


class Clf:
    def __init__(self, name: str, device, toxic_idx: int = 1, token: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from devices import half_dtype

        self.name, self.dev, self.k = name, device, toxic_idx
        self.tok = AutoTokenizer.from_pretrained(name, token=token or None)
        m = AutoModelForSequenceClassification.from_pretrained(name, token=token or None)
        self.m = m.to(device=device, dtype=half_dtype(device)).eval()
        self._torch = torch

    def __call__(self, texts: list[str], batch: int = 16) -> list[float]:
        torch = self._torch
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                b = self.tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                             max_length=256).to(self.dev)
                p = self.m(**b).logits.float().softmax(-1)[:, self.k]
                out += p.cpu().tolist()
        return out


class Detector:
    def __init__(self, names: list[str], device, token: str | None = None):
        self.clfs = []
        for n in names:
            try:
                self.clfs.append(Clf(n, device, token=token))
            except Exception as e:
                log.warning("could not load classifier %s: %s", n, e)
        if not self.clfs:
            raise RuntimeError("no toxicity classifier could be loaded")
        self.names = [c.name for c in self.clfs]

    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        cols = [c(texts) for c in self.clfs]
        return [max(v) for v in zip(*cols)]

    def score_detail(self, texts: list[str]) -> list[dict[str, float]]:
        cols = {c.name: c(texts) for c in self.clfs}
        return [{k: v[i] for k, v in cols.items()} for i in range(len(texts))]
