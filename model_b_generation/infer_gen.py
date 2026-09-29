"""
Raw Model B generation: prompts in, candidate strings out. No safety logic
here; generate.py owns the Model A gate and the fallbacks.

    python -m model_b_generation.infer_gen --adapter artifacts/model_b/adapter \
        --task rewrite --language en --text "..."
"""

import argparse
import sys

import torch

from config import (MODEL_B_BASE, MODEL_B_CKPT_DIR, MODEL_B_MAX_OUTPUT_LENGTH,
                    MODEL_B_REPETITION_PENALTY, MODEL_B_TEMPERATURE, MODEL_B_TOP_P)
from model_b_generation.model_gen import ModelB, load_for_inference
from model_b_generation.prompts import TASKS, build_prompt


@torch.no_grad()
def generate_candidates(mb: ModelB, prompts: list[str], k: int = 2,
                        max_new_tokens: int = MODEL_B_MAX_OUTPUT_LENGTH,
                        sample: bool = True, max_time: float | None = None) -> list[list[str]]:
    """
    k candidates per prompt. With sample=True the first candidate is greedy
    (the model's best guess) and the rest are nucleus samples for variety.
    max_time (seconds) caps wall time; HF stops decoding when it runs out.
    """
    enc = mb.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(mb.device)
    common = dict(max_new_tokens=max_new_tokens, repetition_penalty=MODEL_B_REPETITION_PENALTY)
    if max_time:
        common["max_time"] = max_time
    greedy = mb.model.generate(**enc, num_beams=1, do_sample=False, **common)
    out = [[s] for s in mb.tokenizer.batch_decode(greedy, skip_special_tokens=True)]
    if sample and k > 1:
        sampled = mb.model.generate(**enc, do_sample=True, temperature=MODEL_B_TEMPERATURE,
                                    top_p=MODEL_B_TOP_P, num_return_sequences=k - 1, **common)
        texts = mb.tokenizer.batch_decode(sampled, skip_special_tokens=True)
        for i in range(len(prompts)):
            out[i] += texts[i * (k - 1):(i + 1) * (k - 1)]
    return [[c.strip() for c in cands] for cands in out]


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate raw Model B candidates (no safety gate)")
    p.add_argument("--adapter", default=f"{MODEL_B_CKPT_DIR}/adapter")
    p.add_argument("--base", default=MODEL_B_BASE)
    p.add_argument("--task", choices=TASKS, default="rewrite")
    p.add_argument("--language", default="en")
    p.add_argument("--target", default="unknown")
    p.add_argument("--text", required=True)
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mb = load_for_inference(a.adapter, a.base)
    if mb.adapter_dir is None:
        print(f"WARNING: no adapter at {a.adapter}; using bare {a.base} (zero-shot)")
    prompt = build_prompt(a.task, a.language, a.text, a.target)
    print("prompt:", prompt)
    for i, c in enumerate(generate_candidates(mb, [prompt], a.k)[0], 1):
        print(f"{i}. {c}")


if __name__ == "__main__":
    main()
