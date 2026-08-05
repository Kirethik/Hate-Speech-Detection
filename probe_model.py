"""
Behavioural probe for a trained Model A checkpoint.

A macro-F1 on a held-out split tells you the model fits the data distribution.
It does not tell you whether the model learned *abuse* or learned a shortcut
that happens to correlate with abuse. This script goes after the two failure
modes that a good F1 will happily hide:

  1. IDENTITY-MENTION FALSE POSITIVES. Abuse corpora are full of identity terms
     inside abusive sentences, so a lazy model learns "mentions Muslims/women/
     Dalits -> abusive" and then flags people describing their own identity.
     That model scores well and is unusable for moderation.

  2. OBFUSCATION BRITTLENESS. Real abusers evade filters with leetspeak, spacing
     and letter elongation. `normalize_code_mixed()` is still a stub, so the
     model has no de-obfuscation front-end. This measures the resulting FLIP
     RATE: of the abusive examples the model correctly catches, how many does it
     miss once perturbed? That is the README's RQ1 metric, and it is a property
     of the model that the test-set F1 cannot show you.

Usage:
    python probe_model.py [--checkpoint checkpoints/model_a/best_model.pt]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import SEVERITY_CLASSES, TARGET_CLASSES, normalize_code_mixed
from model import CivitasDetector
from transformers import XLMRobertaTokenizerFast

# Crafted English cases. Kept to English because the point is to assert what the
# CORRECT answer is, and inventing Tamil/Telugu test sentences I cannot verify
# would just produce confident nonsense. Indic behaviour is probed against real
# labelled test rows further down instead.
SANITY_CASES = [
    # (text, expected_abusive, why this case exists)
    ("i really enjoyed this movie, the soundtrack was beautiful", False, "plain benign"),
    ("thanks for helping me out yesterday, you are a good friend", False, "plain benign"),
    ("i am a muslim woman and i am proud of my heritage", False, "IDENTITY MENTION, benign"),
    ("as a jewish immigrant i want to share my family's story", False, "IDENTITY MENTION, benign"),
    ("my sister is disabled and she is the strongest person i know", False, "IDENTITY MENTION, benign"),
    ("the conference on women in engineering was inspiring", False, "IDENTITY MENTION, benign"),
    ("black lives matter and we will keep marching", False, "IDENTITY MENTION, activism"),
    ("all muslims are terrorists and should be deported", True, "religion-targeted hate"),
    ("women are too stupid to lead a company", True, "gender-targeted hate"),
    ("you are a complete idiot and i hate you", True, "personal abuse"),
    ("these immigrants are vermin destroying our country", True, "dehumanising, migrant"),
    ("go back to your country you filthy animal", True, "xenophobic abuse"),
]

LEET = str.maketrans({"a": "@", "i": "1", "e": "3", "o": "0", "s": "$"})


def perturb(text: str, mode: str) -> str:
    """Evasion transforms that preserve human readability but break tokenisation."""
    if mode == "leet":
        return text.translate(LEET)
    if mode == "spaced":
        # space out the letters of the longest word (typical filter evasion)
        words = text.split()
        if not words:
            return text
        i = max(range(len(words)), key=lambda j: len(words[j]))
        words[i] = " ".join(words[i])
        return " ".join(words)
    if mode == "elongate":
        return re.sub(r"([aeiou])", r"\1\1\1", text, count=3)
    if mode == "punct":
        return re.sub(r"(\w)(\w)", r"\1.\2", text, count=5)
    raise ValueError(mode)


@torch.no_grad()
def score_texts(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    probs, severities, targets = [], [], []
    for i in range(0, len(texts), batch_size):
        chunk = [normalize_code_mixed(str(t)) for t in texts[i:i + batch_size]]
        enc = tokenizer(chunk, truncation=True, max_length=128,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs.append(torch.softmax(out["hate_logits"].float(), -1)[:, 1].cpu().numpy())
        severities.append(out["severity_logits"].float().argmax(-1).cpu().numpy())
        targets.append(out["target_logits"].float().argmax(-1).cpu().numpy())
    return (np.concatenate(probs), np.concatenate(severities), np.concatenate(targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model_a/best_model.pt")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--n_per_language", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    threshold = ckpt["threshold"]

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(ckpt["args"]["encoder_name"])
    model = CivitasDetector(
        encoder_name=ckpt["args"]["encoder_name"],
        num_target_classes=len(TARGET_CLASSES),
        num_severity_classes=len(SEVERITY_CLASSES),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded {args.checkpoint}  (decision threshold {threshold:.2f}, fitted on val)\n")

    # ---- 1. crafted sanity cases -------------------------------------------
    print("=" * 78)
    print("1. SANITY CASES — does it separate abuse from mere identity mention?")
    print("=" * 78)
    texts = [c[0] for c in SANITY_CASES]
    probs, sevs, tgts = score_texts(model, tokenizer, texts, device)
    n_wrong, id_wrong = 0, 0
    for (text, expected, why), p, s, t in zip(SANITY_CASES, probs, sevs, tgts):
        pred = bool(p >= threshold)
        ok = pred == expected
        n_wrong += not ok
        if "IDENTITY MENTION" in why and not ok:
            id_wrong += 1
        print(f"  [{'ok ' if ok else 'MISS'}] p(abuse)={p:.3f} -> {'ABUSIVE' if pred else 'clean':<8} "
              f"| exp {'ABUSIVE' if expected else 'clean':<8} | sev={SEVERITY_CLASSES[s]:<19} "
              f"tgt={TARGET_CLASSES[t]:<18} | {why}")
        if not ok:
            print(f"         text: {text!r}")
    n_id = sum('IDENTITY MENTION' in c[2] for c in SANITY_CASES)
    print(f"\n  {len(SANITY_CASES) - n_wrong}/{len(SANITY_CASES)} correct   "
          f"| identity-mention false positives: {id_wrong}/{n_id}")

    # ---- 2. obfuscation flip rate ------------------------------------------
    print("\n" + "=" * 78)
    print("2. OBFUSCATION FLIP RATE — of abuse it CATCHES, how much does evasion hide?")
    print("=" * 78)
    test = pd.read_csv(args.test_csv)
    abusive = test[test["hate_label"] == 1]
    sample = abusive.groupby("language", group_keys=False).head(args.n_per_language)
    base_p, _, _ = score_texts(model, tokenizer, sample["text"].tolist(), device)
    caught = base_p >= threshold
    print(f"  sampled {len(sample)} truly-abusive test rows; model catches "
          f"{caught.sum()} ({caught.mean():.1%}) at threshold {threshold:.2f}")
    caught_texts = sample["text"].astype(str).to_numpy()[caught]
    caught_langs = sample["language"].astype(str).to_numpy()[caught]

    # Flip rate MUST be conditioned on the text actually changing. Leetspeak only
    # substitutes Latin a/e/i/o/s, so it is close to a no-op on Devanagari /
    # Kannada / Malayalam / Telugu — scoring those unchanged rows as "did not
    # flip" would report robustness the model has not demonstrated.
    base_caught = base_p[caught]
    rows = []
    for mode in ("leet", "spaced", "elongate", "punct"):
        pert = np.array([perturb(t, mode) for t in caught_texts], dtype=object)
        changed = pert != caught_texts
        p, _, _ = score_texts(model, tokenizer, list(pert), device)
        flipped = (p < threshold) & changed
        denom = int(changed.sum())
        rate = float(flipped.sum() / denom) if denom else float("nan")
        rows.append({"perturbation": mode, "flip_rate_of_perturbed": rate,
                     "n_flipped": int(flipped.sum()), "n_actually_perturbed": denom,
                     "n_candidates": int(len(pert)),
                     "mean_prob_drop": float((base_caught[changed] - p[changed]).mean()) if denom else None})
        print(f"  {mode:<9} flip rate {rate:6.1%} of the {denom}/{len(pert)} rows the "
              f"transform actually altered  ({flipped.sum()} now evade)  "
              f"mean p drop {(base_caught[changed] - p[changed]).mean():+.3f}")

    print("\n  leetspeak flip rate by language (only rows the transform altered):")
    pert = np.array([perturb(t, "leet") for t in caught_texts], dtype=object)
    changed = pert != caught_texts
    p, _, _ = score_texts(model, tokenizer, list(pert), device)
    flipped = (p < threshold) & changed
    by_lang = {}
    for lang in sorted(set(caught_langs)):
        m = (caught_langs == lang) & changed
        if m.sum() < 10:
            print(f"    {lang:<9}    n/a  (only {int(m.sum())} rows altered — "
                  f"transform does not apply to this script)")
            by_lang[lang] = None
            continue
        by_lang[lang] = float(flipped[m].sum() / m.sum())
        print(f"    {lang:<9} {flipped[m].mean():6.1%}  ({int(flipped[m].sum())}/{int(m.sum())})")

    out = Path("checkpoints/model_a/robustness.json")
    out.write_text(json.dumps({
        "threshold": threshold,
        "sanity_correct": int(len(SANITY_CASES) - n_wrong),
        "sanity_total": len(SANITY_CASES),
        "identity_false_positives": int(id_wrong),
        "identity_cases": int(n_id),
        "catch_rate_on_sampled_abuse": float(caught.mean()),
        "perturbations": rows,
        "leet_flip_rate_by_language": by_lang,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
