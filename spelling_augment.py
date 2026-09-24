"""
Typo augmentation for Model A training.

normalize_code_mixed() already undoes leetspeak, spaced-out letters and
elongation at inference time, so training on those would teach nothing new.
What normalization CANNOT undo are genuine misspellings, and HateCheck shows
the old model collapsing on exactly those:

    spell_space_add_h  0.12   ("I hate wo men")
    spell_char_del_h   0.46   ("I hate wmen")
    spell_char_swap_h  0.50   ("I hate wmoen")

So each augmented copy perturbs one or two words with one of:
    space insert | char delete | adjacent swap | char duplicate

Rows of BOTH classes are perturbed, so the model cannot learn "typo => hate".
Rows with rationale spans are skipped (edits shift character offsets).
"""

import random

import pandas as pd

OPS = ("space", "delete", "swap", "dup")


def perturb_word(word: str, rng: random.Random, op: str | None = None) -> str:
    """Apply one typo to a word of >= 4 characters (shorter words are returned as is)."""
    if len(word) < 4:
        return word
    op = op or rng.choice(OPS)
    i = rng.randrange(1, len(word) - 1)  # never the first letter: keeps the word recognisable
    if op == "space":
        return word[:i] + " " + word[i:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]  # dup


def perturb_text(text: str, rng: random.Random) -> str:
    """Perturb 1 word (2 for texts of 12+ words). Only alphabetic words qualify."""
    words = text.split(" ")
    candidates = [k for k, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not candidates:
        return text
    n = 2 if len(words) >= 12 and len(candidates) >= 2 else 1
    for k in rng.sample(candidates, n):
        words[k] = perturb_word(words[k], rng)
    return " ".join(words)


def expand_with_spelling_augmentation(df: pd.DataFrame, frac: float = 0.15,
                                      seed: int = 42) -> pd.DataFrame:
    """Return `df` plus perturbed copies of a random `frac` of its eligible rows."""
    if frac <= 0:
        return df
    has_spans = (df["rationale_spans"].astype(str).str.len() > 2
                 if "rationale_spans" in df.columns else pd.Series(False, index=df.index))
    eligible = df[~has_spans]
    if eligible.empty:
        return df
    picked = eligible.sample(frac=min(frac, 1.0), random_state=seed)
    rng = random.Random(seed)
    copy = picked.copy()
    copy["text"] = [perturb_text(str(t), rng) for t in copy["text"]]
    copy = copy[copy["text"].values != picked["text"].astype(str).values]
    copy["source"] = copy["source"].astype(str) + "_typo"
    return pd.concat([df, copy], ignore_index=True)
