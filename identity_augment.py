"""
Identity-term augmentation — the fix for the unseen-group failure.

THE PROBLEM THIS SOLVES
-----------------------
The trained detector scored "The Bronzites are a plague on every town they
enter" at p=0.28 (clean), while catching crude Tamil abuse at p=0.94. It had
learned `abuse == known slur/profanity present`, so a dehumanising claim about
an unfamiliar group registered as nothing at all. Adding implicit-hate data
teaches it that hate need not be profane; this teaches it that hate need not be
about a group it has already met.

HOW
---
During training only, rewrite the identity term in a sentence while keeping the
label. Two transforms, and the difference between them matters:

  1. SWAP (within category): "muslims are vermin" -> "hindus are vermin".
     Keeps target_label valid, because the replacement is drawn from the same
     target category. Teaches: the predicate carries the hate, not which
     particular known group it names.

  2. NONCE (invented group): "muslims are vermin" -> "Kelvarians are vermin".
     target_label is MASKED to -1, because an invented group genuinely has no
     category and asserting one would be a label we made up. Teaches the
     property actually needed: "<unknown group> are vermin" is still hate.

Nonce is the one that generalises to unseen groups; swap is what keeps the
target head trainable while it happens. Both are needed.

The nonce vocabulary deliberately EXCLUDES "Bronzites" — that word is the
motivating probe case, and training on it would make the probe measure
memorisation rather than generalisation.

Augmentation is applied to the training split only. Never to val/test: the
whole point is to measure generalisation to groups the model did not train on.
"""

import random
import re

import pandas as pd

# Identity terms grouped to match TARGET_CLASSES, so a within-category swap
# leaves target_label correct by construction. Multi-word phrases are matched
# whole; single words also match their trailing-s plural.
IDENTITY_TERMS: dict[str, list[str]] = {
    "religion": [
        "muslim", "muslims", "jew", "jews", "jewish people", "christian",
        "christians", "hindu", "hindus", "sikh", "sikhs", "buddhist",
        "buddhists", "catholic", "catholics", "atheist", "atheists",
    ],
    "gender": [
        "woman", "women", "man", "men", "girl", "girls", "trans people",
        "transgender people", "gay people", "lesbians", "queer people",
        "feminists", "bisexual people",
    ],
    "caste_ethnicity": [
        "black people", "white people", "asians", "asian people", "dalits",
        "brahmins", "latinos", "mexicans", "chinese people", "arabs",
        "africans", "native americans", "romani people",
    ],
    "disability": [
        "disabled people", "autistic people", "blind people", "deaf people",
        "people with disabilities", "mentally ill people",
    ],
    "nationality_migrant": [
        "immigrants", "refugees", "migrants", "foreigners", "expats",
        "asylum seekers",
    ],
}

# Invented group names: morphologically plausible, unmistakably not real, and
# absent from any pretraining corpus in this sense. "Bronzites" is excluded on
# purpose — see module docstring.
NONCE_GROUPS = [
    "Kelvarians", "Marnites", "Vossians", "Tarquins", "Zelanders",
    "Orvians", "Quillans", "Drennites", "Yarvans", "Pellorians",
    "Skarnites", "Ulvanians", "Threnites", "Corvashi", "Malduins",
]

# Longest-first so "jewish people" wins over "jew", and "black people" over a
# bare colour word. Word boundaries keep "asians" from firing inside "caucasians".
_ALL_TERMS = sorted(
    {t for terms in IDENTITY_TERMS.values() for t in terms},
    key=len,
    reverse=True,
)
_TERM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _ALL_TERMS) + r")\b", re.IGNORECASE
)

_CATEGORY_OF = {t: cat for cat, terms in IDENTITY_TERMS.items() for t in terms}


def contains_identity_term(text: str) -> bool:
    return _TERM_PATTERN.search(text) is not None


def _match_case(replacement: str, original: str) -> str:
    """Keep the original's capitalisation style so augmented text still reads naturally."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class IdentityAugmenter:
    """
    Rewrites identity mentions in training text.

    Args:
        swap_prob:  probability of a within-category swap (target_label kept)
        nonce_prob: probability of substituting an invented group (target masked)

    The two are mutually exclusive per example and together must be <= 1.0; the
    remaining probability mass leaves the text untouched. Defaults keep most
    examples original — augmentation is meant to broaden the distribution, not
    replace it.
    """

    def __init__(self, swap_prob: float = 0.25, nonce_prob: float = 0.15, seed: int = 42):
        if swap_prob + nonce_prob > 1.0:
            raise ValueError(
                f"swap_prob + nonce_prob must be <= 1.0, got {swap_prob + nonce_prob}"
            )
        self.swap_prob = swap_prob
        self.nonce_prob = nonce_prob
        self._rng = random.Random(seed)

    def __call__(self, text: str, target_label: int = -1) -> tuple[str, int]:
        """Returns (possibly rewritten text, possibly masked target_label)."""
        if not text or not contains_identity_term(text):
            return text, target_label

        draw = self._rng.random()
        if draw < self.nonce_prob:
            return self._to_nonce(text), -1
        if draw < self.nonce_prob + self.swap_prob:
            return self._swap_within_category(text), target_label
        return text, target_label

    def _to_nonce(self, text: str) -> str:
        nonce = self._rng.choice(NONCE_GROUPS)
        # One nonce per sentence: if a text names two groups, mapping both to the
        # same invented name would destroy the relation between them.
        return _TERM_PATTERN.sub(lambda m: _match_case(nonce, m.group(0)), text, count=1)

    def augment_always(self, text: str, target_label: int = -1) -> tuple[str, int]:
        """
        Like __call__ but never passes through — used by the expansion path
        below, where the caller has already decided this row gets augmented and
        a no-op would just produce a duplicate of the original.
        """
        if not text or not contains_identity_term(text):
            return text, target_label
        total = self.swap_prob + self.nonce_prob
        nonce_share = self.nonce_prob / total if total > 0 else 0.5
        if self._rng.random() < nonce_share:
            return self._to_nonce(text), -1
        return self._swap_within_category(text), target_label

    def _swap_within_category(self, text: str) -> str:
        def replace(m: re.Match) -> str:
            original = m.group(0)
            category = _CATEGORY_OF.get(original.lower())
            if category is None:
                return original
            options = [t for t in IDENTITY_TERMS[category] if t.lower() != original.lower()]
            if not options:
                return original
            return _match_case(self._rng.choice(options), original)

        return _TERM_PATTERN.sub(replace, text, count=1)


def expand_with_identity_augmentation(
    df: "pd.DataFrame",
    n_variants: int = 1,
    swap_prob: float = 0.6,
    nonce_prob: float = 0.4,
    seed: int = 42,
) -> "pd.DataFrame":
    """
    Return `df` plus augmented COPIES of its identity-bearing rows.

    Expansion rather than in-place rewriting, for two reasons:

      - train.py tokenises the whole frame once up front (that is where its 6.7x
        throughput came from). Stochastic per-epoch augmentation would force
        re-tokenisation every step and give that back.
      - Keeping the originals alongside the variants means augmentation only
        ever ADDS signal. Rewriting in place would trade real examples for
        synthetic ones, which is a worse deal at this corpus size.

    Rows carrying rationale spans are excluded: those spans are character
    offsets, and rewriting the text silently misaligns them.
    """
    if n_variants < 1:
        return df

    has_identity = df["text"].astype(str).map(contains_identity_term)
    if "rationale_spans" in df.columns:
        has_spans = df["rationale_spans"].astype(str).str.len() > 2
    else:
        has_spans = pd.Series(False, index=df.index)

    eligible = df[has_identity & ~has_spans]
    if eligible.empty:
        return df

    augmenter = IdentityAugmenter(swap_prob=swap_prob, nonce_prob=nonce_prob, seed=seed)
    variants = []
    for variant_i in range(n_variants):
        copy = eligible.copy()
        rewritten, targets = [], []
        for text, target in zip(copy["text"].astype(str), copy["target_label"].fillna(-1).astype(int)):
            new_text, new_target = augmenter.augment_always(text, int(target))
            rewritten.append(new_text)
            targets.append(new_target)
        copy["text"] = rewritten
        copy["target_label"] = targets
        copy["source"] = copy["source"].astype(str) + "_aug" if "source" in copy else "aug"
        variants.append(copy)

    out = pd.concat([df] + variants, ignore_index=True)
    return out
