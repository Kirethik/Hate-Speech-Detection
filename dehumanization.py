"""
Dehumanisation-metaphor detection.

WHY THIS EXISTS
---------------
The motivating failure — "The Bronzites are a plague on every town they enter"
(scored p=0.28, clean) — is a textbook DISEASE-metaphor dehumanisation. The
mechanism is well described in the literature (Mendelsohn et al., 2020,
"A Framework for the Computational Linguistic Analysis of Dehumanization"):
hate frequently works by casting a group as non-human — vermin, disease,
filth, animals, cargo — rather than by insulting it with profanity.

That mechanism is lexically narrow and therefore catchable with a lexicon, which
buys two things a neural score cannot:

  1. INTERPRETABILITY. "flagged: disease frame ('plague')" is an explanation a
     moderator can audit and contest. A bare p=0.87 is not.
  2. A FLOOR. The lexicon fires regardless of who the target is, including
     groups absent from training. It is exactly the generalisation the encoder
     lacks, which is why it complements rather than duplicates the model.

WHAT IT IS NOT
--------------
A lexicon is not a classifier. "The flu is a plague on this city" and "this
game is cancer" both contain the trigger and neither is hate speech: the
mechanism requires a HUMAN GROUP in subject position. So `score_text` reports
matched frames, and `is_dehumanizing_claim` additionally requires that the
metaphor be predicated of a plural human referent. Precision is checked against
DynaHate's `dehumanization`-typed rows in validate_dehumanization.py — do not
treat the lexicon as ground truth without reading that output.
"""

import re

# Metaphor frames, after Mendelsohn et al. (2020). Each maps a frame name to the
# terms that instantiate it. Deliberately conservative: only terms whose
# dehumanising sense dominates their ordinary sense in social-media text.
DEHUMANIZATION_FRAMES: dict[str, list[str]] = {
    "vermin": [
        "vermin", "rats", "rat", "cockroaches", "cockroach", "roaches",
        "parasites", "parasite", "lice", "maggots", "maggot", "leeches",
        "infestation", "infest", "infesting", "swarm", "swarming", "breed",
        "breeding like",
    ],
    "disease": [
        "plague", "cancer", "virus", "disease", "infection", "infected",
        "contagion", "epidemic", "tumour", "tumor", "pestilence", "blight",
    ],
    "animal": [
        "animals", "animal", "beasts", "beast", "apes", "ape", "monkeys",
        "monkey", "dogs", "swine", "pigs", "cattle", "savages", "savage",
        "feral", "subhuman", "subhumans", "livestock",
    ],
    "filth": [
        "filth", "filthy", "scum", "trash", "garbage", "waste", "sewage",
        "dirt", "stain", "impure", "unclean", "contaminate", "contaminating",
        "pollute", "polluting",
    ],
    "object": [
        "cargo", "commodity", "merchandise", "property", "stock", "inventory",
        "disposable", "expendable",
    ],
}

_FRAME_OF = {
    term: frame for frame, terms in DEHUMANIZATION_FRAMES.items() for term in terms
}
_ALL_TERMS = sorted(_FRAME_OF, key=len, reverse=True)
_TERM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _ALL_TERMS) + r")\b", re.IGNORECASE
)

# A dehumanising claim needs a human group as its subject. Bare plural nouns are
# the usual carrier ("<group> are vermin"), so this looks for a plural/collective
# referent rather than trying to enumerate every group name — which is the whole
# point, since the failing case names a group that does not exist.
_GROUP_SUBJECT = re.compile(
    r"\b(?:the\s+|these\s+|those\s+|all\s+|these\s+people\s+)?"
    r"([A-Za-z][A-Za-z'-]{2,})s\b\s+(?:are|is|were|was|have|has|keep|keeps|remain)\b",
    re.IGNORECASE,
)
_PRONOUN_SUBJECT = re.compile(
    r"\b(?:they|them|their|those people|these people)\b", re.IGNORECASE
)


def find_frames(text: str) -> dict[str, list[str]]:
    """Return {frame_name: [matched terms]} for every dehumanisation frame present."""
    found: dict[str, list[str]] = {}
    for match in _TERM_PATTERN.finditer(str(text)):
        term = match.group(0).lower()
        frame = _FRAME_OF.get(term)
        if frame is None:
            continue
        found.setdefault(frame, [])
        if term not in found[frame]:
            found[frame].append(term)
    return found


def score_text(text: str) -> dict:
    """
    Lexicon reading of one string.

    Returns matched frames, the triggering terms, and whether the metaphor is
    actually predicated of a group (as opposed to merely appearing in the text).
    """
    frames = find_frames(text)
    has_group_subject = bool(
        _GROUP_SUBJECT.search(str(text)) or _PRONOUN_SUBJECT.search(str(text))
    )
    return {
        "frames": sorted(frames),
        "terms": sorted({t for terms in frames.values() for t in terms}),
        "n_frames": len(frames),
        "has_group_subject": has_group_subject,
        "is_dehumanizing_claim": bool(frames) and has_group_subject,
    }


def is_dehumanizing_claim(text: str) -> bool:
    """
    True when a dehumanisation metaphor is predicated of a plural human referent.

    Requiring the subject is what separates "the Bronzites are a plague on every
    town" (hate) from "the flu is a plague this winter" (not) — the lexicon alone
    cannot tell those apart, and reporting it as if it could would be the same
    keyword-matching mistake the model already makes.
    """
    return score_text(text)["is_dehumanizing_claim"]


def explain(text: str) -> str:
    """One-line human-readable reason, or '' when nothing fired."""
    s = score_text(text)
    if not s["frames"]:
        return ""
    frames = ", ".join(f"{f} frame" for f in s["frames"])
    terms = ", ".join(f"'{t}'" for t in s["terms"])
    if not s["has_group_subject"]:
        return f"{frames} present ({terms}) but not predicated of a group"
    return f"dehumanising {frames} ({terms}) applied to a group"
