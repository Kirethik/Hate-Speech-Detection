"""
Training-time script augmentation for speech robustness.

ASR outputs native script (Devanagari, Tamil-script, etc.) but much of our
training data is romanized or code-mixed. Without augmentation the model will
score native-script ASR output poorly even when the speech is in-distribution.

Usage:
    aug = ScriptAugmenter(p_augment=0.3, langs=["hi", "ta", "te", "ml"])
    new_text, new_script, spans_valid = aug.augment(text, language="hi", script="native")

Note: The Urdu ↔ Roman-Urdu mapper is APPROXIMATE (rule-based character substitution).
It does not handle all Urdu diacritics or context-dependent letter forms. It is useful
for training robustness but should not be used as a production transliterator.

Pluggable backend: the `_transliterate` method uses `indic-transliteration` by default.
To plug in AI4Bharat IndicXlit, subclass ScriptAugmenter and override `_transliterate`.
"""

import random
import unicodedata
import warnings
import re
from dataclasses import dataclass
from typing import Optional

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    _INDIC_TRANSLIT_AVAILABLE = True
except ImportError:
    _INDIC_TRANSLIT_AVAILABLE = False

from config import SUPPORTED_LANGUAGES

# Maps language code → (native sanscript scheme, roman scheme)
_SCHEME_MAP = {
    "hi": (sanscript.DEVANAGARI if _INDIC_TRANSLIT_AVAILABLE else None, "itrans"),
    "ta": (sanscript.TAMIL if _INDIC_TRANSLIT_AVAILABLE else None, "itrans"),
    "te": (sanscript.TELUGU if _INDIC_TRANSLIT_AVAILABLE else None, "itrans"),
    "ml": (sanscript.MALAYALAM if _INDIC_TRANSLIT_AVAILABLE else None, "itrans"),
    "kn": (sanscript.KANNADA if _INDIC_TRANSLIT_AVAILABLE else None, "itrans"),
}

# Simple Urdu -> Roman-Urdu character map (APPROXIMATE: short vowels are not
# written in Urdu script, so "تم" comes out "tm", not "tum"). Lowercase output
# matches how Roman-Urdu is typed in chat. و / ی are consonants word-initially
# and vowels elsewhere; see _urdu_to_roman().
_URDU_TO_ROMAN = {
    'آ': 'aa', 'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't',
    'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
    'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh',
    'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
    'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'ك': 'k', 'گ': 'g',
    'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'ہ': 'h', 'ه': 'h', 'ۃ': 'h', 'ة': 'h',
    'ھ': 'h', 'ے': 'e', 'ۓ': 'e', 'ئ': 'y', 'ء': '', 'ؤ': 'o',
    'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ّ': '', 'ْ': '', 'ٰ': 'a', 'ً': 'an',
    '۔': '.', '،': ',', '؟': '?', '٪': '%',
}
_URDU_SEMIVOWELS = {'و': ('w', 'o'), 'ی': ('y', 'i'), 'ي': ('y', 'i')}


def _urdu_to_roman(text: str) -> str:
    out = []
    for i, c in enumerate(text):
        if c in _URDU_SEMIVOWELS:
            word_initial = i == 0 or not text[i - 1].isalpha()
            out.append(_URDU_SEMIVOWELS[c][0 if word_initial else 1])
        else:
            out.append(_URDU_TO_ROMAN.get(c, c))
    return "".join(out)


@dataclass
class AugmentResult:
    text: str
    script: str
    spans_valid: bool  # False if char lengths changed; caller should mask rationale


class ScriptAugmenter:
    def __init__(self, p_augment: float = 0.3, langs: Optional[list] = None):
        """
        p_augment: probability of augmenting any given sample.
        langs: list of language codes to augment (default: all non-English).
        """
        self.p_augment = p_augment
        if p_augment > 0 and not _INDIC_TRANSLIT_AVAILABLE:
            warnings.warn("indic-transliteration is not installed: Indic script "
                          "augmentation is DISABLED (pip install indic-transliteration)")
        self.langs = langs or [l for l in SUPPORTED_LANGUAGES if l != "en"]

    def augment(self, text: str, language: str, script: str) -> AugmentResult:
        """Randomly transliterate text between native and latin scripts."""
        if language not in self.langs or random.random() > self.p_augment:
            return AugmentResult(text=text, script=script, spans_valid=True)

        if language in ("ur", "ur_roman"):
            return self._augment_urdu(text, script)
        return self._augment_indic(text, language, script)

    def _transliterate(self, text: str, from_scheme, to_scheme: str) -> str:
        """Transliterate using indic-transliteration. Override to use IndicXlit."""
        if not _INDIC_TRANSLIT_AVAILABLE:
            return text  # no-op if package missing
        try:
            return transliterate(text, from_scheme, to_scheme)
        except Exception:
            return text  # never crash training on bad transliteration

    def _augment_indic(self, text: str, language: str, script: str) -> AugmentResult:
        if language not in _SCHEME_MAP:
            return AugmentResult(text=text, script=script, spans_valid=True)
        native_scheme, roman_scheme = _SCHEME_MAP[language]
        if native_scheme is None:
            return AugmentResult(text=text, script=script, spans_valid=True)

        if script == "native":
            # ISO 15919 covers every Indic letter (incl. Tamil ன/ற); stripping its
            # diacritics and lowercasing gives chat-style romanization
            # ("மனிதன்" -> "manidhan"), which is what the romanized data looks like.
            new_text = self._transliterate(text, native_scheme, "iso")
            new_text = "".join(c for c in unicodedata.normalize("NFKD", new_text)
                               if not unicodedata.category(c).startswith("M")).lower()
            new_script = "latin"
        elif script == "latin":
            new_text = self._transliterate(text.lower(), roman_scheme, native_scheme)
            if language == "hi":
                # Hindi drops the word-final virama that ITRANS input produces ("तुम्" -> "तुम")
                new_text = re.sub(r"्(?=\W|$)", "", new_text)
            new_script = "native"
        else:  # mixed — don't touch
            return AugmentResult(text=text, script=script, spans_valid=True)

        spans_valid = len(new_text) == len(text)
        return AugmentResult(text=new_text, script=new_script, spans_valid=spans_valid)

    def _augment_urdu(self, text: str, script: str) -> AugmentResult:
        """APPROXIMATE Urdu script ↔ Roman-Urdu conversion."""
        if script == "native":
            new_text = _urdu_to_roman(text)
            new_script = "latin"
        else:
            return AugmentResult(text=text, script=script, spans_valid=True)  # roman→urdu not implemented
        spans_valid = len(new_text) == len(text)
        return AugmentResult(text=new_text, script=new_script, spans_valid=spans_valid)


def expand_with_script_augmentation(df, frac: float = 0.3, seed: int = 42):
    """
    Return `df` plus transliterated COPIES of a random `frac` of its non-English
    rows (native -> latin or latin -> native). This is what makes a model
    trained mostly on romanized text usable on native-script ASR output.

    Copies are added rather than rewriting in place, matching
    identity_augment.expand_with_identity_augmentation. Rows with rationale
    spans are skipped (transliteration changes character offsets), as are
    rows whose transliteration is a no-op.
    """
    import pandas as pd
    from text_norm import detect_script

    if frac <= 0 or not _INDIC_TRANSLIT_AVAILABLE:
        if frac > 0:
            warnings.warn("script augmentation skipped: indic-transliteration not installed")
        return df
    has_spans = (df["rationale_spans"].astype(str).str.len() > 2
                 if "rationale_spans" in df.columns else pd.Series(False, index=df.index))
    eligible = df[(df["language"] != "en") & ~has_spans]
    if eligible.empty:
        return df
    picked = eligible.sample(frac=min(frac, 1.0), random_state=seed)
    random.seed(seed)
    aug = ScriptAugmenter(p_augment=1.0)
    texts, scripts = [], []
    for text, lang in zip(picked["text"].astype(str), picked["language"]):
        res = aug.augment(text, lang, detect_script(text))
        texts.append(res.text)
        scripts.append(res.script)
    copy = picked.copy()
    copy["text"] = texts
    copy["script"] = scripts
    copy = copy[copy["text"].values != picked["text"].astype(str).values]
    copy["source"] = copy["source"].astype(str) + "_translit"
    return pd.concat([df, copy], ignore_index=True)
