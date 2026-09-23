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
    "hi": (sanscript.DEVANAGARI if _INDIC_TRANSLIT_AVAILABLE else None, "ITRANS"),
    "ta": (sanscript.TAMIL if _INDIC_TRANSLIT_AVAILABLE else None, "ITRANS"),
    "te": (sanscript.TELUGU if _INDIC_TRANSLIT_AVAILABLE else None, "ITRANS"),
    "ml": (sanscript.MALAYALAM if _INDIC_TRANSLIT_AVAILABLE else None, "ITRANS"),
}

# Simple Urdu ↔ Roman-Urdu character map (APPROXIMATE)
_URDU_TO_ROMAN = {
    'آ': 'aa', 'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'T',
    'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
    'ڈ': 'D', 'ذ': 'z', 'ر': 'r', 'ڑ': 'R', 'ز': 'z', 'ژ': 'zh',
    'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
    'ع': "'", 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
    'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'w', 'ہ': 'h',
    'ی': 'y', 'ے': 'e', 'ئ': 'y',
}


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
        self.langs = langs or [l for l in SUPPORTED_LANGUAGES if l != "en"]

    def augment(self, text: str, language: str, script: str) -> AugmentResult:
        """Randomly transliterate text between native and latin scripts."""
        if language not in self.langs or random.random() > self.p_augment:
            return AugmentResult(text=text, script=script, spans_valid=True)

        if language == "ur_roman":
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
            new_text = self._transliterate(text, native_scheme, roman_scheme)
            new_script = "latin"
        elif script == "latin":
            new_text = self._transliterate(text, roman_scheme, native_scheme)
            new_script = "native"
        else:  # mixed — don't touch
            return AugmentResult(text=text, script=script, spans_valid=True)

        spans_valid = len(new_text) == len(text)
        return AugmentResult(text=new_text, script=new_script, spans_valid=spans_valid)

    def _augment_urdu(self, text: str, script: str) -> AugmentResult:
        """APPROXIMATE Urdu script ↔ Roman-Urdu conversion."""
        if script == "native":
            new_text = "".join(_URDU_TO_ROMAN.get(c, c) for c in text)
            new_script = "latin"
        else:
            return AugmentResult(text=text, script=script, spans_valid=True)  # roman→urdu not implemented
        spans_valid = len(new_text) == len(text)
        return AugmentResult(text=new_text, script=new_script, spans_valid=spans_valid)
