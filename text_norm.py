"""
Text utilities shared by training, inference and the converters.
Deliberately torch-free so converters and the server can import it cheaply.

    detect_script(text)          -> "native" | "latin" | "mixed"
    normalize_code_mixed(text)   -> (normalized_text, offset_map)
    to_original_span(s, e, offset_map, original) -> (orig_start, orig_end)

Normalization undoes common evasion tricks so the detector sees the word the
writer meant, while keeping a per-character offset map back to the ORIGINAL
text (rationale spans are predicted on normalized text but shown on the
original). Rules, in order:

  1. NFKC per grapheme cluster (fullwidth "ｈａｔｅ" -> "hate"); strip
     invisible characters (zero-width space, BOM, word joiner). ZWJ/ZWNJ are
     kept inside Indic/emoji text where they change spelling, and removed only
     between two Latin letters ("h‍ate").
  2. Curly quotes -> straight quotes.
  3. URLs, e-mail addresses and @mentions are PROTECTED from steps 4-6.
  4. De-spacing: a run of >= 3 isolated single Latin letters joined by the
     SAME separator collapses ("h a t e", "h.a.t.e", "h-a-t-e" -> "hate").
     Ordinary text with one-letter words ("i am a boy") is untouched because
     "am"/"boy" are not isolated letters. In a space-separated run of >= 4, a
     leading "a"/"I" is kept as its own word ("a b i t c h" -> "a bitch").
  5. Any non-digit character repeated > 2 times is cut to 2 ("haaaate" ->
     "haate", "!!!!" -> "!!"). Digits are never collapsed ("1000").
  6. Leetspeak reversal, per token, only when the token clearly spells a word:
     at least 2 letters, and not a "letters then digits" / "digits then short
     suffix" shape (covid19, mp3, top10, b4, 4k, 10am, 2nd are left alone).
     "@", "!" and "|" only count as leet when they sit between letters.
  7. Emojis are kept (they carry signal).

Every rule is length-preserving or deleting, so offset_map[i] is always the
index in the original string that normalized[i] came from.
"""

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Script detection
# --------------------------------------------------------------------------- #
_NATIVE_RANGES = (
    (0x0600, 0x06FF),  # Arabic (Urdu)
    (0x0750, 0x077F),  # Arabic supplement
    (0xFB50, 0xFDFF),  # Arabic presentation forms A
    (0xFE70, 0xFEFF),  # Arabic presentation forms B
    (0x0900, 0x097F),  # Devanagari
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)


def _is_native(c: str) -> bool:
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in _NATIVE_RANGES)


def detect_script(text: str) -> str:
    """
    'native' if text is predominantly Indic/Arabic script, 'latin' if
    predominantly Latin, 'mixed' if both are significant (>= 20% each).
    Text with no letters at all (emoji only, digits) counts as 'latin'.
    """
    native = sum(1 for c in text if _is_native(c))
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 0x0250)
    total = native + latin
    if total == 0:
        return "latin"
    ratio = native / total
    if ratio > 0.8:
        return "native"
    if ratio < 0.2:
        return "latin"
    return "mixed"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_INVISIBLE = frozenset({"​", "﻿", "⁠", "­"})
_JOINERS = frozenset({"‌", "‍"})
_QUOTES = {"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'"}

_PROTECT_RE = re.compile(
    r"(?:https?://|www\.)\S+"          # URLs
    r"|[\w.+-]+@[\w-]+\.[\w.]+"         # e-mail
    r"|(?<![\w@])@\w+",                 # @mentions
    re.IGNORECASE,
)
# >= 3 isolated ASCII letters separated by one repeated separator char
_SPACED_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]([ .\-_*])[A-Za-z](?:\1[A-Za-z])+(?![A-Za-z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9@$!|]+")
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s",
         "!": "i", "|": "l"}
_INTERIOR_ONLY = frozenset("@!|")
_NUMBERISH = (
    re.compile(r"^[A-Za-z]+\d+$"),          # covid19, mp3, top10, b4, win10
    re.compile(r"^\d+[A-Za-z]{1,3}$"),      # 4k, 10am, 2nd, 5g
)


def _is_latin_letter(c: str) -> bool:
    return c.isalpha() and ord(c) < 0x0250


def _clusters(text: str):
    """Yield (start, end) of grapheme-ish clusters: a base char + its combining marks."""
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.category(text[j]).startswith("M"):
            j += 1
        yield i, j
        i = j


def _deleet_token(tok: str) -> str | None:
    """Return the de-leeted token, or None if it should be left as is."""
    letters = sum(c.isalpha() for c in tok)
    if letters < 2 or not any(c in _LEET for c in tok):
        return None
    if any(p.match(tok) for p in _NUMBERISH):
        return None
    out = []
    for k, c in enumerate(tok):
        if c in _INTERIOR_ONLY:
            interior = 0 < k < len(tok) - 1 and tok[k - 1].isalnum() and tok[k + 1].isalnum()
            out.append(_LEET[c] if interior else c)
        else:
            out.append(_LEET.get(c, c))
    return "".join(out)


def normalize_code_mixed(text: str) -> tuple[str, list[int]]:
    """
    Normalize text for hate speech detection.
    Returns (normalized_text, offset_map) where offset_map[i] is the index in
    `text` that normalized_text[i] came from.
    """
    text = str(text)
    chars: list[str] = []
    offs: list[int] = []

    # 1. NFKC per cluster, invisible-char stripping
    for a, b in _clusters(text):
        cluster = text[a:b]
        if cluster in _INVISIBLE:
            continue
        if cluster in _JOINERS:
            prev_latin = a > 0 and _is_latin_letter(text[a - 1])
            next_latin = b < len(text) and _is_latin_letter(text[b])
            if prev_latin and next_latin:
                continue
        norm = unicodedata.normalize("NFKC", cluster)
        if len(norm) == len(cluster):
            idx = range(a, b)
        else:
            idx = [a] * len(norm)
        for c, o in zip(norm, idx):
            # 2. quotes
            chars.append(_QUOTES.get(c, c))
            offs.append(o)

    s = "".join(chars)

    # 3. protected spans
    protected = [False] * len(s)
    for m in _PROTECT_RE.finditer(s):
        for k in range(m.start(), m.end()):
            protected[k] = True

    # 4. de-spacing (delete the separators inside matched runs)
    keep = [True] * len(s)
    for m in _SPACED_RE.finditer(s):
        if any(protected[m.start():m.end()]):
            continue
        start = m.start()
        # "a b i t c h": the leading "a"/"I" is almost always a real word
        if m.group(1) == " " and s[start] in "aAI" and (m.end() - start + 1) // 2 >= 4:
            start += 2
        for k in range(start, m.end()):
            if not s[k].isalpha():
                keep[k] = False
    chars = [c for c, k in zip(chars, keep) if k]
    offs = [o for o, k in zip(offs, keep) if k]
    protected = [p for p, k in zip(protected, keep) if k]

    # 5. collapse runs of > 2 identical non-digit characters
    c2, o2, p2 = [], [], []
    for c, o, p in zip(chars, offs, protected):
        if (not p and not c.isdigit() and len(c2) >= 2
                and c2[-1] == c and c2[-2] == c and not p2[-1]):
            continue
        c2.append(c); o2.append(o); p2.append(p)
    chars, offs, protected = c2, o2, p2

    # 6. leetspeak reversal, token by token (length-preserving)
    s = "".join(chars)
    for m in _TOKEN_RE.finditer(s):
        if any(protected[m.start():m.end()]):
            continue
        new = _deleet_token(m.group(0))
        if new is not None:
            chars[m.start():m.end()] = list(new)

    return "".join(chars), offs


def to_original_span(norm_start: int, norm_end: int, offset_map: list[int],
                     original: str) -> tuple[int, int]:
    """
    Map a [start, end) span on normalized text back to the original text.
    The end is extended over trailing combining marks so Indic vowel signs
    are not cut off.
    """
    if not offset_map or norm_end <= norm_start:
        return 0, 0
    norm_start = max(0, min(norm_start, len(offset_map) - 1))
    norm_end = max(norm_start + 1, min(norm_end, len(offset_map)))
    start = offset_map[norm_start]
    end = offset_map[norm_end - 1] + 1
    while end < len(original) and unicodedata.category(original[end]).startswith("M"):
        end += 1
    return start, end
