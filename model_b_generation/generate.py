"""
The single entry point for alternate speech. Model B proposes, Model A
disposes: every candidate is re-scored by Model A and anything that still
looks hateful, echoes the input, or reuses the flagged words is dropped.

Two callers:
  generate_suggestions(...)  text mode: Rewrite + Respond lists for the UI
  rewrite_for_speech(...)    live call: ONE safe line to speak to the listener,
                             or None (the caller withholds the utterance)

`score_fn(texts) -> [hate_prob]` is Model A (pipeline.registry builds it).
Without a score_fn nothing generated can be proven safe, so only the
hand-written templates are returned.
"""

import re
import time
from typing import Callable

from config import MODEL_B_CANDIDATES, MODEL_B_SAFETY_THRESHOLD
from model_b_generation import gen_metrics
from model_b_generation.prompts import build_prompt

ScoreFn = Callable[[list[str]], list[float]]

# Last-resort Respond lines. Rewrite has no template: a canned sentence would
# put words in the speaker's mouth, so rewrite falls back to masking instead.
# te / ml / ur_roman written by Claude: have a native speaker check them.
FALLBACK_RESPONSES = {
    "en": ["Every person deserves to be treated with dignity. Let's keep this respectful.",
           "We can disagree without targeting a whole group of people."],
    "hi": ["हर इंसान सम्मान का हकदार है। आइए, बातचीत सम्मान से करें।",
           "हम असहमत हो सकते हैं, पर किसी पूरे समुदाय को निशाना बनाना ठीक नहीं।"],
    "ur_roman": ["Har insaan izzat ka haqdaar hai. Aaiye baat izzat se karein.",
                 "Hum ikhtilaf kar sakte hain, magar kisi poori community ko nishana banana theek nahi."],
    "ta": ["ஒவ்வொரு மனிதனும் மரியாதைக்கு உரியவர். மரியாதையுடன் பேசுவோம்.",
           "கருத்து வேறுபாடு இருக்கலாம், ஆனால் ஒரு முழு சமூகத்தையும் குறிவைக்க வேண்டாம்."],
    "te": ["ప్రతి వ్యక్తి గౌరవానికి అర్హుడు. మనం గౌరవంగా మాట్లాడుకుందాం.",
           "అభిప్రాయ భేదం ఉండొచ్చు, కానీ ఒక సమూహం మొత్తాన్ని లక్ష్యంగా చేసుకోవద్దు."],
    "ml": ["ഓരോ വ്യക്തിയും ബഹുമാനം അർഹിക്കുന്നു. നമുക്ക് ബഹുമാനത്തോടെ സംസാരിക്കാം.",
           "അഭിപ്രായ വ്യത്യാസം ആകാം, പക്ഷേ ഒരു സമൂഹത്തെ മുഴുവൻ ലക്ഷ്യമിടരുത്."],
}

_WORD = re.compile(r"\w+", re.UNICODE)
MIN_MASKED_WORDS = 3          # fewer words left after masking -> nothing worth saying
MIN_MASKED_KEEP_RATIO = 0.4   # masking removed most of the sentence -> withhold


def _words(s: str) -> set[str]:
    return {w.casefold() for w in _WORD.findall(s) if len(w) >= 3}


def reuses_flagged(candidate: str, spans: list[dict] | None) -> bool:
    flagged = set().union(*(_words(s.get("text", "")) for s in spans)) if spans else set()
    return bool(flagged & _words(candidate))


def mask_spans(text: str, spans: list[dict] | None) -> str:
    """Remove the flagged character spans and tidy the leftover whitespace/punctuation."""
    if not spans:
        return text
    keep, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start_char"]):
        keep.append(text[pos:s["start_char"]])
        pos = max(pos, s["end_char"])
    keep.append(text[pos:])
    out = re.sub(r"\s+", " ", " ".join(keep))
    out = re.sub(r"\s+([,.!?;:।])", r"\1", out)
    out = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", out)
    return out.strip(" ,;:")


def _safe(cands: list[str], source: str, spans, score_fn: ScoreFn | None,
          threshold: float) -> list[tuple[str, float]]:
    seen, pool = set(), []
    for c in cands:
        c = c.strip()
        key = c.casefold()
        if not c or key in seen or gen_metrics.is_copy(c, source) or reuses_flagged(c, spans):
            continue
        seen.add(key)
        pool.append(c)
    if not pool or score_fn is None:
        return []
    probs = score_fn(pool)
    return sorted([(c, p) for c, p in zip(pool, probs) if p < threshold], key=lambda x: x[1])


def _templates(language: str, n: int) -> list[dict]:
    lines = FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["en"])
    lang = language if language in FALLBACK_RESPONSES else "en"
    return [{"text": t, "language": lang, "safety_check_hate_prob": None, "template": True}
            for t in lines[:n]]


def generate_suggestions(text: str, language: str, target: str | None, model_b=None,
                         score_fn: ScoreFn | None = None,
                         threshold: float = MODEL_B_SAFETY_THRESHOLD,
                         spans: list[dict] | None = None, k: int = MODEL_B_CANDIDATES,
                         n_return: int = 2, tasks=("rewrite", "respond")) -> dict:
    """Text mode. Returns {"rewrite": [...], "respond": [...], "fallback": bool}."""
    out = {"rewrite": [], "respond": [], "fallback": False}
    if model_b is not None and score_fn is not None:
        from model_b_generation.infer_gen import generate_candidates
        prompts = [build_prompt(t, language, text, target) for t in tasks]
        for task, cands in zip(tasks, generate_candidates(model_b, prompts, k)):
            out[task] = [{"text": c, "language": language, "safety_check_hate_prob": round(p, 4)}
                         for c, p in _safe(cands, text, spans, score_fn, threshold)[:n_return]]
    if "rewrite" in tasks and not out["rewrite"]:
        masked = masked_rewrite(text, spans, score_fn, threshold)
        if masked:
            out["rewrite"] = [{"text": masked[0], "language": language,
                               "safety_check_hate_prob": masked[1], "masked": True}]
            out["fallback"] = True
    if "respond" in tasks and not out["respond"]:
        out["respond"] = _templates(language, n_return)
        out["fallback"] = True
    return out


def masked_rewrite(text: str, spans, score_fn: ScoreFn | None,
                   threshold: float) -> tuple[str, float | None] | None:
    """The flagged words removed, if enough of the sentence survives and Model A clears it."""
    masked = mask_spans(text, spans)
    n_orig, n_left = len(_WORD.findall(text)), len(_WORD.findall(masked))
    if not spans or n_left < MIN_MASKED_WORDS or n_left < MIN_MASKED_KEEP_RATIO * n_orig:
        return None
    if score_fn is None:
        return None  # cannot prove the leftover is safe
    p = score_fn([masked])[0]
    return (masked, round(p, 4)) if p < threshold else None


def rewrite_for_speech(text: str, language: str, target: str | None, spans: list[dict] | None,
                       model_b=None, score_fn: ScoreFn | None = None,
                       threshold: float = MODEL_B_SAFETY_THRESHOLD,
                       budget_ms: float = 1500, k: int = 2) -> dict | None:
    """
    Live call. One line for the listener, or None -> the caller withholds.
    {"text", "method": "model_b"|"masked", "safety_check_hate_prob", "ms"}
    """
    t0 = time.perf_counter()
    if model_b is not None and score_fn is not None:
        from model_b_generation.infer_gen import generate_candidates
        cands = generate_candidates(model_b, [build_prompt("rewrite", language, text, target)], k,
                                    max_new_tokens=48, max_time=budget_ms / 1000)[0]
        safe = _safe(cands, text, spans, score_fn, threshold)
        if safe:
            return {"text": safe[0][0], "method": "model_b",
                    "safety_check_hate_prob": round(safe[0][1], 4),
                    "ms": round((time.perf_counter() - t0) * 1000, 1)}
    masked = masked_rewrite(text, spans, score_fn, threshold)
    if masked:
        return {"text": masked[0], "method": "masked", "safety_check_hate_prob": masked[1],
                "ms": round((time.perf_counter() - t0) * 1000, 1)}
    return None
