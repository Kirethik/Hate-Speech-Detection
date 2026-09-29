"""
Generation metrics shared by train_gen.py (val, for checkpoint selection) and
evaluate_gen.py (test, RUN ONCE).

  chrF        character n-gram F-score vs the reference (works for every script,
              unlike word BLEU on agglutinative Tamil/Malayalam). 0-100.
  bleu        sacrebleu corpus BLEU if sacrebleu is installed, else None.
  copy_rate   share of outputs that are (almost) the input. A rewrite model that
              just echoes the hate back scores well on chrF but is useless.
  safety_rate share of outputs Model A scores below its threshold (needs a scorer).
  distinct_2  unique bigrams / all bigrams (diversity).
"""

import difflib
import re
from collections import Counter

_WS = re.compile(r"\s+")


def _ngrams(s: str, n: int) -> Counter:
    s = _WS.sub(" ", s.strip())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def corpus_chrf(hyps, refs) -> float:
    return sum(chrf(h, r) for h, r in zip(hyps, refs)) / max(len(hyps), 1)


def corpus_bleu(hyps, refs) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    return float(sacrebleu.corpus_bleu(list(hyps), [list(refs)]).score)


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s).casefold()).strip()


def is_copy(output: str, source: str, threshold: float = 0.9) -> bool:
    a, b = _norm(output), _norm(source)
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def copy_rate(outputs, sources) -> float:
    pairs = list(zip(outputs, sources))
    return sum(is_copy(o, s) for o, s in pairs) / max(len(pairs), 1)


def distinct_2(texts) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = str(t).split()
        for g in zip(toks, toks[1:]):
            grams.add(g)
            total += 1
    return len(grams) / total if total else 0.0


def safety_rate(outputs, score_fn, threshold: float) -> float | None:
    """score_fn(list[str]) -> list[float] hate probabilities."""
    if score_fn is None or not outputs:
        return None
    probs = score_fn(list(outputs))
    return sum(p < threshold for p in probs) / len(probs)


def report(outputs, refs, sources, groups=None, score_fn=None, threshold=0.5) -> dict:
    """Overall metrics plus the same metrics per group key (e.g. "rewrite|ta")."""
    def one(o, r, s):
        return {"n": len(o), "chrf": round(corpus_chrf(o, r), 2), "bleu": corpus_bleu(o, r),
                "copy_rate": round(copy_rate(o, s), 4), "distinct_2": round(distinct_2(o), 4),
                "safety_rate": safety_rate(o, score_fn, threshold)}

    out = one(outputs, refs, sources)
    if groups is not None:
        by = {}
        for g in sorted(set(groups)):
            idx = [i for i, x in enumerate(groups) if x == g]
            by[g] = one([outputs[i] for i in idx], [refs[i] for i in idx], [sources[i] for i in idx])
        out["by_group"] = by
    return out
