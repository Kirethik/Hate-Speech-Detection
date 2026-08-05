"""
NLI entailment second stage — generalising to groups the detector never saw.

WHY A SECOND MODEL
------------------
The fine-tuned encoder learns "which strings co-occur with abuse in these five
corpora". That is why an invented group ("Bronzites") in a dehumanising frame
scored clean: no training row ever paired that token with hate. An NLI model
reasons about ENTAILMENT between a premise and a hypothesis, compositionally,
so "The Bronzites are a plague" entails "a group of people is being described as
subhuman" whether or not it has ever met the word "Bronzites". That is precisely
the property the detector lacks, which is why this complements it rather than
duplicating it.

WHY ONLY ON A BAND
------------------
Running a second 280M-parameter model on every input triples inference cost to
change very few decisions: where the detector is confident it is usually right
(p=0.94 crude abuse, p=0.002 benign). The uncertainty is concentrated in the
middle — the trained model put activism at 0.645 against a 0.71 threshold, one
nudge from a false positive. So the cascade only consults NLI when the detector's
probability falls inside [low, high), and reports how often that happens.

CAVEAT
------
mDeBERTa-xnli is multilingual, but its NLI training is dominated by English and
the hypothesis templates below are English. Expect it to help most on English
and least on Romanised Indic text. Measure before trusting it on the Indic
slice — do not assume the multilingual label means uniform quality.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Hypotheses are phrased about "a group of people" rather than naming groups, so
# nothing here depends on knowing which groups exist. Each targets a distinct
# mechanism, so a text need only entail one to be flagged.
HYPOTHESES = [
    "This text describes a group of people as subhuman, vermin, or a disease.",
    "This text expresses hatred or contempt toward a group of people.",
    "This text says a group of people is inferior or worthless.",
    "This text calls for a group of people to be harmed, removed, or excluded.",
]


class NLIStage:
    """Zero-shot entailment scorer over the dehumanisation hypotheses."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Label order differs between NLI checkpoints; read it off the config
        # rather than assuming index 0 is entailment.
        self.entail_idx = next(
            (i for i, lab in self.model.config.id2label.items()
             if str(lab).lower().startswith("entail")),
            0,
        )

    @torch.no_grad()
    def entailment_scores(self, text: str) -> dict[str, float]:
        pairs = [(text, h) for h in HYPOTHESES]
        enc = self.tokenizer(
            [p for p, _ in pairs], [h for _, h in pairs],
            truncation=True, max_length=256, padding=True, return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1)[:, self.entail_idx]
        return {h: float(p) for h, p in zip(HYPOTHESES, probs.cpu())}

    def max_entailment(self, text: str) -> tuple[float, str]:
        """Highest entailment probability across hypotheses, and which one."""
        scores = self.entailment_scores(text)
        best = max(scores, key=scores.get)
        return scores[best], best


def cascade_decision(
    p_abuse: float,
    threshold: float,
    text: str,
    nli: "NLIStage | None",
    band: tuple[float, float] = (0.25, 0.85),
    nli_threshold: float = 0.70,
) -> dict:
    """
    Combine the detector's probability with NLI, consulting NLI only inside the
    uncertainty band.

    Outside the band the detector's own decision stands unchanged, so this can
    only ever alter borderline calls — it cannot silently overturn a confident
    correct answer.
    """
    base = {"p_abuse": p_abuse, "threshold": threshold,
            "label": "ABUSIVE" if p_abuse >= threshold else "clean",
            "nli_consulted": False, "nli_score": None, "nli_hypothesis": None,
            "changed_by_nli": False}

    low, high = band
    if nli is None or not (low <= p_abuse < high):
        return base

    score, hypothesis = nli.max_entailment(text)
    base.update(nli_consulted=True, nli_score=score, nli_hypothesis=hypothesis)

    if score >= nli_threshold and base["label"] == "clean":
        base["label"] = "ABUSIVE"
        base["changed_by_nli"] = True
    return base
