"""
Run the trained Model A detector on arbitrary text.

Three stages, each covering a different failure mode of the one before it:

  1. The fine-tuned encoder      — fast, multilingual, but keyed on the abuse
                                   vocabulary it saw in training.
  2. The dehumanisation lexicon  — interpretable, group-agnostic; catches
                                   "<unknown group> are a plague" that the
                                   encoder misses. See dehumanization.py.
  3. NLI entailment (optional)   — compositional reasoning for groups never seen
                                   in training, consulted only on borderline
                                   scores. See nli_stage.py.

This module is the ONE place Model A is loaded and scored; the pipeline, the
Model B safety gate and the eval scripts all go through load_model() and
score_batch().

Usage:
    python infer.py --text "some text to check"
    python infer.py                       # interactive, Ctrl-D to quit
    python infer.py --nli                 # enable the NLI second stage
    python infer.py --device cpu          # when a training run owns the GPU
"""

import argparse
import warnings
from dataclasses import dataclass

import torch
from transformers import XLMRobertaTokenizerFast

from dehumanization import explain as explain_dehumanization
from dehumanization import is_dehumanizing_claim
from label_maps import SEVERITY_CLASSES, TARGET_CLASSES
from model import CivitasDetector
from text_norm import normalize_code_mixed, to_original_span

# Must match train.NORMALIZATION_VERSION for the checkpoint to see the text the
# way it was trained. Duplicated (not imported) so inference never imports train.py.
NORMALIZATION_VERSION = 2
RATIONALE_THRESHOLD = 0.5


@dataclass
class ModelA:
    model: CivitasDetector
    tokenizer: XLMRobertaTokenizerFast
    threshold: float
    target_classes: list
    severity_classes: list
    device: torch.device
    max_length: int = 128

    # legacy tuple unpacking: model, tokenizer, threshold = load_model(...)
    def __iter__(self):
        return iter((self.model, self.tokenizer, self.threshold))

    def __getitem__(self, i):
        return (self.model, self.tokenizer, self.threshold)[i]


def load_model(checkpoint_path: str, device, half: bool | None = None) -> ModelA:
    """
    Load a train.py checkpoint. Class lists come from the checkpoint itself,
    so a model trained with different classes can never be silently mislabelled.
    half: fp16 weights (default: on for CUDA). Halves VRAM (~1.1 GB -> ~0.55 GB).
    """
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = ckpt["args"]["encoder_name"]
    target_classes = ckpt.get("target_classes", TARGET_CLASSES)
    severity_classes = ckpt.get("severity_classes", SEVERITY_CLASSES)
    if ckpt.get("normalization_version", 1) != NORMALIZATION_VERSION:
        warnings.warn(
            f"{checkpoint_path} was trained with normalization v{ckpt.get('normalization_version', 1)}, "
            f"inference uses v{NORMALIZATION_VERSION}. Scores will be off until you retrain.")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(encoder)
    model = CivitasDetector(encoder_name=encoder,
                            num_target_classes=len(target_classes),
                            num_severity_classes=len(severity_classes))
    model.load_state_dict(ckpt["model_state_dict"])
    if half is None:
        half = device.type == "cuda"
    if half:
        model.half()
    model.to(device).eval()
    return ModelA(model, tokenizer, float(ckpt["threshold"]), list(target_classes),
                  list(severity_classes), device,
                  int(ckpt["args"].get("max_length", 128)))


def _rationale_spans(text: str, token_offsets, token_scores, offset_map) -> list[dict]:
    """
    Token rationale probabilities -> merged character spans on the ORIGINAL text.
    Subword pieces are widened to whole words so the UI never underlines half a word.
    """
    hits = []
    for (s, e), p in zip(token_offsets, token_scores):
        if s == e or p < RATIONALE_THRESHOLD:
            continue
        o_s, o_e = to_original_span(s, e, offset_map, text)
        while o_s > 0 and text[o_s - 1].isalnum():
            o_s -= 1
        while o_e < len(text) and text[o_e].isalnum():
            o_e += 1
        hits.append([o_s, o_e, float(p)])
    hits.sort()
    merged = []
    for s, e, p in hits:
        # merge overlapping spans and spans separated only by whitespace
        if merged and s <= merged[-1][1] + 1 and not text[merged[-1][1]:s].strip():
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], p)
        else:
            merged.append([s, e, p])
    return [{"start_char": s, "end_char": e, "text": text[s:e], "score": round(p, 4)}
            for s, e, p in merged]


@torch.no_grad()
def score_batch(m: ModelA, texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Score texts with every head. Each result:
      hate_prob, is_hate, severity{label, probs}, target{label, probs},
      rationale[{start_char, end_char, text, score}], normalized_text
    """
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[i:i + batch_size]]
        normed = [normalize_code_mixed(t) for t in chunk]
        enc = m.tokenizer([n for n, _ in normed], truncation=True, max_length=m.max_length,
                          padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(m.device) for k, v in enc.items()}
        out = m.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        hate = torch.softmax(out["hate_logits"].float(), -1)[:, 1].cpu().tolist()
        sev = torch.softmax(out["severity_logits"].float(), -1).cpu().tolist()
        tgt = torch.softmax(out["target_logits"].float(), -1).cpu().tolist()
        rat = torch.softmax(out["rationale_logits"].float(), -1)[..., 1].cpu().tolist()

        for j, text in enumerate(chunk):
            sev_probs = dict(zip(m.severity_classes, sev[j]))
            tgt_probs = dict(zip(m.target_classes, tgt[j]))
            results.append({
                "hate_prob": hate[j],
                "is_hate": hate[j] >= m.threshold,
                "severity": {"label": max(sev_probs, key=sev_probs.get), "probs": sev_probs},
                "target": {"label": max(tgt_probs, key=tgt_probs.get), "probs": tgt_probs},
                "rationale": _rationale_spans(text, offsets[j], rat[j], normed[j][1]),
                "normalized_text": normed[j][0],
            })
    return results


def predict(model, tokenizer, threshold, text, device=None, nli=None):
    """
    Legacy single-text API used by model_b_generation (keys: p_abuse, label,
    severity, target, reasons). `model` may also be a ModelA, in which case the
    other positional arguments are ignored.
    """
    m = model if isinstance(model, ModelA) else ModelA(
        model, tokenizer, threshold, TARGET_CLASSES, SEVERITY_CLASSES,
        device or next(model.parameters()).device)
    r = score_batch(m, [text])[0]
    result = {
        "text": text,
        "p_abuse": r["hate_prob"],
        "label": "ABUSIVE" if r["is_hate"] else "clean",
        "threshold": m.threshold,
        "severity": r["severity"]["label"],
        "target": r["target"]["label"],
        "rationale": r["rationale"],
        "reasons": [],
    }

    # Stage 2 — lexicon. Reported as a REASON, and allowed to escalate a clean
    # verdict, because a dehumanising metaphor predicated of a group is the one
    # pattern the encoder provably misses on unseen groups. Checked on the
    # normalised text too, so "p e o p l e are v3rmin" cannot slip past it.
    lex_text = next((t for t in (text, r["normalized_text"]) if is_dehumanizing_claim(t)), None)
    if lex_text is not None:
        result["reasons"].append(explain_dehumanization(lex_text))
        if result["label"] == "clean":
            result["label"] = "ABUSIVE"
            result["escalated_by"] = "lexicon"

    # Stage 3 — NLI, only inside the uncertainty band (see nli_stage.py).
    if nli is not None:
        from nli_stage import cascade_decision
        decision = cascade_decision(r["hate_prob"], m.threshold, text, nli)
        result["nli_score"] = decision["nli_score"]
        if decision["changed_by_nli"] and result["label"] == "clean":
            result["label"] = "ABUSIVE"
            result["escalated_by"] = "nli"
            result["reasons"].append(
                f"NLI entails '{decision['nli_hypothesis']}' ({decision['nli_score']:.2f})")
    return result


def print_result(r):
    escalated = f"  [escalated by {r['escalated_by']}]" if r.get("escalated_by") else ""
    nli = f"  nli={r['nli_score']:.2f}" if r.get("nli_score") is not None else ""
    print(f"  -> {r['label']}  (p={r['p_abuse']:.3f}, threshold={r['threshold']:.2f})"
          f"  severity={r['severity']}  target={r['target']}{nli}{escalated}")
    for span in r.get("rationale", []):
        print(f"     rationale: {span['text']!r} ({span['score']:.2f})")
    for reason in r["reasons"]:
        print(f"     reason: {reason}")


def main():
    from config import MODEL_A_PT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=MODEL_A_PT_PATH)
    ap.add_argument("--text", default=None, help="score a single string and exit")
    ap.add_argument("--nli", action="store_true",
                    help="enable the NLI second stage (downloads mDeBERTa on first use)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="use cpu when a training run owns the GPU")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    m = load_model(args.checkpoint, device)

    nli = None
    if args.nli:
        from nli_stage import NLIStage
        nli = NLIStage(device=device)

    print(f"loaded {args.checkpoint} on {device} (threshold {m.threshold:.2f})"
          f"{'  + NLI stage' if nli else ''}\n")

    if args.text is not None:
        print_result(predict(m, None, None, args.text, nli=nli))
        return

    print("Enter text to score (Ctrl-D to quit):")
    try:
        while True:
            line = input("> ")
            if line.strip():
                print_result(predict(m, None, None, line, nli=nli))
    except EOFError:
        print()


if __name__ == "__main__":
    main()
