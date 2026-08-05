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

Usage:
    python infer.py --text "some text to check"
    python infer.py                       # interactive, Ctrl-D to quit
    python infer.py --nli                 # enable the NLI second stage
    python infer.py --device cpu          # when a training run owns the GPU
"""

import argparse

import torch
from transformers import XLMRobertaTokenizerFast

from dataset import SEVERITY_CLASSES, TARGET_CLASSES, normalize_code_mixed
from dehumanization import explain as explain_dehumanization
from dehumanization import is_dehumanizing_claim
from model import CivitasDetector


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(ckpt["args"]["encoder_name"])
    model = CivitasDetector(
        encoder_name=ckpt["args"]["encoder_name"],
        num_target_classes=len(TARGET_CLASSES),
        num_severity_classes=len(SEVERITY_CLASSES),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt["threshold"]


@torch.no_grad()
def predict(model, tokenizer, threshold, text, device, nli=None):
    cleaned = normalize_code_mixed(str(text))
    enc = tokenizer(cleaned, truncation=True, max_length=128,
                    padding=True, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    p_abuse = torch.softmax(out["hate_logits"].float(), -1)[0, 1].item()
    result = {
        "text": text,
        "p_abuse": p_abuse,
        "label": "ABUSIVE" if p_abuse >= threshold else "clean",
        "threshold": threshold,
        "severity": SEVERITY_CLASSES[out["severity_logits"].float().argmax(-1).item()],
        "target": TARGET_CLASSES[out["target_logits"].float().argmax(-1).item()],
        "reasons": [],
    }

    # Stage 2 — lexicon. Reported as a REASON, and allowed to escalate a clean
    # verdict, because a dehumanising metaphor predicated of a group is the one
    # pattern the encoder provably misses on unseen groups.
    if is_dehumanizing_claim(text):
        result["reasons"].append(explain_dehumanization(text))
        if result["label"] == "clean":
            result["label"] = "ABUSIVE"
            result["escalated_by"] = "lexicon"

    # Stage 3 — NLI, only inside the uncertainty band (see nli_stage.py).
    if nli is not None:
        from nli_stage import cascade_decision
        decision = cascade_decision(p_abuse, threshold, text, nli)
        result["nli_score"] = decision["nli_score"]
        if decision["changed_by_nli"] and result["label"] == "clean":
            result["label"] = "ABUSIVE"
            result["escalated_by"] = "nli"
            result["reasons"].append(
                f"NLI entails '{decision['nli_hypothesis']}' ({decision['nli_score']:.2f})"
            )
    return result


def print_result(r):
    escalated = f"  [escalated by {r['escalated_by']}]" if r.get("escalated_by") else ""
    nli = f"  nli={r['nli_score']:.2f}" if r.get("nli_score") is not None else ""
    print(f"  -> {r['label']}  (p={r['p_abuse']:.3f}, threshold={r['threshold']:.2f})"
          f"  severity={r['severity']}  target={r['target']}{nli}{escalated}")
    for reason in r["reasons"]:
        print(f"     reason: {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model_a/best_model.pt")
    ap.add_argument("--text", default=None, help="score a single string and exit")
    ap.add_argument("--nli", action="store_true",
                    help="enable the NLI second stage (downloads mDeBERTa on first use)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="use cpu when a training run owns the GPU")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    model, tokenizer, threshold = load_model(args.checkpoint, device)

    nli = None
    if args.nli:
        from nli_stage import NLIStage
        nli = NLIStage(device=device)

    print(f"loaded {args.checkpoint} on {device} (threshold {threshold:.2f})"
          f"{'  + NLI stage' if nli else ''}\n")

    if args.text is not None:
        print_result(predict(model, tokenizer, threshold, args.text, device, nli))
        return

    print("Enter text to score (Ctrl-D to quit):")
    try:
        while True:
            line = input("> ")
            if line.strip():
                print_result(predict(model, tokenizer, threshold, line, device, nli))
    except EOFError:
        print()


if __name__ == "__main__":
    main()
