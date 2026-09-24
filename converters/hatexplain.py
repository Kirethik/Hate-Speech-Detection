"""
Converts hate-alert/HateXplain (data/raw/HateXplain/Data/dataset.json) to the
unified schema. This is the only source with all four label types (hate,
target, severity, rationale), so it doubles as the "rationale anchor" the
README calls out.

Label mapping:
  - severity_label: HateXplain's 3-way post label maps 1:1 onto
    SEVERITY_CLASSES (normal/offensive_profanity/hate), taking the majority
    vote across the 3 annotators (ties broken toward the more severe class,
    since under-calling hate speech is the worse failure mode here).
  - hate_label: 0 if majority label is "normal", else 1.
  - target_label: each annotator's target community list is mapped onto our
    8-class TARGET_CLASSES via TARGET_MAP below, then majority-voted. Always
    present (HateXplain annotates target — including "None" — for every
    post), so target_mask is always 1 for this source.
  - rationale_spans: only exists when the post isn't "normal" (that's how
    HateXplain collected annotations). post_tokens are joined with single
    spaces to build `text`, and a token is "rationale" if at least half of
    the annotators who supplied a rationale array marked it.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from label_maps import TARGET_CLASSES, SEVERITY_CLASSES, HATEXPLAIN_TARGET_MAP  # noqa: E402


LABEL_TO_SEVERITY = {"normal": "normal", "offensive": "offensive_profanity", "hatespeech": "hate"}
LABEL_PRIORITY = ["hatespeech", "offensive", "normal"]  # more severe first, for tie-breaks


def _majority(values, priority):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best_count = max(counts.values())
    tied = [v for v in priority if counts.get(v, 0) == best_count]
    return tied[0] if tied else max(counts, key=counts.get)


def _majority_target(target_lists):
    mapped = []
    for targets in target_lists:
        for t in targets:
            mapped.append(HATEXPLAIN_TARGET_MAP.get(t, "other"))
    if not mapped:
        return "none"
    return _majority(mapped, ["none"] + [c for c in TARGET_CLASSES if c != "none"])


def convert(raw_dir="data/raw"):
    root = Path(raw_dir) / "HateXplain" / "Data"
    data = json.load(open(root / "dataset.json", encoding="utf-8"))
    splits = json.load(open(root / "post_id_divisions.json", encoding="utf-8"))
    split_of = {}
    for post_id in splits["train"]:
        split_of[post_id] = "train"
    for post_id in splits["val"] + splits["test"]:
        split_of[post_id] = "val"

    rows = []
    for post_id, rec in data.items():
        labels = [a["label"] for a in rec["annotators"]]
        majority_label = _majority(labels, ["hatespeech", "offensive", "normal"])
        hate_label = 0 if majority_label == "normal" else 1
        severity_label = SEVERITY_CLASSES.index(LABEL_TO_SEVERITY[majority_label])
        target_label = TARGET_CLASSES.index(
            _majority_target([a["target"] for a in rec["annotators"]])
        )

        text = " ".join(rec["post_tokens"])
        rationale_spans = []
        if rec["rationales"]:
            n = len(rec["rationales"])
            length = len(rec["post_tokens"])
            votes = [0] * length
            for arr in rec["rationales"]:
                for i, v in enumerate(arr[:length]):
                    votes[i] += v
            offsets = []
            cursor = 0
            for tok in rec["post_tokens"]:
                offsets.append((cursor, cursor + len(tok)))
                cursor += len(tok) + 1  # +1 for the joining space
            for i, v in enumerate(votes):
                if v * 2 >= n:  # majority of rationale annotators flagged this token
                    rationale_spans.append(list(offsets[i]))

        rows.append(
            {
                "text": text,
                "language": "en",
                "hate_label": hate_label,
                "target_label": target_label,
                "severity_label": severity_label,
                "rationale_spans": json.dumps(rationale_spans),
                "source": "hatexplain",
                "split": split_of.get(post_id, "train"),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = convert()
    print(df.shape, df["split"].value_counts().to_dict())
    print(df.head(3))
