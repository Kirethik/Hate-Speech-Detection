"""
Converts the Implicit Hate Corpus / "Latent Hatred" (ElSherief et al., EMNLP
2021) to the unified schema.

This is the single most on-target source for the failure this project hit:
"The Bronzites are a plague on every town they enter" scored p=0.28 (clean),
because every existing source teaches abuse-via-profanity. Every row here is
hate expressed WITHOUT a slur or profanity — exactly the distribution that was
missing.

The six implicit_class values are the *rhetorical mechanism* of the hate
(white_grievance, incitement, stereotypical, inferiority, irony, threatening),
not the targeted group, so they do NOT populate target_label — they are
recorded in the source name only. Target stays -1 (masked).

Crucially these rows carry severity_label = "hate" rather than
"offensive_profanity". The whole point of the severity head is to separate
"rude" from "hateful", and before this source only 6,704 rows in 190k taught
that distinction at all.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

from label_maps import TARGET_CLASSES, SEVERITY_CLASSES

from ._hf import cap, hash_split

REPO = "SALT-NLP/ImplicitHate"
FILENAME = "implicit_hate.csv"


def convert(raw_dir="data/raw", max_rows: int | None = None):
    path = hf_hub_download(REPO, FILENAME, repo_type="dataset")
    df = pd.read_csv(path).dropna(subset=["post"])

    rows = []
    for _, r in df.iterrows():
        text = str(r["post"]).strip()
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "language": "en",
                "hate_label": 1,  # every row in this file is implicit hate
                "target_label": -1,
                "severity_label": SEVERITY_CLASSES.index("hate"),
                "rationale_spans": "[]",
                "source": "implicithate",
                "split": hash_split(text),
            }
        )
    out = cap(pd.DataFrame(rows), max_rows)
    return out


if __name__ == "__main__":
    d = convert()
    print(d.shape, d["split"].value_counts().to_dict())
    print(d.head(3).to_string())
