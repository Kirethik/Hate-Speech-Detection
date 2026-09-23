"""
Single source of truth for all label class definitions and per-source mappings.
Import from here, never redefine elsewhere.
"""

TARGET_CLASSES = [
    "none", "religion", "gender", "caste_ethnicity",
    "disability", "political", "nationality_migrant", "other",
]
SEVERITY_CLASSES = ["normal", "offensive_profanity", "hate"]

# Per-source target label → TARGET_CLASSES string mapping
# -1 means unknown/unmappable for this source
HATEXPLAIN_TARGET_MAP = {
    "None": "none", "African": "caste_ethnicity", "Caucasian": "caste_ethnicity",
    "Asian": "caste_ethnicity", "Hispanic": "caste_ethnicity", "Arab": "caste_ethnicity",
    "Indian": "caste_ethnicity", "Indigenous": "caste_ethnicity", "Minority": "caste_ethnicity",
    "Islam": "religion", "Jewish": "religion", "Christian": "religion",
    "Hindu": "religion", "Buddhism": "religion", "Nonreligious": "religion",
    "Women": "gender", "Men": "gender", "Homosexual": "gender",
    "Heterosexual": "gender", "Bisexual": "gender", "Asexual": "gender",
    "Refugee": "nationality_migrant", "Disability": "disability",
    "Economic": "other", "Other": "other",
}

DRAVIDIAN_TARGET_MAP = {
    "Individual": "none",  # individual-targeted → severity head handles it
    "Group": "other",      # group-targeted but no finer category given
    "Other": "other",
    "None": "none",
}

HASOC_SEVERITY_MAP = {
    # task2 labels: HATE, OFFN, PRFN, NONE
    "HATE": "hate",
    "OFFN": "offensive_profanity",
    "PRFN": "offensive_profanity",
    "NONE": "normal",
    # HOF/NOT from task1
    "HOF": None,   # hate-or-offensive; resolve via task2 if available, else None
    "NOT": "normal",
}

RUHSOLD_SEVERITY_MAP = {
    "Normal": "normal",
    "Abusive": "offensive_profanity",
    "Offensive": "offensive_profanity",
    "Hate Speech": "hate",
    "Extreme": "hate",
}

SBIC_TARGET_MAP = {
    # SBIC whoTarget field — a free-text field, map common values
    "women": "gender", "men": "gender", "lgbtq": "gender", "gay": "gender",
    "black": "caste_ethnicity", "white": "caste_ethnicity", "asian": "caste_ethnicity",
    "hispanic": "caste_ethnicity", "latinx": "caste_ethnicity", "jewish": "religion",
    "muslim": "religion", "christian": "religion", "refugee": "nationality_migrant",
    "immigrant": "nationality_migrant", "disabled": "disability", "disability": "disability",
}

TOXIGEN_TARGET_MAP = {
    "black": "caste_ethnicity", "asian": "caste_ethnicity", "native_american": "caste_ethnicity",
    "latino": "caste_ethnicity", "jewish": "religion", "muslim": "religion",
    "lgbtq": "gender", "women": "gender", "mental_dis": "disability",
    "physical_dis": "disability",
}


def map_target(label: str, source_map: dict) -> int:
    """Map a raw source target label to TARGET_CLASSES index. Returns -1 if unknown."""
    mapped = source_map.get(label)
    if mapped is None:
        return -1
    try:
        return TARGET_CLASSES.index(mapped)
    except ValueError:
        return -1


def map_severity(label: str, source_map: dict) -> int:
    """Map a raw source severity label to SEVERITY_CLASSES index. Returns -1 if unknown."""
    mapped = source_map.get(label)
    if mapped is None:
        return -1
    try:
        return SEVERITY_CLASSES.index(mapped)
    except ValueError:
        return -1
