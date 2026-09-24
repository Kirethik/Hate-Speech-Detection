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
    # The 5 fine-grained RUHSOLD classes, exactly as converters/ruhsold.py names them
    "Normal": "normal",
    "Profane/Untargeted": "offensive_profanity",
    "Abusive/Offensive": "offensive_profanity",
    "Religious Hate": "hate",
    "Sexism": "hate",
}

SBIC_TARGET_MAP = {
    # SBIC targetCategory values (casefolded). This is the only source of
    # "political" supervision in the corpus (via "social").
    "race": "caste_ethnicity",
    "gender": "gender",
    "culture": "religion",       # SBIC's culture bucket is religion/culture
    "disabled": "disability",
    "social": "political",
    "body": "other",
    "victim": "other",
}

TOXIGEN_TARGET_MAP = {
    # ToxiGen target_group values (casefolded, spaces/hyphens -> "_").
    # converters/toxigen.py also does substring matching ("black folks" -> black).
    "jewish": "religion",
    "muslim": "religion",
    "women": "gender",
    "trans": "gender",
    "lgbtq": "gender",
    "bisexual": "gender",
    "black": "caste_ethnicity",
    "asian": "caste_ethnicity",
    "chinese": "caste_ethnicity",
    "latino": "caste_ethnicity",
    "mexican": "caste_ethnicity",
    "native_american": "caste_ethnicity",
    "middle_east": "caste_ethnicity",
    "mental_dis": "disability",   # also matches "mental_disability"
    "physical_dis": "disability",  # also matches "physical_disability"
    "immigrant": "nationality_migrant",
}

# CONSTRAINT 2021 Hindi hostility. Posts carry a comma-separated label set.
# Resolved in converters/constraint2021.py by priority: hate > offensive/defamation
# > non-hostile. "fake" alone is misinformation, not abuse -> row is dropped.
CONSTRAINT_SEVERITY_MAP = {
    "hate": "hate",
    "offensive": "offensive_profanity",
    "defamation": "offensive_profanity",  # attacks an individual's reputation, not a group
    "non-hostile": "normal",
    "fake": None,
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
