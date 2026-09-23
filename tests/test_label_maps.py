import pytest
from label_maps import map_target, map_severity, HATEXPLAIN_TARGET_MAP, TARGET_CLASSES, SEVERITY_CLASSES

def test_map_target_valid():
    assert map_target("Islam", HATEXPLAIN_TARGET_MAP) == TARGET_CLASSES.index("religion")

def test_map_target_unknown():
    assert map_target("UnknownLabel", HATEXPLAIN_TARGET_MAP) == -1

def test_map_severity():
    from label_maps import HASOC_SEVERITY_MAP
    assert map_severity("HATE", HASOC_SEVERITY_MAP) == SEVERITY_CLASSES.index("hate")
    assert map_severity("Unknown", HASOC_SEVERITY_MAP) == -1

def test_all_hatexplain_targets_exist():
    for v in HATEXPLAIN_TARGET_MAP.values():
        if v != "none":
            assert v in TARGET_CLASSES
