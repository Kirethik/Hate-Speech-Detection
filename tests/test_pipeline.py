
import os
os.environ["CIVITAS_MOCK"] = "1"
from pipeline.analyze import analyze_text
from pipeline.registry import ModelRegistry

def test_pipeline_bronzites():
    reg = ModelRegistry()
    res = analyze_text("The Bronzites are a plague", registry=reg)
    assert res["detection"]["is_hate"] == True

def test_pipeline_identity():
    reg = ModelRegistry()
    res = analyze_text("I am a muslim woman and I am proud", registry=reg)
    assert res["detection"]["is_hate"] == False

def test_pipeline_timing():
    res = analyze_text("test")
    assert res["timing_ms"]["model_a"] > 0
