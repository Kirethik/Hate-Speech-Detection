
from model_b_generation.generate import generate_suggestions
def test_generate_mock():
    res = generate_suggestions("text", "en", "unknown", None, None)
    assert res['fallback'] == True
