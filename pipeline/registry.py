
from pipeline.config import MOCK_MODE

class ModelRegistry:
    def __init__(self, cfg=None):
        pass
    def get_model_a(self):
        if MOCK_MODE:
            return "mock_a"
        return None
    def get_nli(self):
        return None
    def get_model_b(self):
        return None
    def vram_report(self) -> dict:
        return {}
    def mock_mode(self) -> bool:
        return MOCK_MODE
