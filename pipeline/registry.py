
from config import mock_mode

class ModelRegistry:
    def __init__(self, cfg=None):
        pass
    def get_model_a(self):
        if mock_mode():
            return "mock_a"
        return None
    def get_nli(self):
        return None
    def get_model_b(self):
        return None
    def vram_report(self) -> dict:
        return {}
    def mock_mode(self) -> bool:
        return mock_mode()
