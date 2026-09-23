
import os
SILERO_VAD_THRESHOLD = float(os.getenv("SILERO_VAD_THRESHOLD", "0.5"))
SILERO_MIN_SPEECH_MS = int(os.getenv("SILERO_MIN_SPEECH_MS", "250"))
SILERO_MIN_SILENCE_MS = int(os.getenv("SILERO_MIN_SILENCE_MS", "500"))

class VAD:
    def __init__(self):
        self._model = None
    
    def _load(self):
        pass
    
    def get_segments(self, audio_path: str) -> list[dict]:
        return []
    
    def is_available(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False
