
class TTSBackend:
    def is_available(self) -> bool: return False
    def speak(self, text: str, language: str) -> bytes: raise NotImplementedError
