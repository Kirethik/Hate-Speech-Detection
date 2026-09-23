
class StreamSession:
    def __init__(self, lang_hint=None, registry=None):
        pass
    async def push_chunk(self, pcm_bytes: bytes) -> None:
        pass
    async def next_result(self) -> dict | None:
        return None
    async def close(self) -> list[dict]:
        return []
